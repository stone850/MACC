from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch as th
import yaml

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot


FORMAT_VERSION = 1
TRANSITION_FIELDS = (
    "state",
    "obs",
    "actions",
    "avail_actions",
    "reward",
    "terminated",
    "filled",
    "subtask_state",
    "subtask_obs",
    "subtask_visible",
    "subtask_mask",
)
EPISODE_FIELDS = ("subtask_id",)

_DTYPE_TO_NAME = {
    th.float16: "float16",
    th.float32: "float32",
    th.float64: "float64",
    th.int8: "int8",
    th.uint8: "uint8",
    th.int16: "int16",
    th.int32: "int32",
    th.int64: "int64",
    th.bool: "bool",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _as_vshape(value):
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _field_shape(field_info, groups, max_seq_length, transition):
    shape = []
    if transition:
        shape.append(max_seq_length)
    if "group" in field_info:
        shape.append(groups[field_info["group"]])
    shape.extend(_as_vshape(field_info["vshape"]))
    return shape


def _build_schema_manifest(scheme, groups, max_seq_length):
    fields = {}
    for field in TRANSITION_FIELDS + EPISODE_FIELDS:
        if field == "filled":
            field_info = {"vshape": (1,), "dtype": th.long}
        else:
            if field not in scheme:
                raise KeyError("Required offline field '{}' is missing from the scheme".format(field))
            field_info = scheme[field]

        transition = field in TRANSITION_FIELDS
        entry = {
            "storage": "transition" if transition else "episode",
            "vshape": list(_as_vshape(field_info["vshape"])),
            "vshape_is_int": isinstance(field_info["vshape"], int),
            "dtype": _DTYPE_TO_NAME[field_info.get("dtype", th.float32)],
            "shape": _field_shape(field_info, groups, max_seq_length, transition),
        }
        if "group" in field_info:
            entry["group"] = field_info["group"]
        if field_info.get("episode_const", False):
            entry["episode_const"] = True
        fields[field] = entry

    return {
        "groups": dict(groups),
        "max_seq_length": int(max_seq_length),
        "fields": fields,
        "preprocess": {
            "actions": {
                "target": "actions_onehot",
                "transform": "one_hot",
                "out_dim": int(scheme["avail_actions"]["vshape"][0]),
            }
        },
    }


def _components_from_manifest(manifest):
    scheme = {}
    for field, entry in manifest["fields"].items():
        if field == "filled":
            continue
        vshape_values = tuple(entry["vshape"])
        vshape_is_int = entry.get("vshape_is_int", field in ("state", "obs") and len(vshape_values) == 1)
        field_info = {
            "vshape": vshape_values[0] if vshape_is_int else vshape_values,
            "dtype": _NAME_TO_DTYPE[entry["dtype"]],
        }
        if "group" in entry:
            field_info["group"] = entry["group"]
        if entry.get("episode_const", False):
            field_info["episode_const"] = True
        scheme[field] = field_info

    preprocess_info = manifest["preprocess"]["actions"]
    if preprocess_info["transform"] != "one_hot":
        raise ValueError("Unsupported offline preprocess {}".format(preprocess_info["transform"]))
    preprocess = {
        "actions": (
            preprocess_info["target"],
            [OneHot(out_dim=preprocess_info["out_dim"])],
        )
    }
    return scheme, dict(manifest["groups"]), preprocess


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(".{}.tmp".format(path.name))
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_yaml(path, value):
    temporary = path.with_name(".{}.tmp".format(path.name))
    with open(temporary, "w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
    os.replace(temporary, path)


def _split_episode_ids(n_episodes, ratios, seed):
    names = ("train", "validation", "test")
    weights = np.asarray([float(ratios[name]) for name in names], dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("split ratios must be non-negative and sum to 1")

    exact_counts = weights * n_episodes
    counts = np.floor(exact_counts).astype(np.int64)
    for index in np.argsort(-(exact_counts - counts))[:n_episodes - int(counts.sum())]:
        counts[index] += 1

    episode_ids = np.arange(n_episodes, dtype=np.int64)
    np.random.RandomState(seed).shuffle(episode_ids)
    splits = {}
    offset = 0
    for name, count in zip(names, counts):
        splits[name] = sorted(episode_ids[offset:offset + count].tolist())
        offset += count
    splits["all"] = list(range(n_episodes))
    return splits


class OfflineDatasetWriter:
    def __init__(
        self,
        dataset_path,
        scheme,
        groups,
        max_seq_length,
        metadata,
        shard_size=100,
        split_ratios=None,
        split_seed=0,
    ):
        self.path = Path(dataset_path).expanduser().resolve()
        if self.path.exists():
            raise FileExistsError("Dataset path already exists: {}".format(self.path))
        self.path.mkdir(parents=True)

        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.split_ratios = split_ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
        self.split_seed = int(split_seed)
        self.schema = _build_schema_manifest(scheme, groups, max_seq_length)
        self.base_metadata = dict(metadata)
        self.max_seq_length = int(max_seq_length)
        self.pending = []
        self.shards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.natural_terminations = 0
        self.n_episodes = 0
        self.finalized = False

    def add_episode(self, batch):
        if self.finalized:
            raise RuntimeError("Cannot add episodes after finalizing the dataset")
        if batch.batch_size != 1 or batch.max_seq_length != self.max_seq_length:
            raise ValueError("Writer expects one full episode with max_seq_length={}".format(self.max_seq_length))

        transition_data = {}
        for field in TRANSITION_FIELDS:
            tensor = batch.data.transition_data[field]
            transition_data[field] = tensor.detach().cpu().clone()
        episode_data = {}
        for field in EPISODE_FIELDS:
            tensor = batch.data.episode_data[field]
            episode_data[field] = tensor.detach().cpu().clone()

        filled_states = int(transition_data["filled"][0].sum().item())
        episode_length = filled_states - 1
        if episode_length <= 0 or episode_length > self.max_seq_length - 1:
            raise ValueError("Invalid episode length {}".format(episode_length))
        episode_return = float(transition_data["reward"][0, :episode_length].sum().item())
        naturally_terminated = bool(transition_data["terminated"][0, episode_length - 1].item())

        self.pending.append({
            "episode_id": self.n_episodes,
            "transition_data": transition_data,
            "episode_data": episode_data,
        })
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.natural_terminations += int(naturally_terminated)
        self.n_episodes += 1
        if len(self.pending) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        if not self.pending:
            return

        shard_index = len(self.shards)
        filename = "episodes_{:03d}.pt".format(shard_index)
        final_path = self.path / filename
        temporary_path = self.path / ".{}.tmp".format(filename)
        payload = {
            "format_version": FORMAT_VERSION,
            "episode_ids": th.tensor([item["episode_id"] for item in self.pending], dtype=th.long),
            "transition_data": {
                field: th.cat([item["transition_data"][field] for item in self.pending], dim=0)
                for field in TRANSITION_FIELDS
            },
            "episode_data": {
                field: th.cat([item["episode_data"][field] for item in self.pending], dim=0)
                for field in EPISODE_FIELDS
            },
        }
        th.save(payload, temporary_path)

        reloaded = th.load(temporary_path, map_location="cpu")
        self._assert_payload_equal(payload, reloaded)
        os.replace(temporary_path, final_path)
        self.shards.append({
            "file": filename,
            "count": len(self.pending),
            "first_episode_id": int(payload["episode_ids"][0].item()),
            "sha256": _sha256(final_path),
        })
        self.pending = []

    @staticmethod
    def _assert_payload_equal(expected, actual):
        if expected["format_version"] != actual["format_version"]:
            raise IOError("Shard format version changed during round trip")
        if not th.equal(expected["episode_ids"], actual["episode_ids"]):
            raise IOError("Episode IDs changed during shard round trip")
        for storage in ("transition_data", "episode_data"):
            if set(expected[storage]) != set(actual[storage]):
                raise IOError("Shard fields changed during round trip")
            for field in expected[storage]:
                if not th.equal(expected[storage][field], actual[storage][field]):
                    raise IOError("Field '{}' changed during shard round trip".format(field))

    def finalize(self):
        if self.finalized:
            raise RuntimeError("Dataset has already been finalized")
        if self.n_episodes == 0:
            raise ValueError("Cannot finalize an empty dataset")
        self._flush_shard()

        splits = _split_episode_ids(self.n_episodes, self.split_ratios, self.split_seed)
        statistics = {
            "n_episodes": self.n_episodes,
            "returns": {
                "mean": float(np.mean(self.episode_returns)),
                "std": float(np.std(self.episode_returns)),
                "min": float(np.min(self.episode_returns)),
                "max": float(np.max(self.episode_returns)),
            },
            "episode_lengths": {
                "mean": float(np.mean(self.episode_lengths)),
                "std": float(np.std(self.episode_lengths)),
                "min": int(np.min(self.episode_lengths)),
                "max": int(np.max(self.episode_lengths)),
            },
            "natural_terminations": self.natural_terminations,
            "time_limit_episodes": self.n_episodes - self.natural_terminations,
        }
        metadata = dict(self.base_metadata)
        metadata.update({
            "format_version": FORMAT_VERSION,
            "complete": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_episodes": self.n_episodes,
            "schema": self.schema,
            "shards": self.shards,
            "split_seed": self.split_seed,
            "split_ratios": {key: float(value) for key, value in self.split_ratios.items()},
        })

        _atomic_json(self.path / "statistics.json", statistics)
        _atomic_json(self.path / "split.json", splits)
        _atomic_yaml(self.path / "metadata.yaml", metadata)
        self.finalized = True
        return self.path


class OfflineEpisodeDataset:
    def __init__(
        self,
        dataset_path,
        split="train",
        seed=0,
        device="cpu",
        verify_checksums=False,
        cache_size=2,
        max_episodes=None,
    ):
        self.path = Path(dataset_path).expanduser().resolve()
        metadata_path = self.path / "metadata.yaml"
        if not metadata_path.is_file():
            raise FileNotFoundError("Complete dataset metadata not found: {}".format(metadata_path))
        with open(metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = yaml.safe_load(handle)
        if not self.metadata.get("complete", False) or self.metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported or incomplete offline dataset")

        with open(self.path / "split.json", "r", encoding="utf-8") as handle:
            splits = json.load(handle)
        if split not in splits:
            raise KeyError("Unknown dataset split '{}'; expected one of {}".format(split, sorted(splits)))

        self.split = split
        self.episode_ids = [int(value) for value in splits[split]]
        if max_episodes is not None:
            max_episodes = int(max_episodes)
            if max_episodes <= 0 or max_episodes > len(self.episode_ids):
                raise ValueError(
                    "max_episodes must be in [1, {}] for split '{}', got {}".format(
                        len(self.episode_ids), split, max_episodes
                    )
                )
            self.episode_ids = self.episode_ids[:max_episodes]
        self.rng = np.random.RandomState(seed)
        self.device = device
        self.schema_manifest = self.metadata["schema"]
        self._base_scheme, self.groups, self.preprocess = _components_from_manifest(self.schema_manifest)
        self.max_seq_length = int(self.schema_manifest["max_seq_length"])
        self.scheme = EpisodeBatch(
            self._base_scheme,
            self.groups,
            1,
            self.max_seq_length,
            preprocess=self.preprocess,
            device="cpu",
        ).scheme
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()
        self._episode_index = {}
        self._shards = self.metadata["shards"]

        for shard_index, shard in enumerate(self._shards):
            first_id = int(shard["first_episode_id"])
            for row in range(int(shard["count"])):
                episode_id = first_id + row
                if episode_id in self._episode_index:
                    raise ValueError("Duplicate episode ID {} in metadata".format(episode_id))
                self._episode_index[episode_id] = (shard_index, row)
        if set(self._episode_index) != set(range(int(self.metadata["n_episodes"]))):
            raise ValueError("Dataset episode index is not contiguous")
        if not set(self.episode_ids).issubset(self._episode_index):
            raise ValueError("Split references an unknown episode ID")
        if verify_checksums:
            self.verify_shards()

    def __len__(self):
        return len(self.episode_ids)

    @property
    def episodes_in_dataset(self):
        return len(self)

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def verify_shards(self):
        for shard in self._shards:
            path = self.path / shard["file"]
            if not path.is_file():
                raise FileNotFoundError("Missing dataset shard: {}".format(path))
            actual = _sha256(path)
            if actual != shard["sha256"]:
                raise IOError("Checksum mismatch for {}".format(path))

    def _load_shard(self, shard_index):
        if shard_index in self._cache:
            payload = self._cache.pop(shard_index)
            self._cache[shard_index] = payload
            return payload

        descriptor = self._shards[shard_index]
        payload = th.load(self.path / descriptor["file"], map_location="cpu")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported shard format in {}".format(descriptor["file"]))
        if len(payload["episode_ids"]) != int(descriptor["count"]):
            raise ValueError("Shard count mismatch in {}".format(descriptor["file"]))
        self._validate_payload(payload, descriptor["file"])
        self._cache[shard_index] = payload
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def _validate_payload(self, payload, filename):
        for field, entry in self.schema_manifest["fields"].items():
            storage = "transition_data" if entry["storage"] == "transition" else "episode_data"
            if field not in payload[storage]:
                raise KeyError("Field '{}' missing from {}".format(field, filename))
            tensor = payload[storage][field]
            expected_shape = [len(payload["episode_ids"])] + list(entry["shape"])
            if list(tensor.shape) != expected_shape:
                raise ValueError(
                    "Field '{}' in {} has shape {}, expected {}".format(
                        field, filename, list(tensor.shape), expected_shape
                    )
                )
            if tensor.dtype != _NAME_TO_DTYPE[entry["dtype"]]:
                raise TypeError("Field '{}' in {} has dtype {}".format(field, filename, tensor.dtype))

    def _batch_from_ids(self, episode_ids):
        rows = []
        for episode_id in episode_ids:
            if episode_id not in self._episode_index:
                raise KeyError("Unknown episode ID {}".format(episode_id))
            shard_index, row = self._episode_index[episode_id]
            rows.append((self._load_shard(shard_index), row))

        batch = EpisodeBatch(
            self._base_scheme,
            self.groups,
            len(rows),
            self.max_seq_length,
            preprocess=self.preprocess,
            device="cpu",
        )
        for field in TRANSITION_FIELDS:
            values = th.stack([payload["transition_data"][field][row] for payload, row in rows], dim=0)
            batch.data.transition_data[field].copy_(values)
        for field in EPISODE_FIELDS:
            values = th.stack([payload["episode_data"][field][row] for payload, row in rows], dim=0)
            batch.data.episode_data[field].copy_(values)

        for source, (target, transforms) in self.preprocess.items():
            value = batch.data.transition_data[source]
            for transform in transforms:
                value = transform.transform(value)
            batch.data.transition_data[target].copy_(value)

        if self.device != "cpu":
            batch.to(self.device)
        return batch

    def get_episode(self, episode_id):
        return self._batch_from_ids([int(episode_id)])

    def get_raw_episode(self, episode_id):
        episode_id = int(episode_id)
        if episode_id not in self._episode_index:
            raise KeyError("Unknown episode ID {}".format(episode_id))
        shard_index, row = self._episode_index[episode_id]
        payload = self._load_shard(shard_index)
        return {
            "transition_data": {
                field: payload["transition_data"][field][row].clone()
                for field in TRANSITION_FIELDS
            },
            "episode_data": {
                field: payload["episode_data"][field][row].clone()
                for field in EPISODE_FIELDS
            },
        }

    def sample(self, batch_size):
        if not self.can_sample(batch_size):
            raise ValueError("Cannot sample {} episodes from split '{}' with {} episodes".format(
                batch_size, self.split, len(self)
            ))
        positions = self.rng.choice(len(self.episode_ids), int(batch_size), replace=False)
        episode_ids = [self.episode_ids[position] for position in positions]
        return self._batch_from_ids(episode_ids)
