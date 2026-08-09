from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import pprint
import subprocess
import sys
from types import SimpleNamespace as SN

import numpy as np
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th
import yaml

from components.episode_schema import build_episode_components
from components.episode_buffer import EpisodeBatch
from components.offline_dataset import OfflineDatasetWriter
from controllers import REGISTRY as mac_REGISTRY
from runners import REGISTRY as runner_REGISTRY
from utils.logging import Logger, get_logger


SETTINGS["CAPTURE_MODE"] = "fd"
console_logger = get_logger()
ex = Experiment("offline_lbf_collection")
ex.logger = console_logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


def _config_copy(config):
    if isinstance(config, dict):
        return {key: _config_copy(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_config_copy(value) for value in config]
    return deepcopy(config)


def _recursive_dict_update(target, update):
    for key, value in update.items():
        if isinstance(value, Mapping):
            target[key] = _recursive_dict_update(target.get(key, {}), value)
        else:
            target[key] = value
    return target


def _get_named_config(params, argument, subfolder):
    config_name = None
    for index, value in enumerate(params):
        if value.split("=")[0] == argument:
            config_name = value.split("=", 1)[1]
            del params[index]
            break
    if config_name is None:
        raise ValueError("{} is required".format(argument))

    config_path = Path(__file__).parent / "config" / subfolder / "{}.yaml".format(config_name)
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_checkpoint(checkpoint_path, load_step):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if (checkpoint_path / "agent.th").is_file():
        step = int(checkpoint_path.name) if checkpoint_path.name.isdigit() else int(load_step)
        return checkpoint_path, step
    if not checkpoint_path.is_dir():
        raise FileNotFoundError("Checkpoint directory does not exist: {}".format(checkpoint_path))

    available = []
    for candidate in checkpoint_path.iterdir():
        if candidate.is_dir() and candidate.name.isdigit() and (candidate / "agent.th").is_file():
            available.append(int(candidate.name))
    if not available:
        raise FileNotFoundError("No numbered checkpoint containing agent.th under {}".format(checkpoint_path))
    step = max(available) if int(load_step) == 0 else min(available, key=lambda value: abs(value - int(load_step)))
    return checkpoint_path / str(step), step


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_version(repo_root):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def collect_dataset(config, sacred_log):
    config = _config_copy(config)
    if config["env"] != "foraging":
        raise ValueError("Offline subtask collection currently supports only env=foraging")
    if not 0.0 <= float(config["collection_epsilon"]) <= 1.0:
        raise ValueError("collection_epsilon must be in [0, 1]")
    if int(config["collection_episodes"]) <= 0:
        raise ValueError("collection_episodes must be positive")
    if not config.get("checkpoint_path"):
        raise ValueError("checkpoint_path is required for frozen offline collection")

    if config["use_cuda"] and not th.cuda.is_available():
        sacred_log.warning("CUDA is unavailable; collecting on CPU")
        config["use_cuda"] = False
    config["device"] = "cuda" if config["use_cuda"] else "cpu"
    config["env_args"]["seed"] = int(config["seed"])
    config["epsilon_start"] = float(config["collection_epsilon"])
    config["epsilon_finish"] = float(config["collection_epsilon"])
    config["epsilon_anneal_time"] = 1

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    if config["use_cuda"]:
        th.cuda.manual_seed_all(config["seed"])

    args = SN(**config)
    logger = Logger(sacred_log)
    sacred_log.info("Offline collection parameters:\n%s", pprint.pformat(config, indent=4, width=100))
    runner = runner_REGISTRY[args.runner](args=args, logger=logger)
    writer = None
    try:
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        scheme, groups, preprocess = build_episode_components(env_info)
        mac_scheme = EpisodeBatch(
            scheme, groups, 1, 1, preprocess=preprocess, device="cpu"
        ).scheme
        mac = mac_REGISTRY[args.mac](mac_scheme, groups, args)

        model_path, checkpoint_step = _resolve_checkpoint(args.checkpoint_path, args.load_step)
        sacred_log.info("Loading frozen policy from %s", model_path)
        mac.load_models(str(model_path))
        if args.use_cuda:
            mac.cuda()
        mac.agent.eval()
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        runner.t_env = checkpoint_step

        repo_root = Path(__file__).resolve().parents[2]
        metadata = {
            "dataset_version": args.dataset_version,
            "algorithm": args.name,
            "checkpoint_path": str(model_path),
            "checkpoint_step": checkpoint_step,
            "checkpoint_agent_sha256": _file_sha256(model_path / "agent.th"),
            "collection_epsilon": float(args.collection_epsilon),
            "seed": int(args.seed),
            "environment": args.env,
            "environment_config": _config_copy(args.env_args),
            "env_info": _config_copy(env_info),
            "resolved_config": config,
            "policy_eval_mode": True,
        }
        metadata.update(_git_version(repo_root))
        split_seed = int(args.dataset_split_seed)
        writer = OfflineDatasetWriter(
            args.dataset_path,
            scheme,
            groups,
            env_info["episode_limit"] + 1,
            metadata,
            shard_size=args.dataset_shard_size,
            split_ratios=args.dataset_split_ratios,
            split_seed=split_seed,
        )

        with th.no_grad():
            for episode in range(int(args.collection_episodes)):
                episode_batch = runner.run(test_mode=False)
                if not np.isclose(mac.action_selector.epsilon, args.collection_epsilon):
                    raise RuntimeError("Collection epsilon drifted to {}".format(mac.action_selector.epsilon))
                writer.add_episode(episode_batch)
                sacred_log.info(
                    "Collected episode %d/%d (length=%d)",
                    episode + 1,
                    args.collection_episodes,
                    int(episode_batch["filled"].sum().item()) - 1,
                )
        dataset_path = writer.finalize()
        sacred_log.info("Frozen dataset written to %s", dataset_path)
        return str(dataset_path)
    finally:
        runner.close_env()
        logger.close()


@ex.main
def collection_main(_config, _log):
    return collect_dataset(_config, _log)


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    config_dir = Path(__file__).parent / "config"
    with open(config_dir / "default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env_config = _get_named_config(params, "--env-config", "envs")
    algorithm_config = _get_named_config(params, "--config", "algs")
    with open(config_dir / "offline_collection.yaml", "r", encoding="utf-8") as handle:
        collection_config = yaml.safe_load(handle)

    config = _recursive_dict_update(config, env_config)
    config = _recursive_dict_update(config, algorithm_config)
    config = _recursive_dict_update(config, collection_config)
    ex.add_config(config)
    ex.run_commandline(params)
