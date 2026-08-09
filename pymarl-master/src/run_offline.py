from collections.abc import Mapping
from copy import deepcopy
import datetime
import hashlib
import json
import math
from pathlib import Path
import pprint
import sys
from types import SimpleNamespace as SN

import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th
import yaml

from components.episode_schema import build_episode_components
from components.offline_dataset import OfflineEpisodeDataset
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as learner_REGISTRY
from runners import REGISTRY as runner_REGISTRY
from utils.logging import Logger, get_logger


SETTINGS["CAPTURE_MODE"] = "fd"
console_logger = get_logger()
ex = Experiment("offline_qmix")
ex.logger = console_logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


def _config_copy(value):
    if isinstance(value, Mapping):
        return {key: _config_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_config_copy(item) for item in value]
    return deepcopy(value)


def _recursive_dict_update(target, update):
    for key, value in update.items():
        if isinstance(value, Mapping):
            target[key] = _recursive_dict_update(target.get(key, {}), value)
        else:
            target[key] = value
    return target


def _get_algorithm_config(params):
    config_name = None
    for index, value in enumerate(params):
        if value.split("=")[0] == "--config":
            config_name = value.split("=", 1)[1]
            del params[index]
            break
    if config_name is None:
        raise ValueError("--config is required")
    path = Path(__file__).parent / "config" / "algs" / "{}.yaml".format(config_name)
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp".format(path.name))
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _parameter_hash(learner):
    digest = hashlib.sha256()
    for parameter in learner.params:
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _resolve_training_checkpoint(checkpoint_path, load_step):
    root = Path(checkpoint_path).expanduser().resolve()
    if (root / "agent.th").is_file():
        step = int(root.name) if root.name.isdigit() else int(load_step)
        return root, step
    if not root.is_dir():
        raise FileNotFoundError("Offline checkpoint directory does not exist: {}".format(root))
    steps = [
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "agent.th").is_file()
    ]
    if not steps:
        raise FileNotFoundError("No numbered offline checkpoint under {}".format(root))
    step = max(steps) if int(load_step) == 0 else min(steps, key=lambda value: abs(value - int(load_step)))
    return root / str(step), step


class OfflineEvaluator:
    def __init__(self, args, logger, mac, expected_env_info, registry=None):
        registry = runner_REGISTRY if registry is None else registry
        eval_args = deepcopy(args)
        eval_args.env_args = deepcopy(args.env_args)
        eval_args.env_args["seed"] = int(args.offline_eval_seed)
        eval_args.test_nepisode = int(args.offline_eval_episodes)
        self.runner = registry[eval_args.runner](args=eval_args, logger=logger)
        actual_env_info = self.runner.get_env_info()
        for key in ("state_shape", "obs_shape", "n_actions", "n_agents", "episode_limit"):
            if actual_env_info[key] != expected_env_info[key]:
                raise ValueError("Evaluation environment '{}' differs from dataset metadata".format(key))
        scheme, groups, preprocess = build_episode_components(actual_env_info)
        self.runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        self.mac = mac
        self.logger = logger
        self.environment_steps = 0
        self.calls = 0

    def evaluate(self, gradient_step):
        self.runner.t_env = int(gradient_step)
        was_training = self.mac.agent.training
        self.mac.agent.eval()
        steps_before = self.environment_steps
        episode_returns = []
        episode_lengths = []
        try:
            for _ in range(self.runner.args.offline_eval_episodes):
                episode_batch = self.runner.run(test_mode=True)
                self.calls += 1
                self.environment_steps += int(self.runner.t)
                episode_returns.append(float(episode_batch["reward"].sum().item()))
                episode_lengths.append(int(self.runner.t))
        finally:
            self.mac.agent.train(was_training)

        return {
            "gradient_step": int(gradient_step),
            "episodes": int(self.runner.args.offline_eval_episodes),
            "environment_steps": self.environment_steps - steps_before,
            "return_mean": float(np.mean(episode_returns)),
            "return_std": float(np.std(episode_returns)),
            "episode_length_mean": float(np.mean(episode_lengths)),
        }

    def close(self):
        self.runner.close_env()


def run_offline_training(config, sacred_log, sacred_run=None, evaluator_registry=None):
    config = _config_copy(config)
    if not config.get("dataset_path"):
        raise ValueError("dataset_path is required")
    if config.get("learner") != "q_learner":
        raise ValueError("The first offline baseline supports only learner=q_learner")
    if int(config["offline_updates"]) <= 0:
        raise ValueError("offline_updates must be positive")
    if config["use_cuda"] and not th.cuda.is_available():
        sacred_log.warning("CUDA is unavailable; offline training will use CPU")
        config["use_cuda"] = False
    config["device"] = "cuda" if config["use_cuda"] else "cpu"

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    if config["use_cuda"]:
        th.cuda.manual_seed_all(config["seed"])

    max_episodes = int(config["offline_max_train_episodes"])
    max_episodes = None if max_episodes == 0 else max_episodes
    dataset = OfflineEpisodeDataset(
        config["dataset_path"],
        split=config["offline_train_split"],
        seed=config["seed"],
        device="cpu",
        verify_checksums=config["offline_verify_checksums"],
        max_episodes=max_episodes,
    )
    if not dataset.can_sample(int(config["batch_size"])):
        raise ValueError("batch_size exceeds the selected offline training episodes")

    metadata = dataset.metadata
    with open(dataset.path / "split.json", "r", encoding="utf-8") as handle:
        dataset_splits = json.load(handle)
    env_info = _config_copy(metadata["env_info"])
    config["env"] = metadata["environment"]
    config["env_args"] = _config_copy(metadata["environment_config"])
    config["n_agents"] = env_info["n_agents"]
    config["n_actions"] = env_info["n_actions"]
    config["state_shape"] = env_info["state_shape"]
    config["test_nepisode"] = int(config["offline_eval_episodes"])
    token = "{}__{}__{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"),
        config["name"],
        config["remarks"],
    )
    config["unique_token"] = token
    args = SN(**config)

    run_directory = Path(args.offline_output_path).expanduser().resolve() / token
    run_directory.mkdir(parents=True, exist_ok=False)
    logger = Logger(sacred_log)
    if sacred_run is not None:
        logger.setup_sacred(sacred_run)
    if args.use_tensorboard:
        logger.setup_tb(str(run_directory / "tb"))
    sacred_log.info("Offline training parameters:\n%s", pprint.pformat(config, indent=4, width=100))

    mac = mac_REGISTRY[args.mac](dataset.scheme, dataset.groups, args)
    learner = learner_REGISTRY[args.learner](mac, dataset.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    start_step = 0
    if args.checkpoint_path:
        checkpoint, start_step = _resolve_training_checkpoint(args.checkpoint_path, args.load_step)
        sacred_log.info("Resuming offline learner from %s", checkpoint)
        learner.load_models(str(checkpoint))

    losses = []
    evaluations = []
    evaluator = None
    final_model_path = None
    shard_checksums = {item["file"]: item["sha256"] for item in metadata["shards"]}
    parameter_hash_before = _parameter_hash(learner)
    summary_path = run_directory / "summary.json"
    try:
        for offset in range(1, int(args.offline_updates) + 1):
            gradient_step = start_step + offset
            batch = dataset.sample(int(args.batch_size))
            max_ep_t = int(batch.max_t_filled().item())
            batch = batch[:, :max_ep_t]
            if batch.device != args.device:
                batch.to(args.device)
            metrics = learner.train(batch, gradient_step, gradient_step)
            if not math.isfinite(metrics["loss"]):
                raise FloatingPointError("Non-finite offline loss at step {}".format(gradient_step))
            losses.append(metrics["loss"])

            if gradient_step == 1 or gradient_step % args.log_interval == 0:
                logger.log_stat("gradient_step", gradient_step, gradient_step)
                logger.log_stat("offline_loss", metrics["loss"], gradient_step)

            if args.offline_eval_interval > 0 and gradient_step % args.offline_eval_interval == 0:
                if evaluator is None:
                    evaluator = OfflineEvaluator(
                        args, logger, mac, env_info, registry=evaluator_registry
                    )
                evaluations.append(evaluator.evaluate(gradient_step))

            if args.save_model and args.offline_save_interval > 0 and gradient_step % args.offline_save_interval == 0:
                final_model_path = run_directory / "models" / str(gradient_step)
                final_model_path.mkdir(parents=True, exist_ok=False)
                learner.save_models(str(final_model_path))

        final_step = start_step + int(args.offline_updates)
        if args.save_model and (final_model_path is None or final_model_path.name != str(final_step)):
            final_model_path = run_directory / "models" / str(final_step)
            final_model_path.mkdir(parents=True, exist_ok=False)
            learner.save_models(str(final_model_path))

        dataset.verify_shards()
        window = min(int(args.offline_overfit_window), len(losses) // 2)
        initial_loss = float(np.mean(losses[:window])) if window else float(losses[0])
        final_loss = float(np.mean(losses[-window:])) if window else float(losses[-1])
        final_ratio = final_loss / initial_loss if initial_loss > 0 else 0.0
        overfit_passed = (not args.offline_overfit_mode) or final_ratio <= args.offline_overfit_final_ratio
        summary = {
            "training_algorithm": args.name,
            "behavior_algorithm": metadata["algorithm"],
            "dataset_path": str(dataset.path),
            "dataset_version": metadata["dataset_version"],
            "train_split": dataset.split,
            "train_episode_ids": dataset.episode_ids,
            "train_episodes": len(dataset),
            "dataset_split_sizes": {
                name: len(episode_ids)
                for name, episode_ids in dataset_splits.items()
                if name != "all"
            },
            "validation_episode_ids": [int(value) for value in dataset_splits["validation"]],
            "test_episode_ids": [int(value) for value in dataset_splits["test"]],
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "target_update_interval": int(args.target_update_interval),
            "seed": int(args.seed),
            "start_gradient_step": start_step,
            "gradient_updates": int(args.offline_updates),
            "final_gradient_step": final_step,
            "training_environment_steps": 0,
            "evaluation_calls": 0 if evaluator is None else evaluator.calls,
            "evaluation_environment_steps": 0 if evaluator is None else evaluator.environment_steps,
            "evaluations": evaluations,
            "initial_loss_mean": initial_loss,
            "final_loss_mean": final_loss,
            "minimum_loss": float(np.min(losses)),
            "maximum_loss": float(np.max(losses)),
            "loss_window": window,
            "final_to_initial_loss_ratio": final_ratio,
            "overfit_mode": bool(args.offline_overfit_mode),
            "overfit_passed": bool(overfit_passed),
            "parameter_hash_before": parameter_hash_before,
            "parameter_hash_after": _parameter_hash(learner),
            "dataset_shard_checksums_before": shard_checksums,
            "dataset_shard_checksums_unchanged": True,
            "final_model_path": None if final_model_path is None else str(final_model_path),
        }
        _atomic_json(summary_path, summary)
        sacred_log.info("Offline training summary written to %s", summary_path)
        if summary["parameter_hash_before"] == summary["parameter_hash_after"]:
            raise AssertionError("Offline learner parameters did not change")
        if not overfit_passed:
            raise AssertionError(
                "Offline overfit failed: final/initial loss ratio {:.6f} > {:.6f}".format(
                    final_ratio, args.offline_overfit_final_ratio
                )
            )
        return summary
    finally:
        if evaluator is not None:
            evaluator.close()
        logger.close()


@ex.main
def offline_main(_run, _config, _log):
    return run_offline_training(_config, _log, sacred_run=_run)


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    config_directory = Path(__file__).parent / "config"
    with open(config_directory / "default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    algorithm_config = _get_algorithm_config(params)
    with open(config_directory / "offline.yaml", "r", encoding="utf-8") as handle:
        offline_config = yaml.safe_load(handle)
    config = _recursive_dict_update(config, algorithm_config)
    config = _recursive_dict_update(config, offline_config)
    ex.add_config(config)
    observer_path = Path(config["offline_output_path"]) / "sacred" / config["name"] / config["remarks"]
    ex.observers.append(FileStorageObserver(str(observer_path)))
    ex.run_commandline(params)
