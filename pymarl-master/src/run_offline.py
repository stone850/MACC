from collections.abc import Mapping
from copy import deepcopy
import datetime
import hashlib
import json
import math
import os
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


OFFLINE_TRAINING_METRICS = (
    "td_loss",
    "td_error_abs_mean",
    "grad_norm",
    "q_data_mean",
    "q_max_mean",
    "q_gap_mean",
    "q_tot_abs_max",
    "target_mean",
    "target_abs_max",
    "ood_action_rate",
    "greedy_action_disagreement_rate",
)
OFFLINE_AUXILIARY_METRICS = (
    "representation_loss",
    "recon_loss",
    "sim_loss",
)
SUPPORTED_OFFLINE_CONFIGS = {
    "q_learner": {"mac": "basic_mac", "mixer": "qmix"},
    "latent_q_learner": {"mac": "macc_mac", "mixer": "qmix_hidden"},
}
OOD_ACTION_RATE_DEFINITION = (
    "Fraction of valid agent-timesteps where the current available-action greedy action "
    "differs from the frozen dataset action; this is a policy-disagreement proxy, not a "
    "conditional behavior-support probability."
)


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
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class JsonlMetricJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(self.path, "x", encoding="utf-8")

    def append(self, record):
        self.handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def _json_safe_metrics(metrics):
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                safe[key] = value
            elif math.isnan(float(value)):
                safe[key] = "nan"
            elif float(value) > 0:
                safe[key] = "inf"
            else:
                safe[key] = "-inf"
        else:
            safe[key] = value
    return safe


def _nonfinite_metric_names(metrics):
    required = ("loss", "target_updated") + OFFLINE_TRAINING_METRICS
    missing = [key for key in required if key not in metrics]
    if missing:
        raise KeyError("Offline learner did not return required metrics: {}".format(", ".join(missing)))
    checked = ("loss",) + OFFLINE_TRAINING_METRICS + tuple(
        key for key in OFFLINE_AUXILIARY_METRICS if key in metrics
    )
    return [
        key
        for key in checked
        if not math.isfinite(float(metrics[key]))
    ]


def _training_metric_record(gradient_step, metrics):
    record = {
        "event": "train",
        "gradient_step": int(gradient_step),
        "loss": float(metrics["loss"]),
        "target_updated": bool(metrics["target_updated"]),
    }
    record.update({key: float(metrics[key]) for key in OFFLINE_TRAINING_METRICS})
    record.update({
        key: float(metrics[key])
        for key in OFFLINE_AUXILIARY_METRICS
        if key in metrics
    })
    return record


def _evaluation_metric_record(evaluation):
    record = {"event": "evaluation"}
    record.update(evaluation)
    return record


def _absolute_growth_ratio(initial_value, final_value):
    return abs(float(final_value)) / max(abs(float(initial_value)), 1e-12)


def summarize_training_metrics(records, window, ratio_threshold):
    window = int(window)
    ratio_threshold = float(ratio_threshold)
    if window <= 0:
        raise ValueError("offline_divergence_window must be positive")
    if ratio_threshold <= 0:
        raise ValueError("offline_divergence_ratio_threshold must be positive")

    metric_summary = {}
    summary_window = min(window, len(records))
    metric_names = ("loss",) + OFFLINE_TRAINING_METRICS + OFFLINE_AUXILIARY_METRICS
    for key in metric_names:
        values = np.asarray(
            [float(record[key]) for record in records if key in record],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        initial_mean = float(np.mean(values[:summary_window]))
        final_mean = float(np.mean(values[-summary_window:]))
        metric_summary[key] = {
            "initial_mean": initial_mean,
            "final_mean": final_mean,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "final_to_initial_abs_ratio": _absolute_growth_ratio(initial_mean, final_mean),
            "window": summary_window,
        }

    divergence = {
        "status": "insufficient_data" if len(records) < 2 * window else "stable",
        "detected": False,
        "first_detected_step": None,
        "window": window,
        "ratio_threshold": ratio_threshold,
        "td_loss_growth_ratio": None,
        "q_tot_abs_max_growth_ratio": None,
    }
    if len(records) < 2 * window:
        return {"metrics": metric_summary, "divergence": divergence}

    initial_loss = np.mean([float(record["td_loss"]) for record in records[:window]])
    initial_q_abs = np.mean([float(record["q_tot_abs_max"]) for record in records[:window]])
    for end in range(2 * window, len(records) + 1):
        recent = records[end - window:end]
        recent_loss = np.mean([float(record["td_loss"]) for record in recent])
        recent_q_abs = np.mean([float(record["q_tot_abs_max"]) for record in recent])
        loss_ratio = _absolute_growth_ratio(initial_loss, recent_loss)
        q_abs_ratio = _absolute_growth_ratio(initial_q_abs, recent_q_abs)
        divergence["td_loss_growth_ratio"] = float(loss_ratio)
        divergence["q_tot_abs_max_growth_ratio"] = float(q_abs_ratio)
        if loss_ratio >= ratio_threshold and q_abs_ratio >= ratio_threshold:
            divergence.update({
                "status": "detected",
                "detected": True,
                "first_detected_step": int(records[end - 1]["gradient_step"]),
            })
            break

    return {"metrics": metric_summary, "divergence": divergence}


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
    learner_name = config.get("learner")
    if learner_name not in SUPPORTED_OFFLINE_CONFIGS:
        raise ValueError(
            "Offline training supports only {}".format(
                ", ".join(sorted(SUPPORTED_OFFLINE_CONFIGS))
            )
        )
    expected = SUPPORTED_OFFLINE_CONFIGS[learner_name]
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                "Offline learner={} requires {}={}".format(learner_name, key, value)
            )
    if int(config["offline_updates"]) <= 0:
        raise ValueError("offline_updates must be positive")
    if int(config["log_interval"]) <= 0:
        raise ValueError("log_interval must be positive")
    if not config.get("use_offline_training", False):
        raise ValueError("The offline runner requires use_offline_training=true")
    if int(config["offline_divergence_window"]) <= 0:
        raise ValueError("offline_divergence_window must be positive")
    if float(config["offline_divergence_ratio_threshold"]) <= 0:
        raise ValueError("offline_divergence_ratio_threshold must be positive")
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
    metrics_path = run_directory / "training_metrics.jsonl"
    failure_path = run_directory / "failure.json"
    summary_path = run_directory / "summary.json"
    metric_journal = JsonlMetricJournal(metrics_path)
    logger = Logger(sacred_log)
    if sacred_run is not None:
        logger.setup_sacred(sacred_run)
    if args.use_tensorboard:
        logger.setup_tb(str(run_directory / "tb"))
    sacred_log.info("Offline training parameters:\n%s", pprint.pformat(config, indent=4, width=100))

    losses = []
    training_metric_records = []
    evaluations = []
    evaluator = None
    learner = None
    final_model_path = None
    shard_checksums = {item["file"]: item["sha256"] for item in metadata["shards"]}
    start_step = 0
    gradient_step = start_step
    last_metrics = None
    try:
        mac = mac_REGISTRY[args.mac](dataset.scheme, dataset.groups, args)
        learner = learner_REGISTRY[args.learner](mac, dataset.scheme, logger, args)
        if args.use_cuda:
            learner.cuda()

        if args.checkpoint_path:
            checkpoint, start_step = _resolve_training_checkpoint(args.checkpoint_path, args.load_step)
            gradient_step = start_step
            sacred_log.info("Resuming offline learner from %s", checkpoint)
            learner.load_models(str(checkpoint))

        parameter_hash_before = _parameter_hash(learner)
        if args.offline_eval_interval > 0 and args.offline_eval_at_start:
            evaluator = OfflineEvaluator(args, logger, mac, env_info, registry=evaluator_registry)
            evaluation = evaluator.evaluate(start_step)
            evaluations.append(evaluation)
            metric_journal.append(_evaluation_metric_record(evaluation))
            logger.log_stat("offline_evaluation_return_mean", evaluation["return_mean"], start_step)
            logger.log_stat("offline_evaluation_return_std", evaluation["return_std"], start_step)

        for offset in range(1, int(args.offline_updates) + 1):
            gradient_step = start_step + offset
            batch = dataset.sample(int(args.batch_size))
            max_ep_t = int(batch.max_t_filled().item())
            batch = batch[:, :max_ep_t]
            if batch.device != args.device:
                batch.to(args.device)
            metrics = learner.train(batch, gradient_step, gradient_step)
            last_metrics = metrics
            nonfinite_names = _nonfinite_metric_names(metrics)
            if nonfinite_names:
                failure_record = {
                    "event": "train",
                    "gradient_step": int(gradient_step),
                    "nonfinite_metrics": nonfinite_names,
                }
                failure_record.update(_json_safe_metrics(metrics))
                metric_journal.append(failure_record)
                raise FloatingPointError(
                    "Non-finite offline metric(s) at step {}: {}".format(
                        gradient_step, ", ".join(nonfinite_names)
                    )
                )
            losses.append(metrics["loss"])

            if offset == 1 or gradient_step % args.log_interval == 0:
                metric_record = _training_metric_record(gradient_step, metrics)
                metric_journal.append(metric_record)
                training_metric_records.append(metric_record)
                logger.log_stat("gradient_step", gradient_step, gradient_step)
                logger.log_stat("offline_loss", metrics["loss"], gradient_step)
                for key in OFFLINE_TRAINING_METRICS:
                    logger.log_stat(key, metrics[key], gradient_step)
                for key in OFFLINE_AUXILIARY_METRICS:
                    if key in metrics:
                        logger.log_stat(key, metrics[key], gradient_step)
                logger.log_stat("target_updated", int(metrics["target_updated"]), gradient_step)

            if args.offline_eval_interval > 0 and gradient_step % args.offline_eval_interval == 0:
                if evaluator is None:
                    evaluator = OfflineEvaluator(
                        args, logger, mac, env_info, registry=evaluator_registry
                    )
                evaluation = evaluator.evaluate(gradient_step)
                evaluations.append(evaluation)
                metric_journal.append(_evaluation_metric_record(evaluation))
                logger.log_stat("offline_evaluation_return_mean", evaluation["return_mean"], gradient_step)
                logger.log_stat("offline_evaluation_return_std", evaluation["return_std"], gradient_step)

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
        diagnostic_summary = summarize_training_metrics(
            training_metric_records,
            args.offline_divergence_window,
            args.offline_divergence_ratio_threshold,
        )
        parameter_hash_after = _parameter_hash(learner)
        summary = {
            "status": "completed",
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
            "metrics_path": str(metrics_path),
            "training_metric_records": len(training_metric_records),
            "evaluation_metric_records": len(evaluations),
            "metric_summary": diagnostic_summary["metrics"],
            "divergence": diagnostic_summary["divergence"],
            "ood_action_rate_definition": OOD_ACTION_RATE_DEFINITION,
            "target_update_records": sum(
                int(record["target_updated"]) for record in training_metric_records
            ),
            "initial_loss_mean": initial_loss,
            "final_loss_mean": final_loss,
            "minimum_loss": float(np.min(losses)),
            "maximum_loss": float(np.max(losses)),
            "loss_window": window,
            "final_to_initial_loss_ratio": final_ratio,
            "overfit_mode": bool(args.offline_overfit_mode),
            "overfit_passed": bool(overfit_passed),
            "parameter_hash_before": parameter_hash_before,
            "parameter_hash_after": parameter_hash_after,
            "dataset_shard_checksums_before": shard_checksums,
            "dataset_shard_checksums_unchanged": True,
            "final_model_path": None if final_model_path is None else str(final_model_path),
        }
        if summary["parameter_hash_before"] == summary["parameter_hash_after"]:
            raise AssertionError("Offline learner parameters did not change")
        if not overfit_passed:
            raise AssertionError(
                "Offline overfit failed: final/initial loss ratio {:.6f} > {:.6f}".format(
                    final_ratio, args.offline_overfit_final_ratio
                )
            )
        _atomic_json(summary_path, summary)
        sacred_log.info("Offline training summary written to %s", summary_path)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        failure = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "gradient_step": int(gradient_step),
            "exception_type": type(error).__name__,
            "message": str(error),
            "last_metrics": None if last_metrics is None else _json_safe_metrics(last_metrics),
            "metrics_path": str(metrics_path),
        }
        _atomic_json(failure_path, failure)
        raise
    finally:
        if evaluator is not None:
            evaluator.close()
        metric_journal.close()
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
