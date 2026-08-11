import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as SN
from unittest.mock import patch

import torch as th
import yaml

from components.episode_buffer import EpisodeBatch
from components.episode_schema import build_episode_components
from components.offline_dataset import OfflineDatasetWriter
from learners import REGISTRY as learner_REGISTRY
from run_offline import (
    OFFLINE_AUXILIARY_METRICS,
    OFFLINE_TRAINING_METRICS,
    OfflineEvaluator,
    run_offline_training,
    summarize_training_metrics,
)


class SilentLog:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class FakeMac:
    def __init__(self):
        self.agent = th.nn.Linear(1, 1)


class FakeRunner:
    def __init__(self, args, logger, env_info):
        self.args = args
        self.logger = logger
        self.env_info = env_info
        self.calls = []
        self.t = 0
        self.closed = False

    def get_env_info(self):
        return self.env_info

    def setup(self, **kwargs):
        self.setup_args = kwargs

    def run(self, test_mode=False):
        self.calls.append(test_mode)
        self.t = 3
        return {"reward": th.tensor([[[float(len(self.calls))]]])}

    def close_env(self):
        self.closed = True


class OfflineRunnerTest(unittest.TestCase):
    def _env_info(self):
        return {
            "state_shape": 24,
            "obs_shape": 24,
            "n_actions": 6,
            "n_agents": 4,
            "episode_limit": 5,
            "n_subtasks": 4,
            "subtask_state_shape": 3,
            "subtask_obs_shape": 3,
        }

    def _episode(self, scheme, groups, preprocess, marker):
        batch = EpisodeBatch(scheme, groups, 1, 6, preprocess=preprocess, device="cpu")
        length = 3 + marker % 2
        batch.data.transition_data["filled"][0, :length + 1] = 1
        batch.data.transition_data["state"][0, :length + 1] = marker
        batch.data.transition_data["obs"][0, :length + 1] = marker
        action = marker % 6
        batch.data.transition_data["avail_actions"][0, :length + 1, :, action] = 1
        batch.data.transition_data["actions"][0, :length + 1] = action
        batch.data.transition_data["reward"][0, :length] = marker / 10.0
        batch.data.transition_data["terminated"][0, length - 1] = 1
        batch.data.transition_data["subtask_state"][0, :length + 1] = marker
        batch.data.transition_data["subtask_obs"][0, :length + 1] = marker
        batch.data.transition_data["subtask_visible"][0, :length + 1] = 1
        batch.data.transition_data["subtask_mask"][0, :length + 1] = 1
        batch.data.episode_data["subtask_id"][0] = th.tensor([3, 5, 7, 9])
        return batch

    def _config(self, dataset_path, output_path, algorithm="qmix"):
        root = Path(__file__).resolve().parents[1]
        with open(root / "src/config/default.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with open(
            root / "src/config/algs/{}.yaml".format(algorithm),
            "r",
            encoding="utf-8",
        ) as handle:
            config.update(yaml.safe_load(handle))
        with open(root / "src/config/offline.yaml", "r", encoding="utf-8") as handle:
            config.update(yaml.safe_load(handle))
        config.update({
            "dataset_path": str(dataset_path),
            "offline_output_path": str(output_path),
            "offline_updates": 1,
            "offline_eval_interval": 0,
            "offline_max_train_episodes": 2,
            "offline_save_interval": 0,
            "batch_size": 2,
            "save_model": False,
            "use_tensorboard": False,
            "use_cuda": False,
            "seed": 1,
            "remarks": "test",
        })
        return config

    def _write_dataset(self, root):
        env_info = self._env_info()
        scheme, groups, preprocess = build_episode_components(env_info)
        dataset_path = root / "dataset"
        writer = OfflineDatasetWriter(
            dataset_path,
            scheme,
            groups,
            6,
            {
                "dataset_version": "test",
                "algorithm": "macc",
                "environment": "foraging",
                "environment_config": {
                    "field_size": 10,
                    "players": 4,
                    "max_food": 4,
                    "force_coop": False,
                    "partially_observe": True,
                    "sight": 2,
                    "is_print": False,
                    "need_render": False,
                    "seed": 1,
                },
                "env_info": env_info,
            },
            shard_size=2,
        )
        for marker in range(1, 5):
            writer.add_episode(self._episode(scheme, groups, preprocess, marker))
        writer.finalize()
        return dataset_path

    @staticmethod
    def _metric_record(step, td_loss, q_abs):
        record = {
            "event": "train",
            "gradient_step": step,
            "target_updated": False,
        }
        record.update({key: 1.0 for key in OFFLINE_TRAINING_METRICS})
        record["td_loss"] = td_loss
        record["q_tot_abs_max"] = q_abs
        return record

    def test_training_does_not_construct_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = self._write_dataset(root)

            class ExplodingRegistry:
                def __getitem__(self, key):
                    raise AssertionError("Training attempted to construct an environment")

            summary = run_offline_training(
                self._config(dataset_path, root / "output"),
                SilentLog(),
                evaluator_registry=ExplodingRegistry(),
            )
            self.assertEqual(summary["training_environment_steps"], 0)
            self.assertEqual(summary["evaluation_calls"], 0)
            self.assertEqual(summary["dataset_split_sizes"], {
                "train": 3,
                "validation": 1,
                "test": 0,
            })
            self.assertEqual(summary["train_episodes"], 2)
            self.assertEqual(summary["learning_rate"], 1e-6)
            self.assertNotEqual(summary["parameter_hash_before"], summary["parameter_hash_after"])
            self.assertTrue(summary["dataset_shard_checksums_unchanged"])
            self.assertEqual(summary["training_metric_records"], 1)
            self.assertEqual(summary["divergence"]["status"], "insufficient_data")
            metrics_path = Path(summary["metrics_path"])
            records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["event"], "train")
            self.assertEqual(records[0]["gradient_step"], 1)
            self.assertTrue(set(OFFLINE_TRAINING_METRICS).issubset(records[0]))

    def test_initial_evaluation_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = self._write_dataset(root)
            created = []

            def factory(args, logger):
                runner = FakeRunner(args, logger, self._env_info())
                created.append(runner)
                return runner

            config = self._config(dataset_path, root / "output")
            config.update({
                "offline_eval_interval": 1,
                "offline_eval_episodes": 2,
            })
            summary = run_offline_training(
                config,
                SilentLog(),
                evaluator_registry={"episode": factory},
            )

            self.assertEqual([item["gradient_step"] for item in summary["evaluations"]], [0, 1])
            self.assertEqual(summary["evaluation_calls"], 4)
            self.assertEqual(created[0].calls, [True, True, True, True])
            records = [
                json.loads(line)
                for line in Path(summary["metrics_path"]).read_text().splitlines()
            ]
            self.assertEqual([record["event"] for record in records], [
                "evaluation", "train", "evaluation"
            ])

    def test_nonfinite_metric_writes_failure_artifacts(self):
        class NonfiniteLearner:
            def __init__(self, mac, scheme, logger, args):
                del mac, scheme, logger, args
                self.params = [th.nn.Parameter(th.tensor([1.0]))]

            def train(self, batch, t_env, episode_num):
                del batch, t_env, episode_num
                metrics = {key: 1.0 for key in OFFLINE_TRAINING_METRICS}
                metrics.update({
                    "loss": float("nan"),
                    "td_loss": float("nan"),
                    "target_updated": False,
                })
                return metrics

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = self._write_dataset(root)
            config = self._config(dataset_path, root / "output")
            with patch.dict(learner_REGISTRY, {"q_learner": NonfiniteLearner}):
                with self.assertRaisesRegex(FloatingPointError, "td_loss"):
                    run_offline_training(config, SilentLog())

            run_directory = next(path for path in (root / "output").iterdir() if path.is_dir())
            failure = json.loads((run_directory / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["gradient_step"], 1)
            self.assertEqual(failure["last_metrics"]["td_loss"], "nan")
            record = json.loads((run_directory / "training_metrics.jsonl").read_text())
            self.assertEqual(record["nonfinite_metrics"], ["loss", "td_loss"])

    def test_macc_training_records_auxiliary_metrics_without_training_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = self._write_dataset(root)

            class ExplodingRegistry:
                def __getitem__(self, key):
                    raise AssertionError("Training attempted to construct an environment")

            config = self._config(dataset_path, root / "output", algorithm="macc")
            config.update({
                "offline_updates": 2,
                "log_interval": 1,
                "target_update_interval": 1,
            })
            summary = run_offline_training(
                config,
                SilentLog(),
                evaluator_registry=ExplodingRegistry(),
            )

            self.assertEqual(summary["training_algorithm"], "macc")
            self.assertEqual(summary["training_environment_steps"], 0)
            self.assertEqual(summary["target_update_records"], 2)
            self.assertTrue(set(OFFLINE_AUXILIARY_METRICS).issubset(summary["metric_summary"]))
            records = [
                json.loads(line)
                for line in Path(summary["metrics_path"]).read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertTrue(set(OFFLINE_AUXILIARY_METRICS).issubset(record))
                self.assertAlmostEqual(
                    record["loss"],
                    record["td_loss"] + record["representation_loss"],
                    places=5,
                )

    def test_macc_checkpoint_can_resume_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = self._write_dataset(root)
            first_config = self._config(dataset_path, root / "first", algorithm="macc")
            first_config.update({
                "save_model": True,
                "offline_save_interval": 1,
            })
            first = run_offline_training(first_config, SilentLog())

            created = []

            def factory(args, logger):
                runner = FakeRunner(args, logger, self._env_info())
                created.append(runner)
                return runner

            resumed_config = self._config(dataset_path, root / "resumed", algorithm="macc")
            resumed_config.update({
                "checkpoint_path": first["final_model_path"],
                "load_step": 0,
                "save_model": True,
                "offline_save_interval": 1,
                "offline_eval_interval": 1,
                "offline_eval_episodes": 2,
            })
            resumed = run_offline_training(
                resumed_config,
                SilentLog(),
                evaluator_registry={"episode": factory},
            )

            self.assertEqual(resumed["start_gradient_step"], 1)
            self.assertEqual(resumed["final_gradient_step"], 2)
            self.assertEqual(
                [item["gradient_step"] for item in resumed["evaluations"]],
                [1, 2],
            )
            self.assertTrue(Path(resumed["final_model_path"], "agent.th").is_file())
            self.assertEqual(created[0].calls, [True, True, True, True])

    def test_keyboard_interrupt_writes_interrupted_status(self):
        class InterruptingLearner:
            def __init__(self, mac, scheme, logger, args):
                del mac, scheme, logger, args
                self.params = [th.nn.Parameter(th.tensor([1.0]))]

            def train(self, batch, t_env, episode_num):
                del batch, t_env, episode_num
                raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = self._write_dataset(root)
            config = self._config(dataset_path, root / "output")
            with patch.dict(learner_REGISTRY, {"q_learner": InterruptingLearner}):
                with self.assertRaises(KeyboardInterrupt):
                    run_offline_training(config, SilentLog())

            run_directory = next(path for path in (root / "output").iterdir() if path.is_dir())
            failure = json.loads((run_directory / "failure.json").read_text())
            self.assertEqual(failure["status"], "interrupted")
            self.assertEqual(failure["gradient_step"], 1)
            self.assertEqual(failure["exception_type"], "KeyboardInterrupt")
            self.assertIsNone(failure["last_metrics"])

    def test_divergence_summary_states(self):
        insufficient = summarize_training_metrics(
            [self._metric_record(1, 1.0, 2.0)], window=2, ratio_threshold=100.0
        )
        self.assertEqual(insufficient["divergence"]["status"], "insufficient_data")

        stable = summarize_training_metrics([
            self._metric_record(step, 1.0, 2.0) for step in range(1, 5)
        ], window=2, ratio_threshold=100.0)
        self.assertEqual(stable["divergence"]["status"], "stable")
        self.assertFalse(stable["divergence"]["detected"])

        exploding = summarize_training_metrics([
            self._metric_record(1, 1.0, 2.0),
            self._metric_record(2, 1.0, 2.0),
            self._metric_record(3, 1000.0, 2000.0),
            self._metric_record(4, 1000.0, 2000.0),
        ], window=2, ratio_threshold=100.0)
        self.assertEqual(exploding["divergence"]["status"], "detected")
        self.assertEqual(exploding["divergence"]["first_detected_step"], 4)

    def test_evaluator_forces_test_mode(self):
        env_info = self._env_info()
        created = []

        def factory(args, logger):
            runner = FakeRunner(args, logger, env_info)
            created.append(runner)
            return runner

        args = SN(
            runner="episode",
            env="foraging",
            env_args={"seed": 1},
            offline_eval_seed=10001,
            offline_eval_episodes=2,
        )
        evaluator = OfflineEvaluator(
            args,
            SilentLog(),
            FakeMac(),
            env_info,
            registry={"episode": factory},
        )
        result = evaluator.evaluate(10)
        self.assertEqual(created[0].calls, [True, True])
        self.assertEqual(result["environment_steps"], 6)
        self.assertEqual(result["return_mean"], 1.5)
        self.assertEqual(result["return_std"], 0.5)
        self.assertEqual(result["episode_length_mean"], 3.0)
        evaluator.close()
        self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
