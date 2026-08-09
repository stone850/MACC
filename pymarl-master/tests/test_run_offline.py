import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as SN

import torch as th
import yaml

from components.episode_buffer import EpisodeBatch
from components.episode_schema import build_episode_components
from components.offline_dataset import OfflineDatasetWriter
from run_offline import OfflineEvaluator, run_offline_training


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

    def _config(self, dataset_path, output_path):
        root = Path(__file__).resolve().parents[1]
        with open(root / "src/config/default.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with open(root / "src/config/algs/qmix.yaml", "r", encoding="utf-8") as handle:
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

    def test_training_does_not_construct_environment(self):
        env_info = self._env_info()
        scheme, groups, preprocess = build_episode_components(env_info)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            self.assertNotEqual(summary["parameter_hash_before"], summary["parameter_hash_after"])
            self.assertTrue(summary["dataset_shard_checksums_unchanged"])

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
        evaluator.close()
        self.assertTrue(created[0].closed)


if __name__ == "__main__":
    unittest.main()
