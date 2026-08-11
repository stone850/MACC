import math
import unittest
from pathlib import Path
from types import SimpleNamespace as SN

import torch as th
import yaml

from components.episode_buffer import EpisodeBatch
from components.episode_schema import build_episode_components
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as learner_REGISTRY
from run_offline import OFFLINE_AUXILIARY_METRICS, OFFLINE_TRAINING_METRICS


class SilentConsole:
    def info(self, *args, **kwargs):
        pass


class SilentLearnerLogger:
    console_logger = SilentConsole()

    def log_stat(self, *args, **kwargs):
        pass


class OfflineLatentQLearnerTest(unittest.TestCase):
    def _args(self):
        root = Path(__file__).resolve().parents[1] / "src/config"
        config = {}
        for path in ("default.yaml", "algs/macc.yaml", "offline.yaml"):
            with open(root / path, "r", encoding="utf-8") as handle:
                config.update(yaml.safe_load(handle))
        config.update({
            "env": "foraging",
            "env_args": {
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
            "n_agents": 4,
            "n_actions": 6,
            "state_shape": 24,
            "device": "cpu",
            "use_cuda": False,
            "use_offline_training": True,
            "target_update_interval": 1,
        })
        return SN(**config)

    def _batch(self):
        env_info = {
            "state_shape": 24,
            "obs_shape": 24,
            "n_actions": 6,
            "n_agents": 4,
            "episode_limit": 4,
            "n_subtasks": 4,
            "subtask_state_shape": 3,
            "subtask_obs_shape": 3,
        }
        scheme, groups, preprocess = build_episode_components(env_info)
        batch = EpisodeBatch(scheme, groups, 2, 5, preprocess=preprocess, device="cpu")
        generator = th.Generator().manual_seed(7)
        batch.data.transition_data["state"].copy_(th.randn(2, 5, 24, generator=generator))
        batch.data.transition_data["obs"].copy_(th.randn(2, 5, 4, 24, generator=generator))
        batch.data.transition_data["filled"][0, :5] = 1
        batch.data.transition_data["filled"][1, :3] = 1
        batch.data.transition_data["terminated"][0, 3] = 1
        batch.data.transition_data["terminated"][1, 1] = 1
        batch.data.transition_data["avail_actions"][0, :5] = 1
        batch.data.transition_data["avail_actions"][1, :3] = 1
        batch.data.transition_data["actions_onehot"][:, :, :, 0] = 1
        return batch

    def test_offline_batch_returns_finite_metrics_and_augmented_mixer_state(self):
        th.manual_seed(1)
        args = self._args()
        env_info = {
            "state_shape": 24,
            "obs_shape": 24,
            "n_actions": 6,
            "n_agents": 4,
            "episode_limit": 4,
            "n_subtasks": 4,
            "subtask_state_shape": 3,
            "subtask_obs_shape": 3,
        }
        _, groups, _ = build_episode_components(env_info)
        batch = self._batch()
        mac = mac_REGISTRY[args.mac](batch.scheme, groups, args)
        learner = learner_REGISTRY[args.learner](
            mac, batch.scheme, SilentLearnerLogger(), args
        )
        parameter_before = next(mac.parameters()).detach().clone()

        metrics = learner.train(batch, t_env=1, episode_num=1)

        self.assertEqual(tuple(learner.new_states.shape), (2, 5, 256))
        self.assertTrue(metrics["target_updated"])
        self.assertTrue(set(OFFLINE_TRAINING_METRICS).issubset(metrics))
        self.assertTrue(set(OFFLINE_AUXILIARY_METRICS).issubset(metrics))
        for key in ("loss",) + OFFLINE_TRAINING_METRICS + OFFLINE_AUXILIARY_METRICS:
            self.assertTrue(math.isfinite(float(metrics[key])), key)
        self.assertAlmostEqual(
            metrics["loss"],
            metrics["td_loss"] + metrics["representation_loss"],
            places=5,
        )
        self.assertFalse(th.equal(parameter_before, next(mac.parameters()).detach()))


if __name__ == "__main__":
    unittest.main()
