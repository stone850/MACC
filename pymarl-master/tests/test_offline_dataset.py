import tempfile
import unittest
from pathlib import Path

import torch as th

from components.episode_buffer import EpisodeBatch
from components.episode_schema import build_episode_components
from components.offline_dataset import OfflineDatasetWriter, OfflineEpisodeDataset, TRANSITION_FIELDS


class OfflineDatasetTest(unittest.TestCase):
    def _make_episode(self, scheme, groups, preprocess, length, marker):
        batch = EpisodeBatch(scheme, groups, 1, 6, preprocess=preprocess, device="cpu")
        batch.data.transition_data["filled"][0, :length + 1] = 1
        batch.data.transition_data["state"][0, :length + 1] = marker
        batch.data.transition_data["obs"][0, :length + 1] = marker
        batch.data.transition_data["avail_actions"][0, :length + 1] = 1
        batch.data.transition_data["actions"][0, :length + 1] = marker % 6
        batch.data.transition_data["reward"][0, :length] = marker
        batch.data.transition_data["terminated"][0, length - 1] = 1
        batch.data.transition_data["subtask_state"][0, :length + 1] = marker
        batch.data.transition_data["subtask_obs"][0, :length + 1] = marker
        batch.data.transition_data["subtask_visible"][0, :length + 1] = 1
        batch.data.transition_data["subtask_mask"][0, :length + 1] = 1
        batch.data.episode_data["subtask_id"][0] = th.tensor([3, 5, 7, 9])
        return batch

    def test_sharded_round_trip_and_sampling(self):
        env_info = {
            "state_shape": 24,
            "obs_shape": 24,
            "n_actions": 6,
            "n_agents": 4,
            "episode_limit": 5,
            "n_subtasks": 4,
            "subtask_state_shape": 3,
            "subtask_obs_shape": 3,
        }
        scheme, groups, preprocess = build_episode_components(env_info)
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary) / "dataset"
            writer = OfflineDatasetWriter(
                dataset_path,
                scheme,
                groups,
                6,
                {"dataset_version": "test", "env_info": env_info},
                shard_size=2,
                split_seed=7,
            )
            episodes = []
            for index, length in enumerate((2, 3, 4, 5, 2)):
                episode = self._make_episode(scheme, groups, preprocess, length, index + 1)
                episodes.append(episode)
                writer.add_episode(episode)
            writer.finalize()

            dataset = OfflineEpisodeDataset(dataset_path, split="all", seed=3, verify_checksums=True)
            self.assertEqual(len(dataset), 5)
            self.assertEqual(len(dataset.metadata["shards"]), 3)
            self.assertIn("actions_onehot", dataset.scheme)
            self.assertIn("filled", dataset.scheme)
            self.assertIsInstance(dataset.scheme["obs"]["vshape"], int)
            for episode_id, expected in enumerate(episodes):
                loaded = dataset.get_episode(episode_id)
                for field in TRANSITION_FIELDS:
                    self.assertTrue(th.equal(loaded[field], expected[field]))
                self.assertTrue(th.equal(loaded["subtask_id"], expected["subtask_id"]))
            sample = dataset.sample(3)
            self.assertIsInstance(sample, EpisodeBatch)
            self.assertEqual(sample.batch_size, 3)
            self.assertEqual(tuple(sample["actions_onehot"].shape), (3, 6, 4, 6))


if __name__ == "__main__":
    unittest.main()
