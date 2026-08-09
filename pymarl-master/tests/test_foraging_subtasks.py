import unittest

import numpy as np

from smac_plus.foraging import ForagingEnv


class ForagingSubtaskTest(unittest.TestCase):
    def setUp(self):
        self.env = ForagingEnv(
            field_size=10,
            players=4,
            max_food=4,
            force_coop=False,
            partially_observe=True,
            sight=2,
            is_print=False,
            seed=1,
            need_render=False,
        )
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_subtask_shapes_and_stable_ids(self):
        data = self.env.get_subtask_data()
        self.assertEqual(data["subtask_state"].shape, (4, 3))
        self.assertEqual(data["subtask_obs"].shape, (4, 4, 3))
        self.assertEqual(data["subtask_visible"].shape, (4, 4))
        self.assertEqual(data["subtask_mask"].shape, (4,))
        np.testing.assert_array_equal(data["subtask_id"], [3, 5, 7, 9])

    def test_aligned_obs_matches_original_compressed_obs(self):
        data = self.env.get_subtask_data()
        for agent, observation in enumerate(self.env.get_obs()):
            compressed = np.asarray(observation)[:12].reshape(4, 3)
            by_level = {int(food[2]): food for food in compressed if food[2] > 0}
            for slot, subtask_id in enumerate(data["subtask_id"]):
                if data["subtask_visible"][agent, slot]:
                    np.testing.assert_array_equal(data["subtask_obs"][agent, slot], by_level[int(subtask_id)])
                else:
                    np.testing.assert_array_equal(data["subtask_obs"][agent, slot], [-1, -1, 0])


if __name__ == "__main__":
    unittest.main()
