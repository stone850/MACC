import unittest

import torch as th

from learners.offline_diagnostics import compute_offline_q_diagnostics


class SumMixer(th.nn.Module):
    def forward(self, agent_qs, states):
        del states
        return agent_qs.sum(dim=2, keepdim=True)


class OfflineQDiagnosticsTest(unittest.TestCase):
    def test_metrics_mask_padding_and_unavailable_actions(self):
        mac_out = th.tensor([[[
            [1.0, 4.0, 3.0],
            [5.0, 2.0, 1.0],
        ], [
            [10.0, 9.0, 8.0],
            [1.0, 7.0, 6.0],
        ], [
            [100.0, 200.0, 300.0],
            [400.0, 500.0, 600.0],
        ], [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]]])
        actions = th.tensor([[[[0], [0]], [[2], [1]], [[0], [0]]]])
        avail_actions = th.tensor([[[
            [1, 1, 1],
            [1, 1, 1],
        ], [
            [0, 1, 1],
            [1, 1, 1],
        ], [
            [0, 0, 0],
            [0, 0, 0],
        ], [
            [0, 0, 0],
            [0, 0, 0],
        ]]])
        states = th.zeros(1, 4, 1)
        data_q_tot = th.tensor([[[6.0], [15.0], [999.0]]])
        targets = th.tensor([[[7.0], [17.0], [999.0]]])
        td_error = data_q_tot - targets
        mask = th.tensor([[[1.0], [1.0], [0.0]]])

        metrics = compute_offline_q_diagnostics(
            mac_out=mac_out,
            actions=actions,
            avail_actions=avail_actions,
            mixer_states=states,
            mixer=SumMixer(),
            data_q_tot=data_q_tot,
            targets=targets,
            td_error=td_error,
            mask=mask,
        )

        self.assertAlmostEqual(metrics["q_data_mean"], 10.5)
        self.assertAlmostEqual(metrics["q_max_mean"], 12.5)
        self.assertAlmostEqual(metrics["q_gap_mean"], 2.0)
        self.assertAlmostEqual(metrics["q_tot_abs_max"], 16.0)
        self.assertAlmostEqual(metrics["target_mean"], 12.0)
        self.assertAlmostEqual(metrics["target_abs_max"], 17.0)
        self.assertAlmostEqual(metrics["td_error_abs_mean"], 1.5)
        self.assertAlmostEqual(metrics["ood_action_rate"], 0.5)
        self.assertEqual(
            metrics["ood_action_rate"], metrics["greedy_action_disagreement_rate"]
        )

    def test_valid_timestep_requires_an_available_action(self):
        mac_out = th.zeros(1, 2, 1, 2)
        with self.assertRaisesRegex(ValueError, "no available action"):
            compute_offline_q_diagnostics(
                mac_out=mac_out,
                actions=th.zeros(1, 1, 1, 1, dtype=th.long),
                avail_actions=th.zeros(1, 2, 1, 2, dtype=th.long),
                mixer_states=th.zeros(1, 2, 1),
                mixer=SumMixer(),
                data_q_tot=th.zeros(1, 1, 1),
                targets=th.zeros(1, 1, 1),
                td_error=th.zeros(1, 1, 1),
                mask=th.ones(1, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
