import torch as th


def compute_offline_q_diagnostics(
    mac_out,
    actions,
    avail_actions,
    mixer_states,
    mixer,
    data_q_tot,
    targets,
    td_error,
    mask,
):
    if mixer is None:
        raise ValueError("Offline QMIX diagnostics require a mixer")

    current_q = mac_out[:, :-1].detach()
    current_avail_actions = avail_actions[:, :-1]
    valid_agent_mask = mask.bool().expand(-1, -1, current_q.size(2))
    available_action_counts = current_avail_actions.sum(dim=3)
    if th.any((available_action_counts == 0) & valid_agent_mask):
        raise ValueError("A valid offline timestep has no available action")

    masked_q = current_q.masked_fill(current_avail_actions == 0, -9999999)
    greedy_actions = masked_q.max(dim=3, keepdim=True)[1]
    greedy_agent_qvals = th.gather(current_q, dim=3, index=greedy_actions).squeeze(3)
    with th.no_grad():
        greedy_q_tot = mixer(greedy_agent_qvals, mixer_states[:, :-1].detach())

    valid_transition_mask = mask.bool()
    data_values = data_q_tot.detach().masked_select(valid_transition_mask)
    greedy_values = greedy_q_tot.masked_select(valid_transition_mask)
    target_values = targets.detach().masked_select(valid_transition_mask)
    td_error_values = td_error.detach().masked_select(valid_transition_mask)
    if data_values.numel() == 0:
        raise ValueError("Offline batch contains no valid transitions")

    data_actions = actions.squeeze(3)
    greedy_action_values = greedy_actions.squeeze(3)
    disagreements = (greedy_action_values != data_actions) & valid_agent_mask
    disagreement_rate = disagreements.float().sum() / valid_agent_mask.float().sum()

    return {
        "q_data_mean": float(data_values.mean().item()),
        "q_max_mean": float(greedy_values.mean().item()),
        "q_gap_mean": float((greedy_values - data_values).mean().item()),
        "q_tot_abs_max": float(
            th.maximum(data_values.abs().max(), greedy_values.abs().max()).item()
        ),
        "target_mean": float(target_values.mean().item()),
        "target_abs_max": float(target_values.abs().max().item()),
        "td_error_abs_mean": float(td_error_values.abs().mean().item()),
        "ood_action_rate": float(disagreement_rate.item()),
        "greedy_action_disagreement_rate": float(disagreement_rate.item()),
    }
