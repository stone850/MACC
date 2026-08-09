import argparse
import json
from pathlib import Path

import numpy as np
import torch as th

from components.offline_dataset import (
    EPISODE_FIELDS,
    TRANSITION_FIELDS,
    OfflineEpisodeDataset,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_splits(dataset_path, n_episodes):
    with open(Path(dataset_path) / "split.json", "r", encoding="utf-8") as handle:
        splits = json.load(handle)
    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    all_ids = set(range(n_episodes))
    _require(not (train & validation or train & test or validation & test), "Dataset splits overlap")
    _require(train | validation | test == all_ids, "Dataset splits do not cover all episodes")
    _require(set(splits["all"]) == all_ids, "The all split is incomplete")


def _check_round_trip(dataset, episode_id, batch):
    raw = dataset.get_raw_episode(episode_id)
    for field in TRANSITION_FIELDS:
        _require(
            th.equal(raw["transition_data"][field], batch[field][0].cpu()),
            "Episode {} field '{}' changed after loading".format(episode_id, field),
        )
    for field in EPISODE_FIELDS:
        _require(
            th.equal(raw["episode_data"][field], batch[field][0].cpu()),
            "Episode {} field '{}' changed after loading".format(episode_id, field),
        )

    expected_onehot = th.zeros_like(batch["actions_onehot"])
    expected_onehot.scatter_(-1, batch["actions"].long(), 1)
    _require(th.equal(expected_onehot, batch["actions_onehot"]), "actions_onehot preprocess is incorrect")


def _check_episode(dataset, episode_id):
    batch = dataset.get_episode(episode_id)
    _check_round_trip(dataset, episode_id, batch)
    env_config = dataset.metadata["environment_config"]
    env_info = dataset.metadata["env_info"]
    n_agents = int(env_info["n_agents"])
    n_subtasks = int(env_info["n_subtasks"])
    episode_limit = int(env_info["episode_limit"])
    field_size = int(env_config["field_size"])
    sight = int(env_config["sight"] if env_config["partially_observe"] else field_size)

    filled = batch["filled"][0, :, 0].cpu()
    _require(th.all((filled == 0) | (filled == 1)).item(), "filled is not binary in episode {}".format(episode_id))
    valid_state_count = int(filled.sum().item())
    _require(2 <= valid_state_count <= episode_limit + 1, "Invalid filled length in episode {}".format(episode_id))
    _require(th.all(filled[:valid_state_count] == 1).item(), "filled has a gap in episode {}".format(episode_id))
    _require(th.all(filled[valid_state_count:] == 0).item(), "filled padding is nonzero in episode {}".format(episode_id))
    episode_length = valid_state_count - 1

    for field in TRANSITION_FIELDS:
        if field == "filled":
            continue
        padding = batch[field][0, valid_state_count:].cpu()
        _require(th.count_nonzero(padding).item() == 0, "Padding for '{}' is nonzero in episode {}".format(field, episode_id))
    _require(th.count_nonzero(batch["reward"][0, episode_length]).item() == 0, "Final state has a reward")
    _require(th.count_nonzero(batch["terminated"][0, episode_length]).item() == 0, "Final state is marked terminated")

    terminated = batch["terminated"][0, :episode_length, 0].bool().cpu()
    _require(not terminated[:-1].any().item(), "terminated occurs before the final transition")
    final_mask = batch["subtask_mask"][0, episode_length].cpu()
    natural_terminal = not final_mask.any().item()
    if episode_length < episode_limit:
        _require(natural_terminal, "Short episode ended while food remained")
    _require(bool(terminated[-1].item()) == natural_terminal, "Natural/time-limit termination is incorrect")

    actions = batch["actions"][0, :valid_state_count, :, 0].long().cpu()
    available = batch["avail_actions"][0, :valid_state_count].cpu()
    selected_available = th.gather(available, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    _require(th.all(selected_available == 1).item(), "An action is unavailable in episode {}".format(episode_id))

    subtask_ids = batch["subtask_id"][0].long().cpu()
    _require(len(th.unique(subtask_ids)) == n_subtasks, "subtask_id is not unique")
    _require(th.all(subtask_ids > 0).item(), "subtask_id must be positive")
    subtask_state = batch["subtask_state"][0, :valid_state_count].cpu()
    subtask_mask = batch["subtask_mask"][0, :valid_state_count].long().cpu()
    subtask_obs = batch["subtask_obs"][0, :valid_state_count].cpu()
    subtask_visible = batch["subtask_visible"][0, :valid_state_count].long().cpu()
    initial_subtasks = subtask_state[0].clone()
    _require(th.all(subtask_mask[0] == 1).item(), "Not all food slots are active at reset")
    _require(th.equal(initial_subtasks[:, 2].long(), subtask_ids), "subtask_id does not match initial food level")
    _require(th.all(subtask_mask[1:] <= subtask_mask[:-1]).item(), "A subtask mask changed from 0 back to 1")

    for timestep in range(valid_state_count):
        for slot in range(n_subtasks):
            if subtask_mask[timestep, slot]:
                _require(
                    th.equal(subtask_state[timestep, slot], initial_subtasks[slot]),
                    "Global subtask slot drifted in episode {}, t={}, slot={}".format(episode_id, timestep, slot),
                )
            else:
                _require(
                    th.count_nonzero(subtask_state[timestep, slot]).item() == 0,
                    "Inactive subtask state is not zero in episode {}, t={}, slot={}".format(episode_id, timestep, slot),
                )

    player_state = batch["state"][0, :valid_state_count, :3 * n_agents].reshape(valid_state_count, n_agents, 3).cpu()
    state_subtasks = batch["state"][0, :valid_state_count, 3 * n_agents:].reshape(
        valid_state_count, n_subtasks, 3
    ).cpu()
    _require(th.equal(state_subtasks, subtask_state), "state food slice does not match subtask_state")
    _require(th.all(player_state[:, :, 2] == player_state[0:1, :, 2]).item(), "Player slots or levels drifted")
    movement = th.abs(player_state[1:, :, :2] - player_state[:-1, :, :2]).sum(dim=-1)
    _require(th.all(movement <= 1).item(), "State sequence contains an impossible player movement")

    original_obs = batch["obs"][0, :valid_state_count].cpu()
    invisible_value = th.tensor([-1.0, -1.0, 0.0])
    for timestep in range(valid_state_count):
        compressed_foods = original_obs[timestep, :, :3 * n_subtasks].reshape(n_agents, n_subtasks, 3)
        for agent in range(n_agents):
            agent_position = player_state[timestep, agent, :2]
            crop_start = th.clamp(agent_position - sight, min=0)
            crop_end = th.minimum(agent_position + sight, th.tensor([field_size - 1.0, field_size - 1.0]))
            compressed_by_id = {
                int(food[2].item()): food
                for food in compressed_foods[agent]
                if food[2].item() > 0
            }
            observed_count = sum(food[2].item() > 0 for food in compressed_foods[agent])
            _require(len(compressed_by_id) == observed_count, "Original observation contains duplicate food IDs")
            expected_visible_ids = set()
            for slot in range(n_subtasks):
                active = bool(subtask_mask[timestep, slot].item())
                global_food = subtask_state[timestep, slot]
                visible = active and bool(th.all(global_food[:2] >= crop_start).item()) and bool(
                    th.all(global_food[:2] <= crop_end).item()
                )
                _require(
                    bool(subtask_visible[timestep, agent, slot].item()) == visible,
                    "Visibility mismatch in episode {}, t={}, agent={}, slot={}".format(
                        episode_id, timestep, agent, slot
                    ),
                )
                if visible:
                    expected_visible_ids.add(int(subtask_ids[slot].item()))
                    expected_obs = th.cat((global_food[:2] - crop_start, global_food[2:3]))
                    _require(
                        th.equal(subtask_obs[timestep, agent, slot], expected_obs),
                        "Aligned local observation mismatch in episode {}, t={}, agent={}, slot={}".format(
                            episode_id, timestep, agent, slot
                        ),
                    )
                    subtask_id = int(subtask_ids[slot].item())
                    _require(subtask_id in compressed_by_id, "Visible food is missing from original observation")
                    _require(th.equal(compressed_by_id[subtask_id], expected_obs), "Original/aligned food observation mismatch")
                else:
                    _require(
                        th.equal(subtask_obs[timestep, agent, slot], invisible_value),
                        "Invisible subtask does not use the sentinel observation",
                    )
            _require(
                set(compressed_by_id) == expected_visible_ids,
                "Original observation visible-food set does not match global state",
            )

    collection_events = int((subtask_mask[:-1] - subtask_mask[1:]).sum().item())
    return {
        "episode_length": episode_length,
        "episode_return": float(batch["reward"][0, :episode_length].sum().item()),
        "collection_events": collection_events,
        "natural_terminal": natural_terminal,
    }


def check_dataset(dataset_path, require_collection_event=False):
    dataset = OfflineEpisodeDataset(dataset_path, split="all", seed=0, verify_checksums=True)
    if dataset.metadata.get("dataset_version") == "lbf_episode_v2":
        _require(dataset.metadata.get("git_dirty") is False, "v2 dataset was not collected from a clean repository")
        runtime_versions = dataset.metadata.get("runtime_versions", {})
        for package in ("python", "torch", "numpy", "gym", "lbforaging", "sacred"):
            _require(runtime_versions.get(package), "v2 metadata is missing runtime version '{}'".format(package))
        if dataset.metadata.get("quality_label") == "medium":
            selection = dataset.metadata.get("checkpoint_selection", {})
            _require(selection.get("evaluation_t") is not None, "Medium dataset lacks checkpoint evaluation step")
            _require(selection.get("evaluation_return") is not None, "Medium dataset lacks checkpoint evaluation return")
    _check_splits(dataset_path, int(dataset.metadata["n_episodes"]))
    summaries = [_check_episode(dataset, episode_id) for episode_id in dataset.episode_ids]
    collection_events = sum(item["collection_events"] for item in summaries)
    with open(Path(dataset_path) / "statistics.json", "r", encoding="utf-8") as handle:
        stored_statistics = json.load(handle)
    lengths = [item["episode_length"] for item in summaries]
    returns = [item["episode_return"] for item in summaries]
    natural_terminations = sum(item["natural_terminal"] for item in summaries)
    _require(stored_statistics["n_episodes"] == len(summaries), "statistics episode count is incorrect")
    _require(stored_statistics["natural_terminations"] == natural_terminations, "termination statistics are incorrect")
    _require(stored_statistics["episode_lengths"]["min"] == min(lengths), "minimum length statistic is incorrect")
    _require(stored_statistics["episode_lengths"]["max"] == max(lengths), "maximum length statistic is incorrect")
    _require(np.isclose(stored_statistics["returns"]["mean"], np.mean(returns)), "mean return statistic is incorrect")
    if require_collection_event:
        _require(collection_events > 0, "Dataset contains no food collection event")
    return {
        "n_episodes": len(summaries),
        "collection_events": collection_events,
        "natural_terminations": natural_terminations,
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def debug_episode(dataset_path, episode):
    dataset = OfflineEpisodeDataset(dataset_path, split="all", seed=0)
    if episode == "random":
        episode_id = int(np.random.RandomState(0).choice(dataset.episode_ids))
    else:
        episode_id = int(episode)
    batch = dataset.get_episode(episode_id)
    valid_state_count = int(batch["filled"][0].sum().item())
    print("episode_id={}".format(episode_id))
    print("subtask_id={}".format(batch["subtask_id"][0].tolist()))
    for timestep in range(valid_state_count):
        print("t={}".format(timestep))
        print("  subtask_state={}".format(batch["subtask_state"][0, timestep].tolist()))
        print("  subtask_mask={}".format(batch["subtask_mask"][0, timestep].tolist()))
        for agent in range(batch.groups["agents"]):
            print("  agent={}".format(agent))
            print("    subtask_visible={}".format(batch["subtask_visible"][0, timestep, agent].tolist()))
            print("    subtask_obs={}".format(batch["subtask_obs"][0, timestep, agent].tolist()))


def main():
    parser = argparse.ArgumentParser(description="Validate a frozen LBF offline episode dataset")
    parser.add_argument("dataset_path")
    parser.add_argument("--require-collection-event", action="store_true")
    parser.add_argument("--debug-episode", metavar="ID_OR_RANDOM")
    args = parser.parse_args()
    summary = check_dataset(args.dataset_path, require_collection_event=args.require_collection_event)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["collection_events"] == 0:
        print("WARNING: no food collection event was present; mask disappearance was not exercised")
    if args.debug_episode is not None:
        debug_episode(args.dataset_path, args.debug_episode)


if __name__ == "__main__":
    main()
