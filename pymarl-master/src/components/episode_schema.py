import torch as th

from components.transforms import OneHot


def build_episode_components(env_info):
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    if "n_subtasks" in env_info:
        n_subtasks = env_info["n_subtasks"]
        scheme.update({
            "subtask_state": {
                "vshape": (n_subtasks, env_info["subtask_state_shape"]),
            },
            "subtask_obs": {
                "vshape": (n_subtasks, env_info["subtask_obs_shape"]),
                "group": "agents",
            },
            "subtask_visible": {
                "vshape": (n_subtasks,),
                "group": "agents",
                "dtype": th.uint8,
            },
            "subtask_mask": {
                "vshape": (n_subtasks,),
                "dtype": th.uint8,
            },
            "subtask_id": {
                "vshape": (n_subtasks,),
                "dtype": th.long,
                "episode_const": True,
            },
        })

    groups = {"agents": env_info["n_agents"]}
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=env_info["n_actions"])])
    }
    return scheme, groups, preprocess
