from smac.env.multiagentenv import MultiAgentEnv
import numpy as np
import gym
from gym.envs.registration import register
import lbforaging

class ForagingEnv(MultiAgentEnv):

    def __init__(self,
                 field_size: int,
                 players: int,
                 max_food: int,
                 force_coop: bool,
                 partially_observe: bool,
                 sight: int,
                 is_print: bool,
                 seed: int, 
                 need_render: bool):
        self.n_agents = players
        self.max_food = max_food
        self.field_size = field_size
        self.sight = sight if partially_observe else field_size
        self.n_actions = 6
        self._total_steps = 0
        self._episode_steps = 0
        self.NN = 0
        self.is_print = is_print
        self.need_render = need_render
        np.random.seed(seed)

        self.episode_limit = 50

        self.agent_score = np.zeros(players)

        env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}-v0".format(field_size, players, max_food,
                                                              "-coop" if force_coop else "",
                                                              "-{}s".format(sight) if partially_observe else "")
        if env_id not in gym.envs.registry.env_specs:
            register(
                id=env_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": players,
                    "max_player_level": 3,
                    "field_size": (field_size, field_size),
                    "max_food": max_food,
                    "sight": self.sight,
                    "max_episode_steps": 50,
                    "force_coop": force_coop,
                },
            )
        print(env_id)
        # env = gym.make("Foraging-8x8-2p-1f-coop-v0")
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self._initial_food_state = None
        self._subtask_ids = None
        self._subtask_level_to_slot = None

    def step(self, actions):
        # print('actions', actions.shape)
        """ Returns reward, terminated, info """
        self._total_steps += 1
        self._episode_steps += 1

        # For visualization, not complete in this version
        # if self.is_print:
        #     actionLog = open('./pics/actions.log', mode = 'a+', encoding='utf-8')
        #     actionLog.write('t_steps: %d\n' % self._episode_steps)
        #     actionLog.write('actions: %s\n' % str(actions.cpu().numpy()))

        # if self.need_render:
        #     fig = plt.figure()
        #     data = self.env.render(mode='rgb_array')
        #     plt.imshow(data)
        #     plt.axis('off')
        #     if not os.path.exists("./pics"):
        #         os.makedirs("./pics")
        #     fig.savefig("pics/game-{}.png".format(self.NN), bbox_inches='tight')
        #     self.NN += 1
        self.obs, rewards, dones, info, self.food_state, self.player_state = self.env.step(actions.cpu().numpy())

        # print('actions', actions.shape)
        # assert actions.shape[0] == 2
        # self.obs, rewards, dones, info = self.env.step(actions)
        self.agent_score += rewards
        # self.agent_score -= 0.002 / self.n_agents

        # reward = np.sum(rewards, axis=1)
        reward = np.sum(rewards)
        # step penalty
        reward -= 0.002
        terminated = np.all(dones)
        info = dict(info or {})
        all_food_collected = not np.any(self.get_subtask_mask())
        info["episode_limit"] = bool(
            terminated and self._episode_steps >= self.episode_limit and not all_food_collected
        )
        # TODO:
        # if reward > 0:
        #     terminated = True

        return reward, terminated, info

    def get_obs(self):
        # print('Im in get_obs')
        """ Returns all agent observations in a list """
        # print('obs', self.obs)
        return self.obs

    def get_obs_agent(self, agent_id):
        # print('Im in get_obs_agent')
        """ Returns observation for agent_id """
        return np.array(self.obs[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env._get_observation_space().shape[0]

    def get_state(self):
        state = self.player_state
        state = np.concatenate([state, self.food_state])
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        # print('self.env._obs_length', self.env._obs_length)
        return 3 * self.n_agents + 3 * self.max_food

    def get_subtask_data(self):
        if self._initial_food_state is None:
            raise RuntimeError("reset() must be called before requesting subtask data")

        subtask_state = np.asarray(self.food_state, dtype=np.float32).reshape(self.max_food, 3).copy()
        subtask_mask = (subtask_state[:, 2] > 0).astype(np.uint8)
        subtask_obs = np.zeros((self.n_agents, self.max_food, 3), dtype=np.float32)
        subtask_obs[..., 0:2] = -1
        subtask_visible = np.zeros((self.n_agents, self.max_food), dtype=np.uint8)
        player_state = np.asarray(self.player_state, dtype=np.float32).reshape(self.n_agents, 3)

        for agent_id, obs in enumerate(self.obs):
            local_foods = np.asarray(obs, dtype=np.float32)[:3 * self.max_food].reshape(self.max_food, 3)
            seen_slots = set()
            for local_food in local_foods:
                level = int(local_food[2])
                if level <= 0:
                    continue
                if level not in self._subtask_level_to_slot:
                    raise ValueError("Unknown local food level {}".format(level))

                slot = self._subtask_level_to_slot[level]
                if slot in seen_slots:
                    raise ValueError("Food {} appears twice in agent {} observation".format(level, agent_id))
                if not subtask_mask[slot] or int(subtask_state[slot, 2]) != level:
                    raise ValueError("Local food {} does not match active global slot {}".format(level, slot))

                crop_origin = np.maximum(player_state[agent_id, :2] - self.sight, 0)
                global_position = local_food[:2] + crop_origin
                if not np.array_equal(global_position, subtask_state[slot, :2]):
                    raise ValueError(
                        "Local/global food mismatch for agent {}, food {}: {} != {}".format(
                            agent_id, level, global_position.tolist(), subtask_state[slot, :2].tolist()
                        )
                    )

                subtask_obs[agent_id, slot] = local_food
                subtask_visible[agent_id, slot] = 1
                seen_slots.add(slot)

        return {
            "subtask_state": subtask_state,
            "subtask_obs": subtask_obs,
            "subtask_visible": subtask_visible,
            "subtask_mask": subtask_mask,
            "subtask_id": self._subtask_ids.copy(),
        }

    def get_subtask_states(self):
        return self.get_subtask_data()["subtask_state"]

    def get_agent_subtask_obs(self):
        return self.get_subtask_data()["subtask_obs"]

    def get_subtask_visibility(self):
        return self.get_subtask_data()["subtask_visible"]

    def get_subtask_mask(self):
        return (np.asarray(self.food_state).reshape(self.max_food, 3)[:, 2] > 0).astype(np.uint8)

    def get_subtask_ids(self):
        if self._subtask_ids is None:
            raise RuntimeError("reset() must be called before requesting subtask IDs")
        return self._subtask_ids.copy()

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        res = [0] * self.n_actions
        t = self.env._valid_actions[self.env.players[agent_id]]
        for i in range(len(t)):
            res[t[i].value] = 1
        return res

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        self.agent_score = np.zeros(self.n_agents)
        # self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.obs, self.food_state, self.player_state = self.env.reset()
        self._initial_food_state = np.asarray(self.food_state, dtype=np.float32).reshape(self.max_food, 3).copy()
        levels = self._initial_food_state[:, 2].astype(np.int64)
        if np.any(levels <= 0) or len(np.unique(levels)) != self.max_food:
            raise ValueError("LBF subtask alignment requires unique positive food levels, got {}".format(levels.tolist()))
        self._subtask_ids = levels
        self._subtask_level_to_slot = {int(level): slot for slot, level in enumerate(levels)}
        return self.get_obs(), self.get_state()

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "n_subtasks": self.max_food,
                    "subtask_state_shape": 3,
                    "subtask_obs_shape": 3}
        return env_info

    def get_stats(self):
        stats = {
            "agent_score": self.agent_score,
        }
        return stats
