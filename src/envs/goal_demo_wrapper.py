import numpy as np
import gym
from gym.spaces import Box, Dict

from utils.helpers import extract_observable_state  # get_raw_state_size
from utils.load_confs import load_parameters, load_paths
params = load_parameters()


def goal_distance(goal_a, goal_b, dim_scale=True, time_feat=False):
    # if dim_scale is True, scale distance by a factor of sqrt(d) so that the distance is invariant to dimensionality
    # of the goals. By using a dimension invariant distance, we can use a constant distance threshold to determine whether
    # a state is equal to the goal, across all environments.
    # if time_feat is True (assumed to be last feature), ignore it
    assert goal_a.shape == goal_b.shape, f"goal a shape: {goal_a.shape}, goal b shape: {goal_b.shape}"
    if time_feat:
        goal_a = goal_a[:-1]
        goal_b = goal_b[:-1]

    if dim_scale:
        dist = np.linalg.norm(goal_a - goal_b, axis=-1,
                              ord=2) / np.sqrt(goal_a.shape[0])
        return dist

    else:
        return np.linalg.norm(goal_a - goal_b, axis=-1, ord=2)


class GoalDemoWrapper(gym.Wrapper):
    def __init__(self, env, env_id, expert_demo, raw=False, displace_t=0,
                 reward_type='sparse', distance_threshold=1e-4, time_feat=False, sin_cos_repr=False,
                 potential_coef=1):
        '''
        This wrapper makes MuJoCo continuous control environments into GoalEnvs, where the expert 
        demonstrations provide the goals. Satisfies requirements of a Gym goalenv without actually inheriting 
        from it. 

        displace_t: if agent's state is s_t, append expert state s^e_{t+displace_t}
        '''
        super(GoalDemoWrapper, self).__init__(env)
        self.env = env
        self.env_id = env_id

        self.expert_demo = expert_demo
        assert len(self.expert_demo.shape) == 2
        self.n_demo_timesteps = self.expert_demo.shape[0]
        self.displace_t = displace_t

        self.raw = raw
        if raw:
            obs_type = "raw"
        else:
            obs_type = "full"

        self.discount = params['sac'][env_id.strip("-v2").lower()]["gamma"]

        # load reward params
        self.reward_type = reward_type
        self.reward_params = params['rew_shaping'][env_id.strip(
            "-v2").lower()][obs_type]
        self.dist_std = self.reward_params["dist_std"]
        self.imit_rew_coef = self.reward_params["imit_rew_coef"]
        self.gamma = self.reward_params["gamma"]
        self.alpha = self.reward_params["alpha"]
        self.terminal_phi0 = params['pbrs']['terminal_phi0']

        self.distance_threshold = distance_threshold

        # correct observation space size
        self.expert_state_size = self.expert_demo.shape[1]
        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.shape[0]
        self.time_feat = time_feat
        self.sin_cos_repr = sin_cos_repr
        self.potential_coef = potential_coef
        # self.obs_size += 1 # taken care of obs size returned by time feat wrapper
        # self.observation_space = Box(low=-np.inf, high=np.inf,
        #                                         shape=(self.obs_size + self.expert_state_size, ),
        #                                         dtype=np.float32)

        self.observation_space = Dict(dict(
                                      desired_goal=Box(low=-np.inf, high=np.inf,
                                                       shape=(
                                                           self.expert_state_size, ),
                                                       dtype=np.float32),
                                      achieved_goal=Box(low=-np.inf, high=np.inf,
                                                        shape=(
                                                            self.expert_state_size, ),
                                                        dtype=np.float32),
                                      observation=Box(low=-np.inf, high=np.inf,
                                                      shape=(self.obs_size, ),
                                                      dtype=np.float32)
                                      ))
        self.replay_buffer_obs_space = Dict(dict(
            desired_goal=Box(low=-np.inf, high=np.inf,
                             shape=(
                                 self.expert_state_size, ),
                             dtype=np.float32),
            observation=Box(low=-np.inf, high=np.inf,
                            shape=(self.obs_size, ),
                            dtype=np.float32)
        ))

        self.current_t = 0

    def get_expert_demo(self, ob):
        '''At time t, append expert's state at time t+displace_t'''
        if (self.current_t + self.displace_t) > self.n_demo_timesteps - 1:
            # append expert state
            self.expert_t = self.n_demo_timesteps - 1
        else:
            self.expert_t = self.current_t + self.displace_t
        self.demo_ob = self.expert_demo[self.expert_t]

        self.current_t += 1

    def compute_reward(self, achieved_goal_tp1, desired_goal_t, info, eval_mode=False):
        """
        TODO: REFACTOR desired_goal_t to desired_goal_tp1
        NOTE: If eval_mode, mutate the info dict to contain the goal reaching information

        Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """        # here the achieved goal is the achieved state at time t+1
        if self.reward_type == 'sparse':
            d = goal_distance(achieved_goal_tp1, desired_goal_t,
                              dim_scale=True, time_feat=self.time_feat)
            total_rew = -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            d = goal_distance(achieved_goal_tp1, desired_goal_t,
                              dim_scale=True, time_feat=self.time_feat)
            total_rew = -d
        elif self.reward_type == 'huber+env':
            # tuning occured with no dimension scaling
            d = goal_distance(achieved_goal_tp1, desired_goal_t,
                              dim_scale=False, time_feat=self.time_feat)
            sq_dist = np.square(d) / self.dist_std
            imit_rew = - (self.alpha * sq_dist) - \
                ((1 - self.alpha) * np.sqrt(self.gamma + sq_dist))
            total_rew = ((1 - self.imit_rew_coef) *
                         info["env_rew"]) + (self.imit_rew_coef * imit_rew)
        elif self.reward_type == 'potential_dense':
            if info["done"] and self.terminal_phi0:
                d_tp1 = 0
            else:
                d_tp1 = goal_distance(
                    achieved_goal_tp1, desired_goal_t, dim_scale=True, time_feat=self.time_feat)
            d_t = goal_distance(info["prev_ob"]["achieved_goal"], info["prev_ob"]
                                ["desired_goal"], dim_scale=True, time_feat=self.time_feat)
            # (s_t | g_t, a_t, r_t, s_t+1 | g_t+1) <- example of a transition with goals
            # note that we need a negative distance here
            potential = - self.discount * d_tp1 + d_t
            total_rew = info["env_rew"] + self.potential_coef * potential
        elif self.reward_type == 'env':
            total_rew = info["env_rew"]
        else:
            print(f"Reward type {self.reward_type} not supported. ")

        if eval_mode:  # Mutate info dict
            # compute potential_corrected here
            if info["done"] and self.terminal_phi0:
                d_tp1 = 0
            else:
                d_tp1 = goal_distance(
                    achieved_goal_tp1, desired_goal_t, dim_scale=True, time_feat=self.time_feat)
            d_t = goal_distance(info["prev_ob"]["achieved_goal"], info["prev_ob"]
                                ["desired_goal"], dim_scale=True, time_feat=self.time_feat)
            # compute potential_dense
            potential = - self.discount * d_tp1 + d_t
            shaped_rew = info["env_rew"] + self.potential_coef * potential
            info["goal_dist"] = d_tp1
            info["shaped_rew"] = shaped_rew

        return total_rew

    def _get_obs(self, agent_ob):
        # HER env wrapper will concatenate all of these
        return {
            'observation': agent_ob.copy(),
            'achieved_goal': extract_observable_state(agent_ob, self.env_id, self.time_feat,
                                                      sin_cos_repr=self.sin_cos_repr) if self.raw else agent_ob.copy(),
            'desired_goal': self.demo_ob.copy()
        }

    def reset(self):
        ob = self.env.reset()
        self.current_t = 0
        self.get_expert_demo(ob)  # sets current_t = 1
        obs_dict = self._get_obs(ob)
        self.prev_obs_dict = obs_dict
        return obs_dict

    def step(self, action, eval_mode=False):
        ob, env_rew, done, env_rew_info = self.env.step(
            action)  # discard env reward for now
        self.get_expert_demo(ob)

        obs_dict = self._get_obs(ob)
        info = {**env_rew_info,
                "env_rew": env_rew,
                "prev_ob": self.prev_obs_dict,
                "done": done
                }
        # TODO: refactor desired_goal_t to desired_goal_tp1
        reward = self.compute_reward(achieved_goal_tp1=obs_dict['achieved_goal'],
                                     desired_goal_t=obs_dict['desired_goal'],
                                     info=info,
                                     eval_mode=eval_mode)  # info will be mutated if eval_mode is True
        self.prev_obs_dict = obs_dict
        return obs_dict, reward, done, info
