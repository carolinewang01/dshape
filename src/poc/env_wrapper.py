import math
import numpy as np
from gym import Wrapper, spaces
from scipy.spatial import cKDTree as KDTree


def manhattan_dist(state0, state1):
    dist = abs(state0[0] - state1[0]) + abs(state0[1] - state1[1])
    return dist

def euclidean_dist(state0, state1):
    dist = math.sqrt((state0[0] - state1[0])**2 + (state0[1] - state1[1])**2)
    return dist

def normalize(x, size):
    return x / size # size of one side of gridworld

def compute_sbs_phi(exp_kdtree, query_state, dist_scale):
    # k for k-nearest neighbors, p for p-norm
    # find expert state closest to query state, use p=1 (Manhattan distance) for gridworld
    dist, idx = exp_kdtree.query(query_state, k=1, p=1, n_jobs=1) 
    similarity = np.exp(- 0.5 * (dist**2) * dist_scale)
    return similarity


class DemoWrappedGridworld(Wrapper):
    def __init__(self, env, demo=None, reward_type=None, rew_coef=1, potential_const=0, dist_scale=1, gamma=1, 
        negate_potential=False, state_aug=False, termphi0=True):
        super(DemoWrappedGridworld, self).__init__(env)

        self.env = env
        self.state_aug = state_aug
        self.demo = demo
        self.set_env_attrs()

        # 
        self.rew_coef = rew_coef
        self.potential_const = potential_const
        self.gamma = gamma

        self.reward_type = reward_type
        assert reward_type in [None, "ng_phi0", "ng_0.5phi0", "pbrs_demo", "pbrs_demo_euclidean", "sbs", "manhattan"]
        self.potential_fcn = None
        # assumes static gold location
        if self.reward_type == "ng_phi0":
            self.potential_fcn = lambda state: - manhattan_dist(state, self.env.goal ) / 0.8
        elif self.reward_type == "ng_0.5phi0":
            self.potential_fcn = lambda state: - 0.5 * (manhattan_dist(state, self.env.goal) / 0.8)
        elif self.reward_type == "pbrs_demo":
            def f(state, goal):
                return - manhattan_dist(state, goal) + self.potential_const
            self.potential_fcn = f

        elif self.reward_type == "pbrs_demo_euclidean":
            def f(state, goal):
                return - euclidean_dist(state, goal) + self.potential_const
            self.potential_fcn = f

        elif self.reward_type == "sbs":
            self.dist_scale = dist_scale
            self.exp_kdtree = KDTree(normalize(np.array(self.demo), self.env.size))

        elif self.reward_type == "manhattan":
            self.dist_rew = lambda state, goal: -manhattan_dist(state, goal)

        # whether to negate the potential
        self.negate_potential = negate_potential
        if self.negate_potential: # i don't think this line works
            self.potential_fcn = lambda args: -self.potential_fcn(args) 

        # whether to set potnetial of terminal state to 0
        self.termphi0 = termphi0

    def set_env_attrs(self):
        self.size = self.env.size
        if self.state_aug is True:
            self.n_dims = self.env.n_dims * 2
            self.observation_space = spaces.Tuple([space for space in self.env.observation_space.spaces] + [spaces.Discrete(self.size), spaces.Discrete(self.size)])
        else: 
            self.n_dims = self.env.n_dims

        self.get_available_actions = self.env.get_available_actions # unwrap?

    def concat_demo(self, state, t):
        demo_state = self.demo[t]
        return (*state, *demo_state)

    def reset(self):
        self.ts = 0
        state = self.env.reset()
        if self.state_aug:
            state = self.concat_demo(state, self.ts)
        self.old_state = state
        return state

    def step(self, action):
        new_state, rew, done, info = self.env.step(action)
        info["env_rew"] = rew
        # shaping 
        shaped_rew = self.compute_reward(rew, self.old_state, new_state, done,
                                       old_goal_state = self.demo[self.ts] if self.demo is not None else None, 
                                       new_goal_state=self.demo[self.ts+1] if self.demo is not None else None
                                       )

        if self.state_aug:
            new_state = self.concat_demo(new_state, self.ts + 1)
            
        self.ts += 1
        self.old_state = new_state
        return new_state, shaped_rew, done, info

    def compute_reward(self, reward, old_state, new_state, done, old_goal_state=None, new_goal_state=None):
        """Conducts potential based reward shaping""" 
        if self.reward_type is None:
            F = 0
        if self.reward_type in ["pbrs_demo", "pbrs_demo_euclidean"]:
            if done and self.termphi0: # set potential of new state to 0
                F = - self.potential_fcn(old_state, old_goal_state)
            else:
                F = self.potential_fcn(new_state, new_goal_state) * self.gamma - self.potential_fcn(old_state, old_goal_state)

        elif self.reward_type in ["ng_phi0", "ng_0.5phi0"]:
            if done and self.termphi0: # set potential of new state to 0
                F = - self.potential_fcn(old_state)
            else:
                F = self.potential_fcn(new_state) * self.gamma - self.potential_fcn(old_state)
        elif self.reward_type == "sbs":
            old_state_sbs = normalize(np.array(old_state), self.env.size)[:-1] # clip time feature
            new_state_sbs = normalize(np.array(new_state), self.env.size)[:-1] # clip time feature
            if done and self.termphi0:
                F = - compute_sbs_phi(self.exp_kdtree, 
                                      query_state=old_state_sbs, 
                                      dist_scale=self.dist_scale)
            else:
                F = compute_sbs_phi(self.exp_kdtree, 
                                    query_state=new_state_sbs, 
                                    dist_scale=self.dist_scale) * self.gamma - compute_sbs_phi(self.exp_kdtree, 
                                                                                               query_state=old_state_sbs, 
                                                                                               dist_scale=self.dist_scale)
        elif self.reward_type == "manhattan":
            F = self.dist_rew(new_state, new_goal_state)

        shaped_reward = reward + self.rew_coef * F
        return shaped_reward


class TimeFeatureWrapper(Wrapper):
    """
    This has been adapted for the DiscreteGridworld setting (tabular learning)

    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps):
        super(TimeFeatureWrapper, self).__init__(env)
        assert isinstance(env.observation_space, spaces.Tuple)

        self._current_step = 0
        self._max_steps = max_steps

        # Add a time feature to the observation
        self.observation_space = spaces.Tuple([space for space in env.observation_space.spaces] + [spaces.Discrete(self._max_steps)])
        self.set_env_attrs()

    def set_env_attrs(self):
        '''set env attributes that must be accessed by the demowrapper'''
        self.generate_solution = self.env.generate_solution
        self.get_available_actions = self.env.get_available_actions
        self.size = self.env.size
        self.n_dims = self.env.n_dims

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._current_step += 1
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        """
        time_feature = self._current_step
        # print("TIME FEAT IS ", time_feature)
        return tuple([obs_i for obs_i in obs] + [time_feature])