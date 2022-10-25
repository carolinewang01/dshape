import numpy as np
from scipy.spatial import cKDTree as KDTree
from gym import Wrapper

from envs.goal_demo_wrapper import goal_distance
from utils.helpers import extract_observable_state
from utils.load_confs import load_reward_params, load_parameters

params = load_parameters()


def logistic(x, tau):
    return 1 / (1 + np.exp(tau * -x))


def compute_sbs_phi(exp_kdtree, query_state, dist_scale):
    # k for k-nearest neighbors, p for p-norm
    dist, idx = exp_kdtree.query(query_state, k=1, p=2, n_jobs=1) # find expert state closest to query state
    similarity = np.exp(- 0.5 * (dist**2) * dist_scale)
    return similarity


class RewardWrapper(Wrapper):
    def __init__(self, env, env_id, expert_demo, reward_type, displace_t=1, 
                 params_filename=None, reward_params=None, raw=False, time_feat=True):
        '''
        displace_t: if agent's state is s_t, use expert state s^e_{t+displace_t}
        reward_type: choices are ['huber', 'kimura', 'env']
        Can choose to pass in params_filename OR reward_params
        '''
        super(RewardWrapper, self).__init__(env)
        self.env = env
        self.env_id = env_id
        self.expert_demo = expert_demo
        assert len(self.expert_demo.shape) ==  2
        self.n_demo_timesteps = self.expert_demo.shape[0]

        self.displace_t = displace_t

        self.reward_type = reward_type
        self.time_feat = time_feat
        self.raw = raw
        if raw: 
            obs_type = "raw"
        else: 
            obs_type = "full"

        # if reward params specified either by passing in or through a params yaml, use those
        # otherwise use ones specified in the paramemters.yml
        self.reward_params = params['rew_shaping'][env_id.strip("-v2").lower()][obs_type]
        self.dist_std = self.reward_params["dist_std"] # no matter what, retrieve the dist_std from parameters.yml
        if params_filename: 
            self.reward_params = load_reward_params(params_filename)
        elif reward_params: 
            self.reward_params = reward_params
        
        if self.reward_type in ['huber', 'huber2']:
            self.imit_rew_coef = self.reward_params['imit_rew_coef']
            self.alpha = self.reward_params["alpha"]
            self.gamma = self.reward_params["gamma"]
        elif self.reward_type == 'potential':
            self.discount = params['sac'][env_id.strip("-v2").lower()]["gamma"]
            self.terminal_phi0 = params['pbrs']['terminal_phi0']
        elif self.reward_type == 'sbs_potential':
            assert not self.raw, "SBS should be run with full states."
            self.discount = params['sac'][env_id.strip("-v2").lower()]["gamma"]
            self.terminal_phi0 = params['pbrs']['terminal_phi0']
            self.dist_scale = self.reward_params["dist_scale"]
            self.tau = self.reward_params["tau"]
            self.exp_kdtree = KDTree(logistic(self.expert_demo, self.tau))

        self.current_t = 0
    
    def get_expert_demo(self, ob):
        '''At time t, append expert's state at time t+displace_t'''
        # current_t = 50, n_demo_timesteps = 50 
        if (self.current_t + self.displace_t) > self.n_demo_timesteps -1 : 
            # append expert state 
            self.demo_ob = self.expert_demo[self.n_demo_timesteps - 1]
        else:
            self.demo_ob = self.expert_demo[self.current_t + self.displace_t]
        self.current_t += 1  

    def reset(self):
        ob = self.env.reset()
        self.prev_ob = ob
        self.current_t = 0
        self.get_expert_demo(ob) # sets current_t = 1
        self.prev_demo_ob = self.demo_ob
        return ob

    def step(self, action):
        ob, reward_env, done, reward_info = self.env.step(action)
        self.get_expert_demo(ob)

        if self.reward_type in  ["huber", "huber2"]:
            if self.raw:
                raw_ob = extract_observable_state(ob, self.env_id)
                d = goal_distance(raw_ob, self.demo_ob, dim_scale=False if self.reward_type=="huber" else True)
            else:
                d = goal_distance(ob, self.demo_ob, dim_scale=False if self.reward_type=="huber" else True)

            sq_dist = np.square(d)
            if self.reward_type == "huber2":
                sq_dist = sq_dist / self.dist_std
            imit_reward = - (self.alpha * sq_dist) - ((1 - self.alpha) * np.sqrt(self.gamma + sq_dist))
            reward = ((1 - self.imit_rew_coef) * reward_env) + (self.imit_rew_coef * imit_reward)

        elif self.reward_type == "potential":
            demo_ob_t = self.prev_demo_ob
            demo_ob_tp1 = self.demo_ob
            
            if self.raw:
                ob_t = extract_observable_state(self.prev_ob, self.env_id)
                ob_tp1 = extract_observable_state(ob, self.env_id)
            else:
                ob_t, ob_tp1 = self.prev_ob, ob

            if done and self.terminal_phi0: # needed for policy invariance in finite horizon domains
                d_tp1 = 0
            else:
                d_tp1 = goal_distance(ob_tp1, demo_ob_tp1, dim_scale=True)
            d_t = goal_distance(ob_t, demo_ob_t, dim_scale=True)
            potential = - self.discount * d_tp1 + d_t
            reward = reward_env + potential

        elif self.reward_type == "sbs_potential":
            if self.time_feat:
                ob_t, ob_tp1 = logistic(self.prev_ob[:-1], self.tau), logistic(ob[:-1], self.tau) # slice to remove time feat
            else: 
                ob_t, ob_tp1 = logistic(self.prev_ob, self.tau), logistic(ob, self.tau)

            if done and self.terminal_phi0: # needed for policy invariance in finite horizon domains
                phi_tp1 = 0
            else:
                phi_tp1 = compute_sbs_phi(self.exp_kdtree, query_state=ob_tp1, dist_scale=self.dist_scale)
            phi_t = compute_sbs_phi(self.exp_kdtree, query_state=ob_t, dist_scale=self.dist_scale)
            # no negative sign necessary on phi_t because phi_t is a similarity score
            potential = self.discount * phi_tp1 - phi_t
            reward = reward_env + potential

        elif self.reward_type == "env":
            reward = reward_env

        reward_info['reward_shaped'] = reward
        reward_info['reward_env'] = reward_env

        self.prev_ob = ob
        self.prev_demo_ob = self.demo_ob
        return ob, reward, done, reward_info
