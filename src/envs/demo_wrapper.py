import copy
import numpy as np 
import gym

from utils.helpers import get_raw_state_size 

class DemoWrapper(gym.Wrapper):
    def __init__(self, env, expert_demo, update_t=1, displace_t=0):
        '''
        update_t: if agent's state is s_t, append the expert state s^e_t', where t' 
                  is the lost multiple of update_t s.t. t' > t.
        displace_t: if agent's state is s_t, append expert state s^e_{t+displace_t}
        '''
        super(DemoWrapper, self).__init__(env)
        self.env = env
        
        self.expert_demo = expert_demo
        assert len(self.expert_demo.shape) == 2
        self.n_demo_timesteps = self.expert_demo.shape[0]
        self.horizon = self.env._max_episode_steps

        self.update_t = update_t 
        self.displace_t = displace_t

        # correct observation space size 
        self.expert_state_size = self.expert_demo.shape[1]
        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(self.obs_size + self.expert_state_size, ),
                                                dtype=np.float32)
        self.current_t = 0

    def get_expert_demo(self, ob):
        '''At time t, append expert's state at time t+displace_t'''
        # current_t = 50, n_demo_timesteps = 50 
        if (self.current_t + self.displace_t) > self.n_demo_timesteps -1 : 
            # append expert state 
            ts = self.n_demo_timesteps - 1
        else:
            ts = self.current_t + self.displace_t
        self.demo_ob = self.expert_demo[ts]

        self.current_t += 1        

    def reset(self):
        ob = self.env.reset()
        self.current_t = 0
        self.get_expert_demo(ob)
        return np.concatenate([ob, self.demo_ob]) # sets current_t = 1

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.get_expert_demo(ob)
        return np.concatenate([ob, self.demo_ob]), reward, done, info
