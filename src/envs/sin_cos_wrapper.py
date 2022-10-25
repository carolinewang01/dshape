import numpy as np 
import gym

from utils.helpers import get_raw_state_size 

class SinCosWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SinCosWrapper, self).__init__(env)

    def reset(self):
        raise NotImplementedError
        ob = self.env.reset()
        return ob

    def step(self, action):
        raise NotImplementedError
        ob, reward, done, info = self.env.step(action)
        return ob, reward, done, info