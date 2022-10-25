import numpy as np 
import gym
from envs.time_feature_wrapper import TimeFeatureWrapper


class SparsifyWrapper(gym.Wrapper):
    """
    Accumulate environment reward and give it all every rew_delay timesteps
    """
    def __init__(self, env, rew_delay):
        super(SparsifyWrapper, self).__init__(env)
        assert rew_delay > 0
        self.rew_delay = rew_delay
        if isinstance(env, TimeFeatureWrapper):
            self._max_episode_steps = env._max_episode_steps

    def reset(self):
        self.ts = 0
        self.total_reward = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_reward += reward
        if self.ts % self.rew_delay == 0:
            sparse_reward = self.total_reward
            self.total_reward = 0
        else:
            sparse_reward = 0
        self.ts += 1
        return obs, sparse_reward, done, info