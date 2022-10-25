import random
import numpy as np
from typing import List, Union


class DiscreteBuffer():
    # based off of stable baselines 2.10.0 ReplayBuffer
    def __init__(self, buffer_size: int):
        self._storage = []
        self._maxsize = buffer_size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        return self._storage

    @property
    def buffer_size(self) -> int:
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """add a new transition to the buffer
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """add a new batch of transitions to the buffer
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def clear(self):
        self._storage = []
        self._next_idx = 0

    # override orig function
    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return (obses_t, actions, rewards, obses_tp1, dones)

    def sample(self, batch_size: int, **_kwargs):
        """Sample a batch of experiences.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class HERBufferWrapper(): # based heavily on sb2 HER implementation
    '''Based heavily on sb2 HER implementation
    Currently implements the random goal sampling strategy, consisting of 
    relabelling with achieved goals from the same episode.
    '''
    def __init__(self, replay_buffer, env, n_sampled_goal:int, goal_selection_strategy:str):
        self.replay_buffer = replay_buffer
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        assert self.goal_selection_strategy in ["random", "episode"]
        self.episode_transitions = []

    def __len__(self):
        return len(self.replay_buffer)
        
    def clear(self):
        self.replay_buffer.clear()
        self.episode_transitions = []

    def is_full(self):
        return self.replay_buffer.is_full()

    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size=batch_size)

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.episode_transitions.append((obs_t, action, reward, obs_tp1, done))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def get_goal_from_obs(self, obs):
        '''Modify per env name?'''
        # get achieved goal
        return np.array(obs[-2:])

    def _sample_achieved_goals(self):
        '''get achieved goals from list of episode transitions'''
        goals = []

        if self.goal_selection_strategy == "random":
            selected_idx = np.random.randint(len(self.replay_buffer) - 1, size=self.n_sampled_goal)
            selected_transitions = self.replay_buffer.storage[selected_idx]
            for trans in selected_transitions:
                obs_t, _, _, obs_tp1, _ = selected_transition 
                goals.append((self.get_goal_from_obs(obs_t), self.get_goal_from_obs(obs_tp1)))
        elif self.goal_selection_strategy == "episode":
            selected_idx = np.random.randint(len(self.episode_transitions) - 1, size=self.n_sampled_goal)
            for idx in selected_idx:
                obs_t, _, _, obs_tp1, _ = self.episode_transitions[idx] 
                goals.append((self.get_goal_from_obs(obs_t), self.get_goal_from_obs(obs_tp1)))

        return goals

    def _store_episode(self):
        for transition in self.episode_transitions:
            self.replay_buffer.add(*transition)
            sampled_goals = self._sample_achieved_goals()

            for goal_t, goal_tp1 in sampled_goals:
                # deepcopy not needed because obses are tuples which are immutables
                obs, action, reward, next_obs, done = transition
                # get desired goal
                obs, next_obs = list(obs), list(next_obs)
                obs[-2:] = goal_t
                next_obs[-2:] = goal_tp1
                done = False
                # recompute reward
                reward = self.env.compute_reward(reward, obs[:2], next_obs[:2], done, 
                    old_goal_state=obs[-2:], new_goal_state=next_obs[-2:])

                obs, next_obs = tuple(obs), tuple(next_obs)
                self.replay_buffer.add(obs, action, reward, next_obs, done)


            
