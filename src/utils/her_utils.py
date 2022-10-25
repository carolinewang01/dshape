from enum import Enum
from collections import OrderedDict
import copy
from typing import Optional, List, Union
import random

import numpy as np
from sklearn.metrics import pairwise_distances
from gym import spaces 

from utils.epsilon_schedules import DecayThenFlatSchedule
from utils.helpers import extract_observable_state
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.her.utils import HERGoalEnvWrapper


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3
    # Selects goal as next achieved state
    NEXT_STATE = 4
    # Select goal as next achieved state as goal, plus n-1 randomly sampled states 
    NEXT_STATE_AND_EP = 5
    # Select goals that were achieved in the episode and lie closest to the exp traj
    EPISODE_NEAREST = 6
    # Select goals that were achieved in the episode and lie closest to future expert trajectories
    EPISODE_NEAREST_FUTURE = 7
    # Select goals randomly from the demonstration states
    DEMO_RANDOM = 8


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM,
    'next_state': GoalSelectionStrategy.NEXT_STATE,
    'next_state_and_ep': GoalSelectionStrategy.NEXT_STATE_AND_EP,
    'episode_nearest': GoalSelectionStrategy.EPISODE_NEAREST,
    'episode_nearest_future': GoalSelectionStrategy.EPISODE_NEAREST_FUTURE,
    'demo_random': GoalSelectionStrategy.DEMO_RANDOM,
}

MOD_KEY_ORDER = ['observation', 'desired_goal']

class CustomHERGoalEnvWrapper(HERGoalEnvWrapper):
    """Delete achieved goal from observation space 
    """
    def __init__(self, env):
        super(CustomHERGoalEnvWrapper, self).__init__(env)
                # self.env = env
        # self.metadata = self.env.metadata
        # self.action_space = env.action_space
        self.rb_spaces = list(env.replay_buffer_obs_space.spaces.values())
        # Check that all spaces are of the same type
        # (current limitation of the wrapper)
        rb_space_types = [type(env.replay_buffer_obs_space.spaces[key]) for key in MOD_KEY_ORDER]
        assert len(set(rb_space_types)) == 1, "The spaces for goal and observation"\
                                           " must be of the same type"

        if isinstance(self.rb_spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.replay_buffer_obs_space.spaces['desired_goal'].shape
            self.obs_dim = env.replay_buffer_obs_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert goal_space_shape[1] == 1, "Only 1D observation spaces are supported yet"
            else:
                assert len(goal_space_shape) == 1, "Only 1D observation spaces are supported yet"

        if isinstance(self.rb_spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.replay_buffer_obs_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.rb_spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.rb_spaces])
            highs = np.concatenate([space.high for space in self.rb_spaces])
            self.replay_buffer_obs_space = spaces.Box(lows, highs, dtype=np.float32)
            # print("DEFINED REPLAY BUFFER OBS  AS ", self.replay_buffer_obs_space)

        elif isinstance(self.rb_spaces[0], spaces.Discrete):
            dimensions = [env.replay_buffer_obs_space.spaces[key].n for key in MOD_KEY_ORDER]
            self.replay_buffer_obs_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError("{} space is not supported".format(type(self.rb_spaces[0])))


    def convert_dict_to_obs(self, obs_dict, preserve_achieved_goal=True):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if preserve_achieved_goal: 
            KEY_ORDER = ["observation", "achieved_goal", "desired_goal"]
        else: 
            KEY_ORDER = MOD_KEY_ORDER
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        return np.concatenate([obs_dict[key] for key in KEY_ORDER]) 

    def convert_obs_to_dict(self, observations, achieved_goal=True):
        """
        Inverse operation of convert_dict_to_obs
        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        if achieved_goal: 
            assert observations.shape[0] > self.obs_dim + self.goal_dim, f"Obs shape is {observations.shape[0]}, not large enough"

            return OrderedDict([
                ('observation', observations[:self.obs_dim]),
                ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
                ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
            ])
        else: 
            return OrderedDict([
                ('observation', observations[:self.obs_dim]),
                ('desired_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim])
            ])

    def step(self, action, eval_mode=False):
        obs, reward, done, info = self.env.step(action, eval_mode)
        return self.convert_dict_to_obs(obs), reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info, eval_mode=False):
        return self.env.compute_reward(achieved_goal, desired_goal, info, eval_mode=eval_mode)



class CustomReplayBuffer(ReplayBuffer):
    """Add info to replay buffer
    """
    def __init__(self, size: int):
        super(CustomReplayBuffer, self).__init__(size=size)

    def add(self, obs_t, action, reward, info, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, info, obs_tp1, done)
        if isinstance(self._next_idx, float):
            self._next_idx = int(self._next_idx)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, info, obs_tp1, done):
        """
        add a new batch of transitions to the buffer
        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch
        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, info, obs_tp1, done):
            if isinstance(self._next_idx, float):
                self._next_idx = int(self._next_idx)
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], env: Optional[VecNormalize] = None):
        obses_t, actions, rewards, infos, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, info, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            infos.append(info)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (self._normalize_obs(np.array(obses_t), env),
                np.array(actions),
                self._normalize_reward(np.array(rewards), env),
                infos,
                self._normalize_obs(np.array(obses_tp1), env),
                np.array(dones))


class HindsightExperienceReplayWrapper(object):
    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, 
                 wrapped_env, expert_demo=None, time_feat=False, sin_cos_repr=False,
                 raw=False, env_id=None,
                 total_timesteps=None):
        """Copy of original HER Hindsight Experience Replay Wrapper
        Modifications:
        - Modified GoalSelectionStrategy, so the assertion in the init now refers to another object
        - Added support for new goal sampling methods 
        - Add time feat 
        - Deepcopy modified transitions 
        - Save the "info" returned by env.step() in the buffer
        Wrapper around a replay buffer in order to use HER.
        This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.
        :param replay_buffer: (ReplayBuffer)
        :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
        :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
            the goals for the artificial transitions.
        :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
            that enables to convert observation to dict, and vice versa
        """
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        self.epsilon_schedule = DecayThenFlatSchedule(start=0.75, finish=0.0, time_length=total_timesteps)

        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer
        self.time_feat = time_feat
        self.sin_cos_repr = sin_cos_repr
        self.raw = raw
        self.env_id = env_id
        self.expert_demo = expert_demo
    
    def add(self, obs_t, action, reward, info, obs_tp1, done, t_env):
        """
        add a new transition to the buffer
        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param info: (dict) the info dict of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append((obs_t, action, reward, info, obs_tp1, done))
        self.epsilon = self.epsilon_schedule.eval(t_env) # used for episode_nearest goal sampling strategy
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return self.replay_buffer.can_sample(n_samples)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_achieved_goal(self, episode_transitions, transition_idx, goal_selection_strategy):
        """
        Modified: double goal sampling instead of single, the next_state goal sampling strategy 
        Sample an achieved goal according to the sampling strategy.
        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        # implement double goal sampling
        if goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx] 
            obs_t, _, _, _, obs_tp1, _ = selected_transition
            achieved_goal_t = self.env.convert_obs_to_dict(copy.deepcopy(obs_t))['achieved_goal']
            achieved_goal_tp1 = self.env.convert_obs_to_dict(copy.deepcopy(obs_tp1))['achieved_goal']
        elif goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
            obs_t, _, _, _, obs_tp1, _ = selected_transition
            achieved_goal_t = self.env.convert_obs_to_dict(copy.deepcopy(obs_t))['achieved_goal']
            achieved_goal_tp1 = self.env.convert_obs_to_dict(copy.deepcopy(obs_tp1))['achieved_goal']
        elif goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
            obs_t, _, _, _, obs_tp1, _ = selected_transition
            achieved_goal_t = self.env.convert_obs_to_dict(copy.deepcopy(obs_t))['achieved_goal']
            achieved_goal_tp1 = self.env.convert_obs_to_dict(copy.deepcopy(obs_tp1))['achieved_goal']
        elif goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = random.randint(0, len(self.replay_buffer) - 1)
            selected_transition = self.replay_buffer.storage[selected_idx]
            obs_t, _, _, _, obs_tp1, _ = selected_transition
            # deleted achieved goals from replay buffer transitions
            achieved_goal_t = self.env.convert_obs_to_dict(copy.deepcopy(obs_t),
                                                           achieved_goal=False)['observation']
            achieved_goal_tp1 = self.env.convert_obs_to_dict(copy.deepcopy(obs_tp1),
                                                             achieved_goal=False)['observation']
            # convert obs to raw if necessary
            achieved_goal_t = extract_observable_state(achieved_goal_t, 
                                                       self.env_id, 
                                                       self.time_feat, sin_cos_repr=self.sin_cos_repr) if self.raw else achieved_goal_t
            achieved_goal_tp1 = extract_observable_state(achieved_goal_tp1, 
                                                         self.env_id, 
                                                         self.time_feat, sin_cos_repr=self.sin_cos_repr) if self.raw else achieved_goal_tp1
        elif goal_selection_strategy == GoalSelectionStrategy.NEXT_STATE:
            # Choose goal achieved at the next state
            std = 1e-4
            n_transitions = len(episode_transitions)
            selected_idx = (transition_idx + 1) if transition_idx < (n_transitions - 1) else -1
            selected_transition = episode_transitions[selected_idx]
            obs_t, _, _, _, obs_tp1, _ = selected_transition
            achieved_goal_t = self.env.convert_obs_to_dict(copy.deepcopy(obs_t))['achieved_goal']
            achieved_goal_tp1 = self.env.convert_obs_to_dict(copy.deepcopy(obs_tp1))['achieved_goal']
            # achieved_goal += np.random.normal(0, std, size=achieved_goal.shape[0])
        elif goal_selection_strategy == GoalSelectionStrategy.EPISODE_NEAREST or goal_selection_strategy == GoalSelectionStrategy.EPISODE_NEAREST_FUTURE:
            # with probability epsilon, samplee random state from episode:
            # if np.random.binomial(n=1, p=self.epsilon, size=1)[0]: # w.p. epsilon, select 1
            #     selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            # else:   

            # select states leading up to the idx that's in ep nearest
            # selected_idx = transition_idx - np.random.choice(np.arange(min(transition_idx, 20))) if transition_idx != 0 else 0
            selected_idx = transition_idx
            selected_transition = episode_transitions[selected_idx]
            obs_t, _, _, _, obs_tp1, _ = selected_transition
            achieved_goal_t = self.env.convert_obs_to_dict(copy.deepcopy(obs_t))['achieved_goal']
            achieved_goal_tp1 = self.env.convert_obs_to_dict(copy.deepcopy(obs_tp1))['achieved_goal']

        elif goal_selection_strategy == GoalSelectionStrategy.DEMO_RANDOM:
            # relabel at random from expert demonstration
            selected_idx = np.random.choice(np.arange(0, self.expert_demo.shape[0] - 1)) 
            achieved_goal_t = self.expert_demo[selected_idx]
            achieved_goal_tp1 = self.expert_demo[selected_idx + 1]

        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return (achieved_goal_t, achieved_goal_tp1)

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.
        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the current transition being relabelled
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.NEXT_STATE_AND_EP and self.n_sampled_goal > 0:
            next_state_trans = self._sample_achieved_goal(episode_transitions, transition_idx, GoalSelectionStrategy.NEXT_STATE)
            ep_trans = [
                   self._sample_achieved_goal(episode_transitions, transition_idx, GoalSelectionStrategy.EPISODE)
                    for _ in range(self.n_sampled_goal - 1)
                        ]
            return [next_state_trans] + ep_trans
        elif (self.goal_selection_strategy == GoalSelectionStrategy.EPISODE_NEAREST) or (self.goal_selection_strategy == GoalSelectionStrategy.EPISODE_NEAREST_FUTURE):
            if transition_idx == 0:
                self.compute_pairwise_distances()
            self.nearest_trans_idxs = self.compute_nearest_ep_states(transition_idx=0 if (self.goal_selection_strategy == GoalSelectionStrategy.EPISODE_NEAREST) else transition_idx)
            return [self._sample_achieved_goal(episode_transitions, idx, self.goal_selection_strategy)
                for idx in self.nearest_trans_idxs
            ]
        else: 
            return [
                self._sample_achieved_goal(episode_transitions, transition_idx, self.goal_selection_strategy)
                for _ in range(self.n_sampled_goal)
            ]

    def compute_pairwise_distances(self):
        """
        Computes pairwise distances between all states in self.episode_transitions and self.expert_demo
        """
        agent_states = []
        for transition in self.episode_transitions:
            obs_t, _, _, _, obs_tp1, _ = transition
            state_tp1 = self.env.convert_obs_to_dict(obs_tp1)['achieved_goal']
            agent_states.append(state_tp1)
        agent_states = np.array(agent_states)
        self.pairwise_dists = pairwise_distances(agent_states, self.expert_demo)

    def compute_nearest_ep_states(self, transition_idx):
        """Returns indices of achieved states that are nearest to future expert states 
        (i.e., expert states after transition_idx).
        transition_idx: index of current transition to relabel
        """
        # for each agent state, compute min distance to expert states AFTER transition_idx
        # assume agent dim=0, exp dim=1
        min_dists = np.amin(self.pairwise_dists[:, transition_idx:], axis=1)
        min_idx = np.argpartition(min_dists, self.n_sampled_goal) 
        return min_idx[:self.n_sampled_goal] 

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions

        for transition_idx, transition in enumerate(self.episode_transitions):
            # print("RELABELLING TRANSITION")

            # Copy transition to avoid modifying the original one
            obs_t, action, reward, info, obs_tp1, done = transition
            # sac records one more state than we do. want the same number of states as in exp demo
            # if done: continue 

            obs_t_dict = self.env.convert_obs_to_dict(obs_t)
            obs_tp1_dict = self.env.convert_obs_to_dict(obs_tp1)

            # Delete achieved goal 
            del obs_t_dict["achieved_goal"]
            del obs_tp1_dict["achieved_goal"]

            ## Correct obs_tp1 so that it shares the goal of obs_t 
            # obs_tp1_dict["desired_goal"] = copy.deepcopy(obs_t_dict["desired_goal"])

            # Transform back to ndarrays
            obs_t_mod = self.env.convert_dict_to_obs(obs_t_dict, preserve_achieved_goal=False)
            obs_tp1_mod = self.env.convert_dict_to_obs(obs_tp1_dict, preserve_achieved_goal=False)

            # Add to the replay buffer
            self.replay_buffer.add(obs_t_mod, action, reward, info, obs_tp1_mod, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for (goal_t, goal_tp1) in sampled_goals:
                # print("SAMPLED GOALS ")
                # print("GOAL T: ", goal_t)
                # print("GOAL TP1: ", goal_tp1)
                # Copy transition to avoid modifying the original one
                obs, action, reward, info, next_obs, done = copy.deepcopy(transition)

                # print("OBS T: ", obs)
                # print("OBS TP1: ", next_obs)
                # Convert concatenated obs to dict, so we can update the goals
                obs_dict = self.env.convert_obs_to_dict(obs)
                next_obs_dict = self.env.convert_obs_to_dict(next_obs)

                # print("OBS T DICT: ", obs_dict)
                # print("OBS TP1 DICT: ", next_obs_dict)

                # goal_tp1 = copy.deepcopy(goal_t) 
                # Force timesteps of desired goal to match timesteps of desired goal at time t
                # if self.time_feat:
                    # use desired goal to carry over correct ts from expert goal (displace t)
                    # goal_t[-1:] = obs_dict["desired_goal"][-1:]
                    # goal_tp1[-1:] = next_obs_dict["desired_goal"][-1:]

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal_t
                info["prev_ob"]["desired_goal"] = goal_t
                next_obs_dict['desired_goal'] = goal_tp1

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(achieved_goal=next_obs_dict['achieved_goal'], 
                                                 desired_goal=next_obs_dict['desired_goal'], # TODO: goal_t or goal_tp1? 
                                                 info=info)

                done = False # why is done set to False?

                # print("RELABELLED OBS T", obs_dict)
                # print("RELABELLED OBS TP1", next_obs_dict)

                # delete achieved goal 
                del obs_dict["achieved_goal"]
                del next_obs_dict["achieved_goal"]

                # Transform back to ndarrays
                obs = self.env.convert_dict_to_obs(obs_dict, preserve_achieved_goal=False)
                next_obs = self.env.convert_dict_to_obs(next_obs_dict, preserve_achieved_goal=False)

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, info, next_obs, done)

if __name__ == '__main__':
    main()

