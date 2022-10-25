import os
import numpy as np
from tqdm import tqdm
from poc.buffer import DiscreteBuffer, HERBufferWrapper

class DiscreteAgent():
    # Intialise
    def __init__(self, env, epsilon=0.28, alpha=0.02, gamma=1, init_value=0.,
                 total_train_ts=250000, max_steps_per_episode=1000, eval_interval=2500,
                 reward_modified=False,
                 use_buffer=False, show_progress=True, 
                 save_policy=False, save_steps=[]):
        self.env = env
        self.reward_modified = reward_modified
        # want to create an n-dim array (maybe n+1 dim?) with range size
        # n=3: (size, size, size, 4)
        shape = [space.n for space in self.env.observation_space.spaces] + [4] # get q_table shape directly from the env
        self.q_table = np.ones(shape=shape) * init_value
        # random initialization
        # self.q_table = np.random.uniform(low=-0.1, high=0.1, size=shape)
        
        self.epsilon = epsilon # 0.05
        self.alpha = alpha # 0.1
        self.gamma = gamma
        self.init_value = init_value

        self.total_train_ts = total_train_ts
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_interval = eval_interval

        self.use_buffer = use_buffer
        self.show_progress = show_progress
        self.save_policy = save_policy
        self.save_steps = save_steps
    

    def choose_action(self, state, test_mode=False):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        available_actions = self.env.get_available_actions()
        if np.random.uniform(0,1) < self.epsilon and not test_mode:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[state]
            max_value = max(q_values_of_state)
            action = np.random.choice([available_actions[i] for i, value in enumerate(q_values_of_state) if value == max_value])
        return action

    # @abstractmethod
    def run_episode(self, test_mode):
        """Run a single episode. Must be implemented by inheriting classes"""
        raise NotImplementedError
    
    def train(self, save_dir=None):
        """The train function runs iterations and updates Q-values if desired."""
        env_reward_per_episode = [] # Initialise performance log
        shaped_reward_per_episode = []
        eval_ts = []

        total_numsteps = 0
        n_evals = -1 # eval at total_numsteps = 0 as well
        pbar = tqdm(total=self.total_train_ts, disable=not self.show_progress)
        # for i in tqdm(range(self.total_train_ts), disable=not self.show_progress): # Run trials
        while total_numsteps < self.total_train_ts:
            _, _, ep_len, _ = self.run_episode(test_mode=False, add_to_buffer=self.use_buffer)
            total_numsteps += ep_len

            if (total_numsteps // self.eval_interval) > n_evals:
                cum_env_rew, cum_shaped_rew, _, _ = self.run_episode(test_mode=True)
                env_reward_per_episode.append(cum_env_rew)
                shaped_reward_per_episode.append(cum_shaped_rew)
                eval_ts.append(total_numsteps)
                n_evals = total_numsteps // self.eval_interval 

            if self.save_policy and (n_evals in self.save_steps):
            # if self.save_policy and (i in self.save_steps) and i != 0:
                self.save(save_dir, n_evals)

            pbar.update(ep_len)

        pbar.close()

        if self.save_policy:
            self.save(save_dir, "last")

        return env_reward_per_episode, shaped_reward_per_episode, eval_ts # Return performance log

    def save(self, save_dir, label):
        """Save q table"""
        save_path = os.path.join(save_dir, f"q-table_eval={label}.npz")
        np.savez_compressed(save_path, q_table=self.q_table)

class Q_Agent(DiscreteAgent):
    def __init__(self, env, epsilon=0.05, alpha=0.1, gamma=1,  # 0.1, 0.02
                 init_value=0.0, 
                 total_train_ts=250000, max_steps_per_episode=1000, 
                 eval_interval=2500,
                 use_buffer=True, buffer_size=None,
                 relabel=False, n_sampled_goal=2,
                 reward_modified=False,
                 updates_per_step=1, show_progress=True, 
                 save_policy=False, save_steps=[],
                 eval_only=False):
        super(Q_Agent, self).__init__(env=env, 
                                      epsilon=epsilon, alpha=alpha, gamma=gamma, init_value=init_value,
                                      total_train_ts=total_train_ts, max_steps_per_episode=max_steps_per_episode, 
                                      eval_interval=eval_interval, reward_modified=reward_modified, use_buffer=use_buffer, 
                                      show_progress=show_progress, save_policy=save_policy, save_steps=save_steps
                                      )
        self.updates_per_step = updates_per_step
        self.buffer_size = buffer_size
        self.relabel = relabel
        self.n_sampled_goal = n_sampled_goal
        if not eval_only:
            self.reset_agent() # init q-table and fill buffer

    def load_agent(self, load_path):
        self.q_table = np.load(load_path)["q_table"]

    def reset_agent(self):
        self.q_table = np.ones_like(self.q_table) * self.init_value
        if self.use_buffer:
            self.buffer = DiscreteBuffer(self.buffer_size)
            if self.relabel:
                self.buffer = HERBufferWrapper(self.buffer, self.env, n_sampled_goal=self.n_sampled_goal, goal_selection_strategy="episode")
            
            while not self.buffer.is_full():
                self.run_episode(test_mode=True, add_to_buffer=True)
        
    def update(self, old_state, reward, new_state, action, done):
        """Updates the Q-value table"""
        current_q_value = self.q_table[old_state][action]

        if done: # implying new_state is terminal
            max_q_value_in_new_state = 0
        else:
            q_values_of_state = self.q_table[new_state]
            max_q_value_in_new_state = max(q_values_of_state)
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)

    def run_episode(self, test_mode, add_to_buffer=False):
        states = []
        cumulative_env_reward, cumulative_shaped_reward = 0, 0 # Initialise values of each game
        step = 0
        done = False
        old_state = self.env.reset()
        states.append(old_state)

        while step < self.max_steps_per_episode - 1 and done != True: # Run until max steps or until game is finished
            action = self.choose_action(state=old_state, test_mode=test_mode)
            new_state, reward, done, info = self.env.step(action)
            # If game is in terminal state, game over and start next trial
            if add_to_buffer:
                self.buffer.add(obs_t=old_state, action=action, reward=reward, obs_tp1=new_state, done=done)

            if not test_mode: # perform update
                if self.use_buffer:
                    s_t, a, r, s_tp1, d = self.buffer.sample(batch_size=self.updates_per_step)
                    for i in range(self.updates_per_step):
                        self.update(old_state=s_t[i], reward=r[i], new_state=s_tp1[i], action=a[i], done=d[i])
                else:
                    self.update(old_state=old_state, reward=reward, new_state=new_state, action=action, done=done)

            # evaluate w.r.t. environment reward if needed
            if self.reward_modified:
                cumulative_env_reward += info["env_rew"]
                cumulative_shaped_reward += reward
            else:             
                cumulative_env_reward += reward
                cumulative_shaped_reward += reward

            old_state = new_state
            states.append(old_state)
            step += 1
        return cumulative_env_reward, cumulative_shaped_reward, step, states

############### SARSA #####################
class SarsaAgent(DiscreteAgent):
    def __init__(self, env, epsilon=0.1, alpha=0.02, gamma=1,
                total_train_ts=500, max_steps_per_episode=1000, eval_interval=5):
        super(SarsaAgent, self).__init__(env=env, epsilon=epsilon, alpha=alpha, gamma=gamma,
                                         total_train_ts=total_train_ts, max_steps_per_episode=max_steps_per_episode,
                                         eval_interval=eval_interval, use_buffer=False
            )

    def update(self, old_state, reward, new_state, old_action, new_action, done):
        """Updates the Q-value table using SARSA"""
        current_q_value = self.q_table[old_state][old_action]
        q_value_next_state_action = self.q_table[new_state][new_action]        
        reward, shaped_rew = self.pbrs(reward, old_state, new_state, old_ts, done) # shaped rew = reward when shaping is off
        self.q_table[old_state][old_action] = (1 - self.alpha) * current_q_value + self.alpha * (shaped_rew + self.gamma * q_value_next_state_action)

    def run_episode(self, test_mode):
        states = []
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        done = False
        old_state = self.env.reset()
        old_action = self.choose_action(state=old_state, test_mode=test_mode) 

        while step < self.max_steps_per_episode and done is False: # Run until max steps or until game is finished
            states.append(old_state)
            new_state, reward, done, _  = self.env.step(old_action)
            new_action = self.choose_action(state=new_state, test_mode=test_mode)
            # If game is in terminal state, game over and start next trial
            if not test_mode:
                self.update(old_state, reward, new_state, old_action, new_action, done=done)  
            
            cumulative_reward += reward
            step += 1
            
            old_state = new_state
            old_action = new_action
        return cumulative_reward, step, states
