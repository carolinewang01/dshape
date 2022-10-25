import pprint
from gym.core import Env
from gym import spaces
import numpy as np


class GridWorld(Env):
    ## Initialise starting data
    def __init__(self, size=5, goal=None, reward_base_value=-1, max_steps=500):
        # Set information about the reward_gridworld
        self.size = size
        self.n_dims = 2 # 2d gridworld assumption is baked into code

        self.max_steps = max_steps
        self.current_step = 0
        
        # Set locations for the goal
        self.start_location = (size-1, 0)
        if goal is None:
            self.goal = (0, size-1)
        else:
            self.goal = goal
        self.terminal_states = [ self.goal]

        # Set current location
        self.reset()

        # Set reward_grid rewards for special cells
        # reward of self.reward_base_value for each timestep the reward isn't reached
        self.reward_base_value = reward_base_value
        self.reward_grid = np.zeros((self.size, self.size)) + self.reward_base_value
        self.reward_grid[ self.goal[0], self.goal[1]] += 1       

        # Set available actions
        self.action_dict = {'UP':0, 'DOWN':1, 'LEFT':2, 'RIGHT':3}
        self.actions = [0, 1, 2, 3]
        # Set probbility of a random action being taken
        self.random_action_prob = 0
    
    @property
    def observation_space(self):
        return spaces.Tuple([spaces.Discrete(self.size) for i in range(self.n_dims)])

    @property
    def action_space(self):
        return  spaces.Discrete(4)

    @property
    def reward_range(self):
        return (-1, 0)

    def reset(self):
        # Set random start location for the agent
        self.state = self.start_location # np.random.randint(0,5)
        self.current_step = 0
        return self.state
    
    def step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        info = {}
        last_location = self.state
        # TODO: WITH PROBABILITY 20%, CHANGE ACTION
        
        if np.random.uniform(0,1) < self.random_action_prob:
            available_actions = self.get_available_actions()
            action = available_actions[np.random.randint(0, len(available_actions))]

        # UP
        if action == self.action_dict['UP']:
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.compute_reward(last_location)
            else:
                self.state = ( self.state[0] - 1, self.state[1])
                reward = self.compute_reward(self.state)
        
        # DOWN
        elif action == self.action_dict['DOWN']:
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.size - 1:
                reward = self.compute_reward(last_location)
            else:
                self.state = ( self.state[0] + 1, self.state[1])
                reward = self.compute_reward(self.state)
                                                               
        # LEFT
        elif action == self.action_dict['LEFT']:
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                reward = self.compute_reward(last_location)
            else:
                self.state = ( self.state[0], self.state[1] - 1)
                reward = self.compute_reward(self.state)

        # RIGHT
        elif action == self.action_dict['RIGHT']:
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.size - 1:
                reward = self.compute_reward(last_location)
            else:
                self.state = ( self.state[0], self.state[1] + 1)
                reward = self.compute_reward(self.state)

        done = self.check_done()
        self.current_step += 1         
        return self.state, reward, done, info # TODNO: return next ob, reward, done, info

    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions

    def compute_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.reward_grid[ new_location[0], new_location[1]]
        
    def check_done(self):
        """Check if the agent is in a terminal state (gold or bomb), if so return 'TERMINAL'"""
        done = (self.current_step >= self.max_steps - 2)
        if self.state in self.terminal_states:
            done = True
        return done

    def generate_solution(self, style, goal=None, vis_solution=False):
        assert style in ["upper", "middle", "lower"]
        if goal is None:
            goal = self.goal
        
        soln_path = []
        state = self.start_location 
        soln_path.append(state) # (9, 0)
        
        if style == "middle":
            while state[0] != goal[0] and state[1] != goal[1]:
                state = (state[0] - 1, state[1])                
                soln_path.append(state)
                state = (state[0], state[1] +1)
                soln_path.append(state)
        elif style == "upper":
            while state[0] != goal[0]:
                state = (state[0] - 1, state[1])    
                soln_path.append(state)
            while state[1] != goal[1]:
                state = (state[0], state[1] + 1)    
                soln_path.append(state)
        elif style == "lower":
            while state[1] != goal[1]:
                state = (state[0], state[1] + 1)    
                soln_path.append(state)
            while state[0] != goal[0]:
                state = (state[0] - 1, state[1])    
                soln_path.append(state)        
        if vis_solution:
            np.set_printoptions(precision=3)
            pp = pprint.PrettyPrinter(width=150)
            pp.pprint(self.traj_on_map(soln_path))
        # compute reward of generated solution
        soln_rew = sum([self.reward_grid[state] for state in soln_path[1:]])

        return soln_path, soln_rew
        
    # TODO: PUT INTO RENDER
    def render(self):
        pass

    def agent_on_map(self):
        """Prints out current location of the agent on the reward_grid (used for debugging)"""
        reward_grid = np.zeros((self.size, self.size))
        reward_grid[ self.state[0], self.state[1]] = 1
        return reward_grid
    
    def traj_on_map(self, traj):
        """Prints out supplied states on the reward_grid (used for debugging)"""
        grid = np.zeros((self.size, self.size))
        for state in traj:
            grid[ state[0], state[1]] += 1
        total_states_visited = np.sum(grid)
        grid = grid / total_states_visited
        return grid
