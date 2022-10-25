from gym.core import Env
from gym.spaces import Box
import numpy as np


class PointWorld(Env):
    def __init__(self, size=1, goal=None, early_term=True, sparse_rew=True, rew_base_value=-1):  # Can set goal to test adaptation.
        self.size = size # 2d boxworld
        self._goal = goal
        self.early_term = early_term
        self.sparse_rew = sparse_rew
        self.rew_base_value = rew_base_value
        self.done_box_size = 0.025 # corresponding to 20x20 gridworld goal size

    @property
    def observation_space(self):
        return Box(low=-self.size/2, high=self.size/2, shape=(2,)) # origin centered

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def sample_goals(self, num_goals):
        return np.random.uniform(-self.size/2, self.size/2, size=(num_goals, 2, ))

    def reset(self, reset_args=None):
        self.time_step = 0
        goal = reset_args
        if goal is not None:
            self._goal = goal
        elif self._goal is None:
            # Only set a new goal if this env hasn't had one defined before.
            self._goal = np.squeeze(self.sample_goals(num_goals=1))
            #goals = [np.array([-0.5,0]), np.array([0.5,0])]
            #goals = np.array([[-0.5,0], [0.5,0],[0.2,0.2],[-0.2,-0.2],[0.5,0.5],[0,0.5],[0,-0.5],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5]])
            #self._goal = goals[np.random.randint(10)]

        self._state = (0., 0.)
        observation = np.copy(self._state)
        return observation

    def compute_rew(self, x_diff, y_diff):
        at_goal = (abs(x_diff) < self.done_box_size and abs(y_diff) < self.done_box_size)

        if self.sparse_rew:
            if at_goal:
                reward = self.rew_base_value + 1
            else: 
                reward = self.rew_base_value
        else: # euclidean distance reward
            reward = - (x_diff ** 2 + x_diff ** 2) ** 0.5
        return reward


    def step(self, action):
        self._state = np.clip(self._state + action, 
                              a_min=self.observation_space.low[0], 
                              a_max=self.observation_space.high[0])
        self.time_step += 1

        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]

        at_goal = (abs(x) < self.done_box_size and abs(y) < self.done_box_size)
        reward = self.compute_rew(x, y)

        if self.early_term:
            done = at_goal
        else:
            done = False
        # done = False
        next_observation = np.copy(self._state)
        return next_observation, reward, done, {"goal": self._goal}

    def generate_solution(self, goal=None, 
        # vis_solution
        ):
        if goal is None:
            goal = self._goal

        state = np.array([0., 0.])
        traj = [state]
        # compute opt act size 
        opt_act = goal - state 
        c = np.amax(np.abs(opt_act)) # maximum element of the absolute action
        lam = 1 / self.action_space.high[0] 
        scale_factor = c * lam # scale vector such that it should fit within the action box
        opt_act /= scale_factor # largest feasible action with optimal direction
        assert self.action_space.contains(opt_act), f"opt_act is {opt_act}"

        ep_return = 0 
        done = False
        while not done:
            if np.amax(np.abs(goal - state)) < self.action_space.high[0]:
                # print("ADDING LAST ACT")
                # distance between goal and state is small enough   
                assert self.action_space.contains(goal - state)
                new_state = state + (goal - state)
            else:
                # print("ADDING SMALL ACTION")
                new_state = state + opt_act # add scaled action instead

            ep_return += self.compute_rew(new_state[0] - goal[0], new_state[1] - goal[1])
            done = abs(new_state[0] - goal[0]) < self.done_box_size and abs(new_state[1] - goal[1]) < self.done_box_size
            traj.append(new_state)
            state = new_state

        return traj, ep_return

    def render(self):
        print('current state:', self._state)


if __name__ == '__main__':
    goal = np.array([-0.33502217,  0.03985494])
    env =  PointWorld(goal=goal)
    soln = env.generate_solution(goal=goal)
    print("SOLN IS ", soln)