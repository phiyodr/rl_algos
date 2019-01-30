import gym
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from datetime import datetime
import time


class QLearning():
    """Q-learning algorithms with epsilon greedy policy.
    
    References:
        https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf
    """
    def __init__(self, env, epsilon = .9, alpha = .1, gamma = .9):
        self.env = env
        self.Q = np.zeros(shape=(self.env.observation_space.n, 
                                 self.env.action_space.n))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def epsilon_greedy(self, S):
        """Epsilon greedy policy
    
        Args:
            S (int): Current state
        Returns:
            action (int): Action to take.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[S, :])
        return action

    def train(self, num_episodes = 10000, verbose = True):
        """Train
        
        Runs `num_episodes` episodes to train the Q table.
    
        Args:
            num_episodes (int): Number of episodes
            verbose (bool): Verbosity
            
        Returns:
            Q (np.array): Q table
        """
        start_time = datetime.now().replace(microsecond=0)
        for e in range(num_episodes):
            S_old = self.env.reset()
            steps = 0
            # there is an interal limit of 100 steps
            while steps < 1000:
                steps += 1
                A = self.epsilon_greedy(S_old)
                S_new, reward, done, info = self.env.step(A)
                self.Q[S_old, A] = self.Q[S_old, A] + self.alpha * \
                    (reward + self.gamma * np.max(self.Q[S_new, :]) - self.Q[S_old, A])
                if done:
                    break
                S_old = S_new
            if verbose:
                clear_output(wait=True)
                now_time = datetime.now().replace(microsecond=0)
                print("Epoch: {}/{} - Steps: {:4} - Duration: {}".format(e+1, num_episodes, steps, now_time-start_time))

        return self.Q

    def run(self, verbose=False, sleep_time=.1):
        """ Performs one single run using Q-table (without epsilon greedy policy)
        
        Args:
            verbose (bool): Verbosity
            
        Returns:
            reward (float): Reward in last step
            steps (int): Number of steps until last state
        """
        S_old = self.env.reset()
        max_steps = 100
        for step in range(max_steps):
            if verbose:
                time.sleep(sleep_time)
                clear_output(wait=True)
                print("Step: {}".format(step))
                self.env.render()
            A = np.argmax(self.Q[S_old, :])
            S_new, reward, done, info = self.env.step(A)
            if verbose:
                #print(self.Q[S_old, :])
                #print("action={}".format(A))
                print(S_old, S_new, reward, done)
            if done:
                if verbose:
                    time.sleep(sleep_time)
                    clear_output(wait=True)
                    print("Final Steps: {}".format(step))
                    self.env.render()
                break
            S_old = S_new
        return reward, step + 1 
    
    def save(self,path="qtable.npy"):
        """Save Q-table (parameters will not be saved)"""
        np.save(path, self.Q)
        
    def load(self, path="qtable.npy"):
        """Load Q table (no parameters)"""
        self.Q = np.load(path)