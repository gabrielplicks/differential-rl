import numpy as np
import gym
import gym_accesscontrol
from matplotlib import pyplot as plt
import random
from pprint import pprint


class TabularDiffQLearning():
    def __init__(self, env=None, alpha=0.01, beta=0.01, epsilon=0.1):
        # Environment
        if env == None:
            print("Environment is None! Exiting...")
            exit(1)
        self.env = env

        # Hyperparams
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # Value function
        self.Q = dict()

        # Average-reward estimate
        self.R = 0

    
    def epsilon_greedy(self, state): 
        if random.random() > self.epsilon: 
            return np.argmax(self.Q[state])
        else: 
            return np.random.randint(0, self.env.action_space.n)


    def train(self, n_steps=2000000):
        # Reset environment
        state = self.env.reset()
        # Add state to Q table if not there
        if state not in self.Q.keys(): self.Q[state] = np.zeros(self.env.action_space.n).tolist()
        # Start learning
        for _ in range(n_steps):
            # Choose action
            action = self.epsilon_greedy(state)
            # Take action
            next_state, reward, _, _ = self.env.step(action)
            # Add state to Q table if not there
            if next_state not in self.Q.keys(): self.Q[next_state] = np.zeros(self.env.action_space.n).tolist()
            # Learn
            delta = reward - self.R + np.max(self.Q[next_state]) - self.Q[state][action]
            self.R += self.beta * delta
            self.Q[state][action] += self.alpha * delta
            # Transition
            state = next_state
