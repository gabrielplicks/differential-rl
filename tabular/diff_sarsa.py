import numpy as np
import gym
import gym_accesscontrol
from matplotlib import pyplot as plt
import random
from pprint import pprint


class TabularDiffSarsa():
    def __init__(self, env=None, alpha=0.01, beta=0.01, epsilon=0.1):
        # Environment
        if env == None:
            print("Environment is None! Exiting...")
            exit(1)
        self.env = gym.make(env)

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


    def train(self, n_steps=1000000):
        # Reset environment
        state = self.env.reset()
        action = self.epsilon_greedy(state)
        # Add state to Q table if not there
        if state not in self.Q.keys(): self.Q[state] = np.zeros(self.env.action_space.n).tolist()
        # Start learning
        for _ in range(n_steps):
            # Take action
            next_state, reward, _, _ = self.env.step(action)
            # Choose next action
            next_action = self.epsilon_greedy(next_state)
            # Add next state with zero values to Q table if not there
            if next_state not in self.Q.keys(): self.Q[next_state] = np.zeros(self.env.action_space.n).tolist()
            # Learn
            delta = reward - self.R + self.Q[next_state][next_action] - self.Q[state][action]
            self.R += self.beta * delta
            self.Q[state][action] += self.alpha * delta
            # Transition
            state = next_state
            action = next_action


if __name__ == '__main__':
    agent = TabularDiffSarsa(env="AccessControl-v0")
    agent.train(n_steps=20000000)
    pprint(agent.Q)
    pprint(agent.R)
