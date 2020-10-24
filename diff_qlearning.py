import numpy as np
import gym
import gym_accesscontrol
from matplotlib import pyplot as plt
import random
from pprint import pprint


class DiffQLearning():
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

        # Training
        self.N_STEPS = 5000000

    
    def epsilon_greedy(self, state):
        if random.random() > self.epsilon:
            return self.Q[state].index(max(self.Q[state]))
        else:
            return random.randrange(self.env.action_space.n)


    def train(self):
        # Stats
        # rewards = list()
        # plt.ion()
        # plt.show()
        # plt.pause(0.0001)

        # Reset environment
        state = self.env.reset()
        state = tuple(state)
        # Add state to Q table if not there
        if state not in self.Q.keys(): self.Q[state] = np.zeros(self.env.action_space.n).tolist()
        # Start learning
        for _ in range(self.N_STEPS):
            # Select and apply action
            action = self.epsilon_greedy(state)
            next_state, reward, _, _ = self.env.step(action)
            next_state = tuple(next_state)
            # Add state to Q table if not there
            if next_state not in self.Q.keys(): self.Q[next_state] = np.zeros(self.env.action_space.n).tolist()
            # Learn
            delta = reward - self.R + np.max(self.Q[next_state]) - self.Q[state][action]
            self.R += self.beta * delta
            self.Q[state][action] += self.alpha * delta
            # Transition
            state = next_state
            
            # Stats
            # rewards.append(reward)
            # plt.plot(rewards)
            # plt.draw()
            # plt.pause(0.0001)


if __name__ == '__main__':
    agent = DiffQLearning(env="AccessControl-v0")
    agent.train()
    pprint(agent.Q)
    pprint(agent.R)
