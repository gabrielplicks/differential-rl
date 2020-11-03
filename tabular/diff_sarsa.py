import numpy as np


class TabularDiffSarsa():
    def __init__(self, env=None, alpha=0.01, eta=0.01, epsilon=0.1):
        # Environment
        if env == None:
            print("Environment is None! Exiting...")
            exit(1)
        self.env = env

        # Hyperparams
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        # Value function
        self.Q = dict()

        # Average-reward estimate
        self.R = 0

    
    def epsilon_greedy(self, state): 
        if np.random.sample() > self.epsilon: 
            return np.argmax(self.Q[state])
        else: 
            return np.random.randint(0, self.env.action_space.n)


    def train(self, n_steps=2000000):
        # Reset environment
        state = self.env.reset()
        # Add state to Q table if not there
        if state not in self.Q.keys(): self.Q[state] = np.zeros(self.env.action_space.n).tolist()
        # Choose action
        action = self.epsilon_greedy(state)
        # Start learning
        for _ in range(n_steps):
            # Take action
            next_state, reward, _, _ = self.env.step(action)
            # Add next state with zero values to Q table if not there
            if next_state not in self.Q.keys(): self.Q[next_state] = np.zeros(self.env.action_space.n).tolist()
            # Choose next action
            next_action = self.epsilon_greedy(next_state)
            # Learn
            delta = reward - self.R + self.Q[next_state][next_action] - self.Q[state][action]
            self.Q[state][action] += self.alpha * delta
            self.R += self.eta * self.alpha * delta
            # Transition
            state = next_state
            action = next_action
