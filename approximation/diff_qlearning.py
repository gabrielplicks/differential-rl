import numpy as np
import torch


class ApproxDiffQLearning():
    def __init__(self, env=None, model=None, optimizer=None, alpha=0.01, eta=0.01, epsilon=0.1):
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
        self.Q = model
        self.optimizer = optimizer

        # Average-reward estimate
        self.R = 0

    
    def epsilon_greedy(self, state): 
        state = torch.Tensor(np.array(state))
        if np.random.sample() > self.epsilon: 
            return torch.argmax(self.Q(state))
        else: 
            return np.random.randint(0, self.env.action_space.n)


    def train(self, n_steps=2000000):
        # Reset environment
        state = self.env.reset()

        deltas = []
        Rs = []

        # Start learning
        for _ in range(n_steps):
            # Choose action
            action = self.epsilon_greedy(state)

            # Take action
            next_state, reward, _, _ = self.env.step(action)
            
            # Learn
            self.optimizer.zero_grad()
            state_tsr = torch.Tensor(np.array(state))
            next_state_tsr = torch.Tensor(np.array(next_state))
            # print(reward, '-', self.R, '+', torch.max(self.Q(next_state_tsr)))
            q_target = reward - self.R + torch.max(self.Q(next_state_tsr))
            q_pred = self.Q(state_tsr)[action]
            # print("Target", q_target)
            # print("Pred  ", q_pred)
            delta = q_target - q_pred
            deltas.append(delta.item())
            # print("Delta ", delta)
            self.R += self.eta * self.alpha * delta.item()
            Rs.append(self.R)
            # print("R     ", self.R)
            delta.backward()
            self.optimizer.step()
            # print()
            # Transition
            state = next_state
        
        return deltas, Rs
