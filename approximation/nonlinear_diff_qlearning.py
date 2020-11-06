import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super(QNet, self).__init__()
        # Architecture
        self.nn = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.nn(x)


class NonLinearDiffQLearningAgent:
    def __init__(self, env=None, alpha=0.01, eta=0.01, epsilon=0.1, rand_seed=22):
        # Set randoms seed
        np.random.seed(rand_seed)

        # Environment
        if env == None:
            print("Environment is None! Exiting...")
            exit(1)
        self.env = env
        self.n_features = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # Hyperparameters
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        # Average-reward estimate
        self.R_current = 0

        # Model
        self.qnet = QNet(self.n_features, self.n_actions)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.alpha)


    def epsilon_greedy(self, state):
        if np.random.sample() > self.epsilon: 
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_value = self.qnet.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def train(self, n_steps=80000):
        # Init stats
        rewards = list()
        avg_rewards = list()
        losses = list()
        # Reset environment
        state = self.env.reset()
        # Start learning
        for step in range(n_steps):
            # Choose action
            action = self.epsilon_greedy(state)
            # Take action
            next_state, reward, _, _ = self.env.step(action)




            # Learn abnd compute average-reward estimate
            torch_state = torch.tensor(list(state), dtype=torch.float)
            torch_action = torch.tensor(action, dtype=torch.long)
            torch_reward = torch.tensor(reward, dtype=torch.float)
            torch_next_state = torch.tensor(list(state), dtype=torch.float)
            torch_r_current = torch.tensor(self.R_current, dtype=torch.float)
            q_value = self.qnet(torch_state)[torch_action]
            q_target = torch_reward.item() - torch_r_current.item() + torch.max(self.qnet(torch_next_state))
            
            loss = (q_target - q_value).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())


            delta = q_target.item() - q_value.item()
            self.R_current += self.eta * self.alpha * delta




            # Transition
            state = next_state
            # Save stats
            rewards.append(float(reward))
            avg_rewards.append(float(self.R_current))
            
        return self.qnet, self.R_current, rewards, avg_rewards, losses
