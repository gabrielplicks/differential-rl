import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
import random


class ReplayMemory():
    def __init__(self, capacity, rand_seed=22):
        self.memory = collections.deque(maxlen=capacity)
        random.seed(rand_seed)

    def __len__(self):
        return len(self.memory)

    def add(self, s0, a, r, s1):
        self.memory.append((s0, [a], [r], s1))

    def sample(self, batch_size):
        s0, a, r, s1 = zip(*random.sample(self.memory, batch_size))
        s0 = torch.tensor(list(s0), dtype=torch.float)
        s1 = torch.tensor(list(s1), dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        return s0, a, r, s1


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


class DiffQNetworkAgent:
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
        self.R_freeze = None

        # Experience replay
        self.memory_size = 10000
        self.memory = ReplayMemory(self.memory_size, rand_seed)
        self.target_update_interval = 128
        self.batch_size = 256

        # Model
        self.qnet = QNet(self.n_features, self.n_actions)
        self.qnet_target = QNet(self.n_features, self.n_actions)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.alpha)


    def epsilon_greedy(self, state):
        if np.random.sample() > self.epsilon: 
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_value = self.qnet.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def learn_batch(self):
        s0, a, r, s1 = self.memory.sample(self.batch_size)

        q_values = self.qnet(s0).gather(1,a)
        max_q_values = self.qnet_target(s1).max(1)[0].unsqueeze(1)
        q_targets = r - self.R_freeze + max_q_values

        loss = (q_targets - q_values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


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
            # Add to replay memory
            self.memory.add(state, action, reward, next_state)
            # Compute average-reward estimate
            torch_state = torch.tensor(list(state), dtype=torch.float)
            torch_action = torch.tensor(action, dtype=torch.long)
            torch_reward = torch.tensor(reward, dtype=torch.float)
            torch_next_state = torch.tensor(list(state), dtype=torch.float)
            torch_r_current = torch.tensor(self.R_current, dtype=torch.float)
            q_value = self.qnet(torch_state)[torch_action].item()
            q_target = torch_reward.item() - torch_r_current.item() + torch.max(self.qnet_target(torch_next_state)).item()
            delta = q_target - q_value
            self.R_current += self.eta * self.alpha * delta
            # Transition
            state = next_state
            # Save stats
            rewards.append(float(reward))
            avg_rewards.append(float(self.R_current))
            # Learn
            if len(self.memory) > self.batch_size:
                if self.R_freeze == None:
                    self.R_freeze = self.R_current
                loss = self.learn_batch()
                losses.append(loss)
            # Update interval
            if (step != 0 and step % self.target_update_interval == 0):
                # Update average-reward estimate
                self.R_freeze = self.R_current
                # Update target weights
                self.qnet_target.load_state_dict(self.qnet.state_dict())
            
        return self.qnet, self.qnet_target, self.R_current, rewards, avg_rewards, losses
