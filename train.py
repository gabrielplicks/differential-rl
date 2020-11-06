from tabular.diff_qlearning import TabularDiffQLearning
from approximation.replay_diff_qlearning import DiffQNetworkAgent
from approximation.nonlinear_diff_qlearning import NonLinearDiffQNetworkAgent
from pprint import pprint
import gym
import gym_accesscontrol
import gym_inventory
import torch
import numpy as np
from matplotlib import pyplot as plt

runs = 30
n_steps = 200000

# Tabular Differential Q-Learning
for run in range(1,runs+1):
    print('Running {}...'.format(run))
    accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
    agent = TabularDiffQLearning(env=accesscontrol, alpha=0.025, eta=0.5, epsilon=0.1, rand_seed=run)
    Q, R, rewards, avg_rewards = agent.train(n_steps=n_steps)
    np.save('results_tabulardiffqlearning/Q_{}.npy'.format(run), Q, allow_pickle=True)
    np.save('results_tabulardiffqlearning/rewards_{}.npy'.format(run), rewards, allow_pickle=True)
    np.save('results_tabulardiffqlearning/avg_rewards_{}.npy'.format(run), avg_rewards, allow_pickle=True)

# Non-linear Differential Q-learning
for run in range(1,runs+1):
    print('Running {}...'.format(run))
    accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
    agent = NonLinearDiffQNetworkAgent(env=accesscontrol, alpha=0.001, eta=0.05, epsilon=0.1, rand_seed=run)
    qnet, R, rewards, avg_rewards, losses = agent.train(n_steps=n_steps)
    np.save('results_nonlineardiffqlearning/losses_{}.npy'.format(run), losses, allow_pickle=True)
    np.save('results_nonlineardiffqlearning/rewards_{}.npy'.format(run), rewards, allow_pickle=True)
    np.save('results_nonlineardiffqlearning/avg_rewards_{}.npy'.format(run), avg_rewards, allow_pickle=True)

# Differential Q-network (experience replay)
for run in range(1,runs+1):
    print('Running {}...'.format(run))
    accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
    agent = DiffQNetworkAgent(env=accesscontrol, alpha=0.0001, eta=0.05, epsilon=0.1, rand_seed=run)
    qnet, qnet_target, R, rewards, avg_rewards, losses = agent.train(n_steps=n_steps)
    np.save('results_diffqnetwork/losses_{}.npy'.format(run), losses, allow_pickle=True)
    np.save('results_diffqnetwork/rewards_{}.npy'.format(run), rewards, allow_pickle=True)
    np.save('results_diffqnetwork/avg_rewards_{}.npy'.format(run), avg_rewards, allow_pickle=True)
