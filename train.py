from tabular.diff_qlearning import TabularDiffQLearning
from approximation.replay_diff_qlearning import DiffQNetworkAgent
from approximation.nonlinear_diff_qlearning import NonLinearDiffQLearningAgent
from approximation.nonlinear_freeze_diff_qlearning import NonLinearFreezeDiffQLearningAgent
import gym
import gym_accesscontrol
import torch
import numpy as np
from matplotlib import pyplot as plt

runs = 15
n_steps = 100000

# Tabular Differential Q-Learning
avg_rs = list()
for run in range(1,runs+1):
    print('Running {}...'.format(run))
    accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
    agent = TabularDiffQLearning(env=accesscontrol, alpha=0.025, eta=0.5, epsilon=0.1, rand_seed=run)
    Q, R, rewards, avg_rewards = agent.train(n_steps=n_steps)
    avg_rs.append(R)
    np.save('results_tabulardiffqlearning/Q_{}.npy'.format(run), Q, allow_pickle=True)
    np.save('results_tabulardiffqlearning/rewards_{}.npy'.format(run), rewards, allow_pickle=True)
    np.save('results_tabulardiffqlearning/avg_rewards_{}.npy'.format(run), avg_rewards, allow_pickle=True)
print(avg_rs)

# Differential Q-network (experience replay)
alphas = [0.01, 0.005, 0.001, 0.0005]
etas = [0.5, 0.1, 0.05, 0.01, 0.005]
update_intervals = [32, 64, 128, 256]
batch_sizes = [32, 64, 128, 256]
alphas.reverse()
etas.reverse()
update_intervals.reverse()
batch_sizes.reverse()
for alpha in alphas:
    for eta in etas:
        for update_interval in update_intervals:
            for batch_size in batch_sizes:
                print('\nalpha {0} | eta {1} | update_interval {2} | batch_size {3}'.format(alpha, eta, update_interval, batch_size))
                avg_rs = list()
                for run in range(1,runs+1):
                    print('Running {}...'.format(run))
                    accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
                    agent = DiffQNetworkAgent(env=accesscontrol, alpha=alpha, eta=eta, epsilon=0.1, rand_seed=run, update_interval=update_interval, batch_size=batch_size, memory_size=10000)
                    qnet, qnet_target, R, rewards, avg_rewards, losses = agent.train(n_steps=n_steps)
                    avg_rs.append(R)
                    np.save('results_diffqnetwork/losses_{0}_{1}_{2}_{3}_{4}.npy'.format(alpha, eta, update_interval, batch_size, run), losses, allow_pickle=True)
                    np.save('results_diffqnetwork/rewards_{0}_{1}_{2}_{3}_{4}.npy'.format(alpha, eta, update_interval, batch_size, run), rewards, allow_pickle=True)
                    np.save('results_diffqnetwork/avg_rewards_{0}_{1}_{2}_{3}_{4}.npy'.format(alpha, eta, update_interval, batch_size, run), avg_rewards, allow_pickle=True)
                print(avg_rs)

# # Non-linear Differential Q-learning
# for run in range(1,runs+1):
#     print('Running {}...'.format(run))
#     accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
#     agent = NonLinearDiffQLearningAgent(env=accesscontrol, alpha=0.0001, eta=0.05, epsilon=0.1, rand_seed=run)
#     qnet, R, rewards, avg_rewards, losses = agent.train(n_steps=n_steps)
#     np.save('results_nonlineardiffqlearning_new/losses_{}.npy'.format(run), losses, allow_pickle=True)
#     np.save('results_nonlineardiffqlearning_new/rewards_{}.npy'.format(run), rewards, allow_pickle=True)
#     np.save('results_nonlineardiffqlearning_new/avg_rewards_{}.npy'.format(run), avg_rewards, allow_pickle=True)

# # Non-linear Differential Q-learning (freezing avg-reward)
# for run in range(1,runs+1):
#     print('Running {}...'.format(run))
#     accesscontrol = gym.make("AccessControl-v0", rand_seed=run)
#     agent = NonLinearFreezeDiffQLearningAgent(env=accesscontrol, alpha=0.0001, eta=0.05, epsilon=0.1, rand_seed=run)
#     qnet, R, rewards, avg_rewards, losses = agent.train(n_steps=n_steps)
#     np.save('results_nonlinearfreezediffqlearning_new/losses_{}.npy'.format(run), losses, allow_pickle=True)
#     np.save('results_nonlinearfreezediffqlearning_new/rewards_{}.npy'.format(run), rewards, allow_pickle=True)
#     np.save('results_nonlinearfreezediffqlearning_new/avg_rewards_{}.npy'.format(run), avg_rewards, allow_pickle=True)

