# This plot script is based on @abhisheknaik96's script
# In https://github.com/abhisheknaik96/average-reward-methods

import numpy as np
from matplotlib import pyplot as plt

folders = ['results_diffqnetwork', 'results_nonlinearfreezediffqlearning', 'results_nonlineardiffqlearning', 'results_tabulardiffqlearning']
labels = {'results_diffqnetwork':           'Diff. Q-network (xp-replay, freezing avg-reward)', 
        'results_nonlinearfreezediffqlearning':   'Non-linear F.A. Diff. Q-learning (freezing avg-reward', 
        'results_nonlineardiffqlearning':   'Non-linear F.A. Diff. Q-learning', 
        'results_tabulardiffqlearning':     'Tabular Diff. Q-learning'}

runs = 30
n_steps = 200000

# Load rewards
all_rewards = dict()
for folder in folders:
    all_rewards[folder] = list()
    for run in range(1,runs+1):
        rewards = np.load('{}/rewards_{}.npy'.format(folder, run), allow_pickle=True)
        all_rewards[folder].append(np.array(rewards))
    all_rewards[folder] = np.array(all_rewards[folder])

# Get reward rates
window = 1000
sample = 1000
conv_arr = np.ones(window)
for folder in folders:
    all_reward_rates = list()
    for run in range(runs):
        reward_rates = np.convolve(all_rewards[folder][run], conv_arr, mode='valid') / window
        all_reward_rates.append(reward_rates)
    all_reward_rates = np.array(all_reward_rates)

    # Compute mean and std error
    mean = np.mean(all_reward_rates, axis=0)
    stderr = np.std(all_reward_rates, axis=0) / np.sqrt(all_reward_rates.shape[0])

    # Plot
    x_axis = np.arange(window, n_steps+1)[::sample]
    y_axis = mean[::sample]
    stderr = stderr[::sample]
    plt.plot(x_axis, y_axis, label=labels[folder])
    plt.fill_between(x_axis, y_axis + stderr, y_axis - stderr, alpha=0.3)

plt.ylabel('Reward rate')
plt.xlabel('Step')
plt.grid()
plt.legend()
plt.show()
