# This plot script is based on @abhisheknaik96's script
# In https://github.com/abhisheknaik96/average-reward-methods

# Load rewards
tab_rewards = list()
dqn_rewards = list()
for run in range(1,runs+1):
    tab_reward = np.load('results_diffqn/rewards_{}.npy'.format(run), allow_pickle=True)
    dqn_reward = np.load('results/rewards_{}.npy'.format(run), allow_pickle=True)
    tab_rewards.append(np.array(tab_reward))
    dqn_rewards.append(np.array(dqn_reward))
tab_rewards = np.array(tab_rewards)
dqn_rewards = np.array(dqn_rewards)

# Get reward rates
window = 2000
sample = 800
conv_arr = np.ones(window)
tab_reward_rates = list()
dqn_reward_rates = list()
for run in range(runs):
    tab_reward_rate = np.convolve(tab_rewards[run], conv_arr, mode='valid') / window
    dqn_reward_rate = np.convolve(dqn_rewards[run], conv_arr, mode='valid') / window
    tab_reward_rates.append(tab_reward_rate)
    dqn_reward_rates.append(dqn_reward_rate)
tab_reward_rates = np.array(tab_reward_rates)
dqn_reward_rates = np.array(dqn_reward_rates)
tab_mean = np.mean(tab_reward_rates, axis=0)
dqn_mean = np.mean(dqn_reward_rates, axis=0)
tab_stderr = np.std(tab_reward_rates, axis=0) / np.sqrt(tab_reward_rates.shape[0])
dqn_stderr = np.std(dqn_reward_rates, axis=0) / np.sqrt(dqn_reward_rates.shape[0])
x_axis = np.arange(window, n_steps+1)[::sample]
tab_y_axis = tab_mean[::sample]
dqn_y_axis = dqn_mean[::sample]
tab_stderr = tab_stderr[::sample]
dqn_stderr = dqn_stderr[::sample]
plt.plot(x_axis, tab_y_axis, label='Tabular Diff. Q-learning')
plt.fill_between(x_axis, tab_y_axis + tab_stderr, tab_y_axis - tab_stderr, alpha=0.3)
plt.plot(x_axis, dqn_y_axis, label='Diff. Q-network (xp-replay)')
plt.fill_between(x_axis, dqn_y_axis + dqn_stderr, dqn_y_axis - dqn_stderr, alpha=0.3)
plt.grid()
plt.legend()
plt.show()
