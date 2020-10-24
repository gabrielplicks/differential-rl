from tabular.diff_qlearning import TabularDiffQLearning
from tabular.diff_sarsa import TabularDiffSarsa
from pprint import pprint
import gym
import gym_accesscontrol


# Make env
accesscontrol = gym.make("AccessControl-v0")

# Q-Learning
qlearn = TabularDiffQLearning(env=accesscontrol)
qlearn.train()
pprint(qlearn.Q)
pprint(qlearn.R)

# Sarsa
sarsa = TabularDiffSarsa(env=accesscontrol)
sarsa.train()
pprint(sarsa.Q)
pprint(sarsa.R)
