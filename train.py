from tabular.diff_qlearning import TabularDiffQLearning
from approximation.replay_diff_qlearning import DiffQNetworkAgent
import gym
import gym_accesscontrol
import numpy as np
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--agent", default="DiffQNetworkAgent", type=str, help="Agent to use")
    parser.add_argument("-E", "--env", default="AccessControl", type=str, help="Environment to use")
    parser.add_argument("-r", "--runs", default="15", type=int, help="Number of runs")
    parser.add_argument("-s", "--steps", default="100000", type=int, help="Number of steps")
    parser.add_argument("-a", "--alpha", default="0.01", type=float, help="Alpha value")
    parser.add_argument("-e", "--eta", default="0.01", type=float, help="Eta value")
    parser.add_argument("-u", "--update-interval", default="128", type=int, help="Update interval")
    parser.add_argument("-b", "--batch-size", default="64", type=int, help="Batch size")
    parser.add_argument("-O", "--out", default="results", type=str, help="Output folder")
    args = parser.parse_args()

    # Verify output path
    out_path = os.path.join('results', args.out)
    if os.path.exists(out_path):
        return args
    else:
        os.makedirs(out_path)
        return args


if __name__ == '__main__':
    # Example
    # python train.py -A DiffQNetworkAgent -E AccessControl -r 15 -s 100000 -a 0.01 -e 0.01 -u 128 -b 64 -O outfolder
    args = parse_arguments()

    for run in range(1,args.runs+1):
        # Intantiate env
        if args.env == 'AccessControl':
            env = gym.make("AccessControl-v0", rand_seed=run)
        else:
            print('Invalid environment.')
            exit(1)
        # Instantiate agent
        if args.agent == 'DiffQNetworkAgent': 
            agent = DiffQNetworkAgent(env=env, alpha=args.alpha, eta=args.eta, epsilon=0.1, rand_seed=run, update_interval=args.update_interval, batch_size=args.batch_size, memory_size=10000)
            # Start training
            print('Running {}...'.format(run))
            qnet, qnet_target, R, rewards, avg_rewards, losses = agent.train(n_steps=args.steps)
            np.save('{0}/losses_{1}_{2}_{3}_{4}_{5}.npy'.format(args.out, args.alpha, args.eta, args.update_interval, args.batch_size, run), losses, allow_pickle=True)
            np.save('{0}/rewards_{1}_{2}_{3}_{4}_{5}.npy'.format(args.out, args.alpha, args.eta, args.update_interval, args.batch_size, run), rewards, allow_pickle=True)
            np.save('{0}/avg_rewards_{1}_{2}_{3}_{4}_{5}.npy'.format(args.out, args.alpha, args.eta, args.update_interval, args.batch_size, run), avg_rewards, allow_pickle=True)
        elif args.agent == 'TabularDiffQLearning': 
            agent = TabularDiffQLearning(env=env, alpha=args.alpha, eta=args.eta, epsilon=0.1, rand_seed=run)
            # Start training
            print('Running {}...'.format(run))
            Q, R, rewards, avg_rewards = agent.train(n_steps=n_steps)
            np.save('{0}/rewards_{1}_{2}_{3}.npy'.format(args.out, args.alpha, args.eta, run), rewards, allow_pickle=True)
            np.save('{0}/avg_rewards_{1}_{2}_{3}.npy'.format(args.out, args.alpha, args.eta, run), avg_rewards, allow_pickle=True)
        else:
            print("Invalid agent.")
            exit(1)
