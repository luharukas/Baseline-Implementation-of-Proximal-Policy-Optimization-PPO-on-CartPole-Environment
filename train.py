#!/usr/bin/env python3
"""
Train PPO agent on CartPole-v1 environment.
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from ppo_cartpole.agent import create_agent, forward_pass, update_policy
from ppo_cartpole.utils import evaluate
from ppo_cartpole.plot import plot_train_rewards, plot_test_rewards, plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description='PPO training for CartPole-v1')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--threshold', type=float, default=475)
    parser.add_argument('--print-interval', type=int, default=10)
    parser.add_argument('--ppo-steps', type=int, default=8)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument(
        '--hidden-dims',
        type=int,
        default=64,
    )
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()


def run_ppo(env_train, env_test, agent, optimizer, args):
    """Run PPO training loop and return performance metrics."""
    train_rewards, test_rewards = [], []
    policy_losses, value_losses = [], []

    for episode in range(1, args.episodes + 1):
        train_reward, states, actions, old_log_probs, advantages, returns = forward_pass(
            env_train, agent, args.discount)
        policy_loss, value_loss = update_policy(
            agent, states, actions, old_log_probs, advantages, returns,
            optimizer, args.ppo_steps, args.epsilon, args.entropy_coef, args.batch_size)
        test_reward = evaluate(env_test, agent)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        if episode % args.print_interval == 0:
            mean_train = np.mean(train_rewards[-args.trials:])
            mean_test = np.mean(test_rewards[-args.trials:])
            mean_policy_loss = np.mean(np.abs(policy_losses[-args.trials:]))
            mean_value_loss = np.mean(np.abs(value_losses[-args.trials:]))
            print(f'Episode {episode:3d} '
                  f'Mean Train Reward: {mean_train:.1f} '
                  f'Mean Test Reward: {mean_test:.1f} '
                  f'Mean Policy Loss: {mean_policy_loss:.3f} '
                  f'Mean Value Loss: {mean_value_loss:.3f}')
        if np.mean(test_rewards[-args.trials:]) >= args.threshold:
            print(f'Solved in {episode} episodes!')
            break

    return train_rewards, test_rewards, policy_losses, value_losses


def main():
    args = parse_args()
    env_train = gym.make(args.env)
    env_test = gym.make(args.env)
    obs_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n
    print(f'Observation space dimension: {obs_dim}, Action space dimension: {action_dim}')
    agent = create_agent(obs_dim, action_dim, args.hidden_dims, args.dropout)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    train_rewards, test_rewards, policy_losses, value_losses = run_ppo(
        env_train, env_test, agent, optimizer, args)

    plot_train_rewards(train_rewards, args.threshold)
    plot_test_rewards(test_rewards, args.threshold)
    plot_losses(policy_losses, value_losses)

    torch.save(agent.actor.state_dict(), 'actor.pth')
    torch.save(agent.critic.state_dict(), 'critic.pth')
    print("Saved actor model to actor.pth")
    print("Saved critic model to critic.pth")


if __name__ == '__main__':
    main()