#!/usr/bin/env python3
"""
Test a trained PPO agent on CartPole-v1 by loading saved actor and critic models.
"""

import argparse
import gymnasium as gym
import torch

from ppo_cartpole.agent import create_agent
from ppo_cartpole.utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test trained PPO agent on CartPole-v1"
    )
    parser.add_argument(
        '--env',
        type=str,
        default='CartPole-v1',
        help='Gym environment name'
    )
    parser.add_argument(
        '--actor-path',
        type=str,
        default='actor.pth',
        help='Path to saved actor model'
    )
    parser.add_argument(
        '--critic-path',
        type=str,
        default='critic.pth',
        help='Path to saved critic model'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        default=64,
        help='Comma-separated sizes for hidden layers, e.g. "64,64"'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout probability for network'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes to evaluate'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = create_agent(obs_dim, action_dim, args.hidden_dims, args.dropout)
    agent.actor.load_state_dict(torch.load(args.actor_path))
    agent.critic.load_state_dict(torch.load(args.critic_path))
    agent.eval()

    rewards = [evaluate(env, agent) for _ in range(args.episodes)]
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")


if __name__ == '__main__':
    main()