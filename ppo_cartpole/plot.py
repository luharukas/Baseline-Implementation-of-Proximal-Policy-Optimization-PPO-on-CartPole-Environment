"""
Plotting functions for PPO training and evaluation metrics.
"""

import matplotlib.pyplot as plt


def plot_train_rewards(train_rewards, threshold):
    """Plot training rewards over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label='Train Reward')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Reward Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # save figure to disk
    plt.savefig('train_rewards.png')
    plt.show()


def plot_test_rewards(test_rewards, threshold):
    """Plot testing rewards over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(test_rewards, label='Test Reward')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Reward Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Testing Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # save figure to disk
    plt.savefig('test_rewards.png')
    plt.show()


def plot_losses(policy_losses, value_losses):
    """Plot policy and value losses over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Policy and Value Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # save figure to disk
    plt.savefig('losses.png')
    plt.show()