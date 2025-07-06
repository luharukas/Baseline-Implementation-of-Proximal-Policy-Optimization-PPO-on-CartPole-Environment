"""Agent and training functions for PPO."""

from typing import Tuple

import torch
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset

from .model import MLPNetwork, ActorCritic
from .utils import calculate_returns, calculate_advantages, calculate_surrogate_loss, calculate_losses


def create_agent(
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    dropout: float
) -> ActorCritic:
    """
    Create an ActorCritic agent with configurable MLP networks.

    Args:
        obs_dim: Dimension of the observation space.
        action_dim: Number of discrete actions.
        hidden_dims: Sequence of hidden layer sizes.
        dropout: Dropout probability after each hidden layer.

    Returns:
        An ActorCritic instance combining policy and value networks.
    """
    actor = MLPNetwork(obs_dim, hidden_dims, action_dim, dropout)
    critic = MLPNetwork(obs_dim, hidden_dims, 1, dropout)
    return ActorCritic(actor, critic)


def forward_pass(env, agent: ActorCritic, discount_factor: float) -> Tuple[
    float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Run one episode to collect training data.

    Returns:
        episode_reward, states, actions, old_log_probs, advantages, returns
    """
    states, actions, log_probs, values, rewards = [], [], [], [], []
    done = False
    episode_reward = 0.0
    state = env.reset()[0]

    agent.train()
    while not done:
        state_tensor =torch.FloatTensor(state).unsqueeze(0)
        states.append(state_tensor)

        logits, value = agent(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        actions.append(action)
        log_probs.append(log_prob)
        # remove all singleton dims to get scalar value prediction
        values.append(value)
        rewards.append(reward)

        episode_reward += reward
        state = next_state

    states = torch.cat(states)
    actions = torch.cat(actions)
    old_log_probs = torch.cat(log_probs)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    return episode_reward, states, actions, old_log_probs, advantages, returns


def update_policy(
    agent: ActorCritic,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ppo_steps: int,
    epsilon: float,
    entropy_coeff: float,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Update policy and value networks using PPO algorithm.

    Returns:
        average policy loss, average value loss (averaged per PPO step).
    """
    BATCH_SIZE = batch_size
    total_policy_loss = 0.0
    total_value_loss = 0.0
    old_log_probs = old_log_probs.detach()
    actions = actions.detach()
    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    for _ in range(ppo_steps):
        for b_states, b_actions, b_old_log_probs, b_advantages, b_returns in loader:
            logits, value_pred = agent(b_states)
            value_pred = value_pred.squeeze(-1)
            probs = torch.softmax(logits, dim=-1)
            dist = distributions.Categorical(probs)

            entropy = dist.entropy()
            new_log_probs = dist.log_prob(b_actions)

            surrogate_loss = calculate_surrogate_loss(
                b_old_log_probs, new_log_probs, epsilon, b_advantages)
            policy_loss, value_loss = calculate_losses(
                surrogate_loss, entropy, entropy_coeff, b_returns, value_pred)

            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


