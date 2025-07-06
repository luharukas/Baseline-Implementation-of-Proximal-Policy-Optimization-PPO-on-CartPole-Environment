"""Agent and training functions for PPO."""

from typing import Tuple

import torch
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset

from .model import MLPNetwork, ActorCritic
from .utils import calculate_returns, calculate_advantages


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
    state, _ = env.reset()

    agent.train()
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
        values.append(value.squeeze(-1))
        rewards.append(reward)

        episode_reward += reward
        state = next_state

    states = torch.cat(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(log_probs).detach()
    values = torch.stack(values).detach()
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
        average policy loss, average value loss over all updates.
    """
    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_policy_loss = 0.0
    total_value_loss = 0.0

    for _ in range(ppo_steps):
        for b_states, b_actions, b_old_log_probs, b_advantages, b_returns in loader:
            logits, values = agent(b_states)
            values = values.squeeze(-1)
            probs = torch.softmax(logits, dim=-1)
            dist = distributions.Categorical(probs)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(b_actions)

            ratio = (new_log_probs - b_old_log_probs).exp()
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * b_advantages
            policy_loss = -(torch.min(surr1, surr2).mean() + entropy_coeff * entropy)

            value_loss = torch.nn.functional.mse_loss(b_returns, values)

            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    num_updates = ppo_steps * len(loader)
    return total_policy_loss / num_updates, total_value_loss / num_updates