"""
Utility functions for PPO: return and advantage computation, evaluation.
"""

import torch


def calculate_returns(rewards, discount_factor: float):
    """
    Compute discounted normalized returns.
    """
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + discount_factor * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def calculate_advantages(returns: torch.Tensor, values: torch.Tensor):
    """
    Compute normalized advantages (returns minus values).
    """
    advantages = returns - values
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def evaluate(env, agent):
    """
    Run one episode using the current policy without exploration.
    """
    agent.eval()
    state, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent(state_tensor)
            action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
    return total_reward


def calculate_surrogate_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    epsilon: float,
    advantages: torch.Tensor
) -> torch.Tensor:
    """Compute the PPO clipped surrogate (policy) loss component."""
    advantages = advantages.detach()
    policy_ratio = (new_log_probs - old_log_probs).exp()
    surrogate1 = policy_ratio * advantages
    surrogate2 = torch.clamp(policy_ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return torch.min(surrogate1, surrogate2)

def calculate_losses(
    surrogate_loss: torch.Tensor,
    entropy: torch.Tensor,
    entropy_coefficient: float,
    returns: torch.Tensor,
    values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate total policy and value losses for PPO update."""
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    value_loss = torch.nn.functional.smooth_l1_loss(returns, values).sum()
    return policy_loss, value_loss