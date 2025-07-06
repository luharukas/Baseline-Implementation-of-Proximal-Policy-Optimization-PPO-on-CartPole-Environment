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