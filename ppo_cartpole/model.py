"""Model definitions for PPO agent: MLPNetwork and ActorCritic."""

import torch.nn as nn


class MLPNetwork(nn.Module):
    """
    Multi-layer perceptron for actor or critic.

    Args:
        input_dim: Dimension of input features.
        hidden_dims: Sequence of hidden layer sizes.
        output_dim: Dimension of output features.
        dropout: Dropout probability applied after each hidden layer.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """Combined actor and critic networks."""

    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_logits = self.actor(state)
        state_value = self.critic(state)
        return action_logits, state_value