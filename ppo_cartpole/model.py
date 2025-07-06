"""Model definitions for PPO agent: MLPNetwork and ActorCritic."""

import torch.nn as nn
import torch.nn.functional as f

class MLPNetwork(nn.Module):
    """
    Multi-layer perceptron for actor or critic.

    Args:
        input_dim: Dimension of input features.
        hidden_dims: Sequence of hidden layer sizes.
        output_dim: Dimension of output features.
        dropout: Dropout probability applied after each hidden layer.
    """
    # This class defines a simple feedforward neural network to be used as a backbone
    # for both the actor and critic networks.
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        # Initialize the network layers.
        # layer1: Linear transformation from input features to hidden dimensions.
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        # layer2: Linear transformation between hidden layers.
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        # layer3: Linear transformation from hidden dimensions to the output features.
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        # dropout: Dropout layer for regularization to prevent overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Define the forward pass of the network.
        # Apply the first linear layer.
        x = self.layer1(x)
        # Apply the ReLU activation function.
        x = f.relu(x)
        # Apply dropout.
        x = self.dropout(x)
        # Apply the second linear layer.
        x = self.layer2(x)
        # Apply the ReLU activation function.
        x = f.relu(x)
        # Apply dropout.
        x = self.dropout(x)
        # Apply the third linear layer to get the final output.
        x = self.layer3(x)
        return x


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

