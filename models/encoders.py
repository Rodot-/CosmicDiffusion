import typing

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BaseEncoder(nn.Module):
    """Base encoder specifying minimal init protocol."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: typing.Tuple[int, ...] = (32, 256),
        activation: str = "LeakyReLU",
        dropout: float = 0.0,
    ) -> None:
        """Initializer for multi-layer perceptron encoder.

        Args:
            observation_dim: input feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(observation_dim=observation_dim)
        self.hidden_dims = hidden_dims

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = observation_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation_fn,
                    nn.Dropout(p=dropout),
                )
            )
            # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            in_channels = h_dim
        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class Label_MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        label_dim: int = 4,
        activation: str = "LeakyReLU",
        is_residual_on: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializer for multi-layer perceptron encoder.

        Args:
            observation_dim: input feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(observation_dim=observation_dim)
        hidden_dims = [hidden_dim] * hidden_layers
        self.hidden_dims = hidden_dims
        self.is_residual_on = is_residual_on

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = int(observation_dim + label_dim)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    activation_fn,
                    nn.Dropout(p=dropout),
                )
            )
            # modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            in_channels = h_dim
        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.is_residual_on:
            outputs_pre = self.mlp[0](inputs)
            outputs_curr = self.mlp[1](outputs_pre)
            for t in range(2, len(self.mlp)):
                outputs = self.mlp[t](outputs_pre + outputs_curr)
                outputs_pre = outputs_curr
                outputs_curr = outputs
        else:
            outputs = self.mlp(inputs)
        return outputs
