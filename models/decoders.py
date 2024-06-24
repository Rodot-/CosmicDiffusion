import typing

import numpy as np
import torch
from torch import nn


class BaseDecoder(nn.Module):
    """Base decoder specifying minimal init protocol."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs


class MLPDecoder(BaseDecoder):
    """Multi-layer perceptron decoder."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: typing.Tuple[int, ...] = (64, 256),
        activation: str = "LeakyReLU",
    ) -> None:
        """Initializer for multi-layer perceptron decoder.

        Args:
            latent_dim: input latent space feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(latent_dim=latent_dim)
        self.hidden_dims = hidden_dims

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), activation_fn))
            in_channels = h_dim
        self.mlp = nn.Sequential(*modules)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        dec = self.mlp(inputs)
        return dec


class Label_MLPDecoder(BaseDecoder):
    """Multi-layer perceptron decoder."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        output_dim: int = 1100,
        label_dim: int = 4,
        activation: str = "LeakyReLU",
        is_residual_on: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initializer for multi-layer perceptron decoder.

        Args:
            latent_dim: input latent space feature dimension
            hidden_dims: tuple specifying feature dimensions of hidden layers
        """
        super().__init__(latent_dim=latent_dim)
        hidden_dims = [hidden_dim] * hidden_layers + [output_dim]
        self.hidden_dims = hidden_dims
        self.is_residual_on = is_residual_on

        # TODO: modify to a proper method
        activation_fn = eval("nn.%s()" % activation)

        modules = []
        in_channels = latent_dim + label_dim
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

        self.observation_dim = hidden_dims[-1]

    def std_to_bound(self, reconstruction_std):
        bound = (
            0.5
            * torch.prod(torch.tensor(self.observation_dim, dtype=torch.float32))
            * (
                1.0
                + torch.log(torch.tensor(2 * np.pi, dtype=torch.float32))
                + 2 * torch.log(torch.tensor(reconstruction_std, dtype=torch.float32))
            )
        )
        return bound

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
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
