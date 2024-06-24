import numpy as np
import torch
from torch import nn


# TODO: inherit ResNet and DenseNet from MLP.
class MLP(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden,
        n_layers,
        init,
        activation,
        dropout,
        norm,
        epsilon=1e-4,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.init = init
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.n_inputs = n_inputs
        self.epsilon = epsilon

        self.layers = []
        for i in range(0, self.n_layers):
            # sigma-pi, plus initialisation
            layer = nn.Linear(in_features=n_inputs, out_features=self.n_hidden)
            eval("torch.nn.init.%s(layer.weight)" % self.init)
            self.layers.append(layer)
            # then the activation function
            cmd = "nn.%s()" % self.activation
            self.layers.append(eval(cmd))
            n_inputs = self.n_hidden
            # then normalisation layers
            if not np.isclose(self.dropout, 0):
                self.layers.append(nn.Dropout(self.dropout))
            if self.norm == "batch":
                self.layers.append(nn.BatchNorm1d(self.n_hidden))
            elif self.norm == "layer":
                self.layers.append(nn.LayerNorm(self.n_hidden))
        # and a linear output at the end
        self.layers.append(nn.Linear(self.n_hidden, self.n_outputs))
        self.layers.append(eval(cmd))
        self.neuralnet = nn.Sequential(*self.layers)

        self.mean_layer = nn.Linear(self.n_outputs, self.n_outputs)
        self.std_layer = nn.Sequential(
            nn.Linear(self.n_outputs, self.n_outputs), nn.Softplus()
        )

    def forward(self, inputs):
        outputs = self.neuralnet(inputs)
        mu = self.mean_layer(outputs)
        sigma = self.std_layer(outputs) + self.epsilon
        return mu, sigma


class ResNet(nn.Module):
    # TODO: remove un-used parameters

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden,
        n_layers,
        init,
        activation,
        dropout,
        norm,
        epsilon=1e-4,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.activation = activation
        self.init = init

        dims = (n_inputs,) + (n_hidden,) * n_layers + (n_outputs,)

        sequential = []
        for t in range(len(dims) - 1):
            sequential.append(
                nn.Sequential(
                    nn.Linear(dims[t], dims[t + 1]),
                    # eval("torch.nn.init.%s(layer.weight)" % self.init),
                    eval("nn.%s()" % self.activation),
                )
            )

        self.sequential = nn.Sequential(*sequential)

        self.mean_layer = nn.Linear(n_outputs, n_outputs)
        self.std_layer = nn.Sequential(nn.Linear(n_outputs, n_outputs), nn.Softplus())

        for p in self.sequential.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

        for p in self.mean_layer.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

        for p in self.std_layer.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

    def forward(self, inputs):
        outputs_pre = self.sequential[0](inputs)
        outputs_curr = self.sequential[1](outputs_pre)
        for t in range(2, len(self.sequential)):
            outputs = self.sequential[t](outputs_pre + outputs_curr)
            outputs_pre = outputs_curr
            outputs_curr = outputs

        mu = self.mean_layer(outputs)
        sigma = self.std_layer(outputs) + self.epsilon
        return mu, sigma


class DenseNet(nn.Module):
    # TODO: remove un-used parameters

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden,
        n_layers,
        init,
        activation,
        dropout=0.0,
        norm="none",
        epsilon=1e-4,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.activation = activation
        self.init = init

        dims = (n_inputs,) + (n_hidden,) * n_layers + (n_outputs,)

        sequential = [
            nn.Sequential(
                nn.Linear(dims[0], dims[1]),
                # eval("torch.nn.init.%s(layer.weight)" % self.init),
                eval("nn.%s()" % self.activation),
                nn.Dropout(p=dropout),
            )
        ]
        for t in range(len(dims) - 2):
            sequential.append(
                nn.Sequential(
                    nn.Linear(dims[t] + dims[t + 1], dims[t + 2]),
                    # eval("torch.nn.init.%s(layer.weight)" % self.init),
                    eval("nn.%s()" % self.activation),
                    nn.Dropout(p=dropout),
                )
            )

        self.sequential = nn.Sequential(*sequential)

        self.mean_layer = nn.Linear(n_outputs, n_outputs)
        self.std_layer = nn.Sequential(nn.Linear(n_outputs, n_outputs), nn.Softplus())

        for p in self.sequential.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

        for p in self.mean_layer.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

        for p in self.std_layer.parameters():
            if p.dim() > 1:
                eval("torch.nn.init.%s(p)" % self.init)

    def forward(self, inputs):
        outputs = self.sequential[0](inputs)
        outputs_pre = inputs
        outputs_curr = outputs
        for t in range(len(self.sequential) - 1):
            outputs = self.sequential[t + 1](
                torch.cat((outputs_pre, outputs_curr), dim=-1)
            )
            outputs_pre = outputs_curr
            outputs_curr = outputs

        mu = self.mean_layer(outputs)
        sigma = self.std_layer(outputs) + self.epsilon

        return mu, sigma
