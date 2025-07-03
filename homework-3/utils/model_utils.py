from typing import List

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int],
                 use_dropout: bool = False, dropout_p: float = 0.0,
                 use_batchnorm: bool = False):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if use_dropout and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_mlp(input_dim: int, output_dim: int, hidden_layers: List[int],
               use_dropout: bool = False, dropout_p: float = 0.0,
               use_batchnorm: bool = False) -> nn.Module:
    return MLP(input_dim, output_dim, hidden_layers, use_dropout, dropout_p, use_batchnorm)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
