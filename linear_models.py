import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class LinearRegressor(nn.Module):
    def __init__(self, n_features, n_targets=1, standardize=True):
        super(LinearRegressor, self).__init__()
        self.weight = nn.Parameter(torch.randn(n_features, n_targets))
        self.bias = nn.Parameter(torch.randn(1))
        if standardize:
            self.layer_norm = nn.LayerNorm(n_features)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        x = self.layer_norm(x)
        return torch.matmul(x, self.weight) + self.bias


class LogisticRegressor(nn.Module):
    def __init__(self, n_features, n_classes, standardize=True):
        super(LogisticRegressor, self).__init__()
        self.weight = nn.Parameter(torch.randn(n_features, n_classes))
        self.bias = nn.Parameter(torch.randn(n_classes))
        if standardize:
            self.layer_norm = nn.LayerNorm(n_features)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        x = self.layer_norm(x)
        linear = torch.matmul(x, self.weight) + self.bias
        return torch.softmax(linear, dim=1)


class MultiTargetLinearRegressor(nn.Module):
    def __init__(self, n_features, n_targets, standardize=True):
        super(MultiTargetLinearRegressor, self).__init__()
        # Initialize parameters: weight vector and bias
        self.weight = nn.Parameter(torch.randn(n_features, n_targets))
        self.bias = nn.Parameter(torch.randn(n_targets))
        if standardize:
            self.layer_norm = nn.LayerNorm(n_features)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        # Linear prediction function
        return torch.matmul(x, self.weight) + self.bias