import torch
from taskchain.parameter import AutoParameterObject
from torch import nn


class JoinedNN(AutoParameterObject, nn.Module):
    def __init__(self, networks: list):
        super(JoinedNN, self).__init__()
        self.networks = networks
        self.seq = nn.Sequential(*networks)

    def forward(self, x: torch.Tensor):
        return self.seq(x)
