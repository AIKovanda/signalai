import torch
from torch.nn import ModuleList

from signalai.models.tools import get_activation
from taskchain.parameter import AutoParameterObject
from torch import nn


class Spec2Map(AutoParameterObject, nn.Module):

    def __init__(self, convs_2d: list, convs_1d: list, activation=None, out_activation=None, in_channels=1,
                 kernel_1d=5, vertical_channels=513, outputs=85):

        super().__init__()
        self.convs_2d = convs_2d
        self.convs_1d = convs_1d
        self.activation = activation
        self.out_activation = out_activation
        self.kernel_1d = kernel_1d
        self.in_channels = in_channels
        self.vertical_channels = vertical_channels
        self.outputs = outputs

        # part of the 2D convs
        channels = [in_channels, *convs_2d, 1]

        self.seq_convs2d = ModuleList([
            nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=5,
                padding='same',
                bias=True,
            ) for i in range(len(channels) - 1)
        ])

        # part of the 1D convs
        channels = [vertical_channels, *convs_1d, outputs]
        self.seq_convs1d = ModuleList([
            nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_1d,
                padding='same',
                bias=True,
            ) for i in range(len(channels) - 1)
        ])

        self.activation_function = get_activation(activation)
        self.out_activation_function = get_activation(out_activation)

    def forward(self, x):
        for conv in self.seq_convs2d:
            x = self.activation_function(conv(x))

        z = torch.squeeze(x, 1)

        for conv in self.seq_convs1d[:-1]:
            z = self.activation_function(conv(z))

        z = self.seq_convs1d[-1](z)
        return self.out_activation_function(z)

    def weight_reset(self):
        return type(self)(
            convs_2d=self.convs_2d,
            convs_1d=self.convs_1d,
            activation=self.activation,
            out_activation=self.out_activation,
            kernel_1d=self.kernel_1d,
            in_channels=self.in_channels,
            vertical_channels=self.vertical_channels,
            outputs=self.outputs,
        )
