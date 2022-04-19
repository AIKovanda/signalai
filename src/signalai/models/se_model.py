from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from taskchain.parameter import AutoParameterObject
from torch.nn import ModuleList, Sequential

from signalai.models.tools import get_activation


class Signal2TimeFreq(nn.Module):
    """
    X must be divisible by the smallest kernel. All kernels must be divisible by each other.
    """

    def __init__(self, kernel_sizes: List[int], output_channels=256):
        super(Signal2TimeFreq, self).__init__()
        self.kernel_sizes = sorted(kernel_sizes)
        smallest_kernel = self.kernel_sizes[0]
        self.paddings = [
            int(smallest_kernel * (2 ** (i / smallest_kernel - 1) - 1))  # sum 0 + 1 + 2 + 4 etc.
            for i in self.kernel_sizes
        ]
        self.output_channels = output_channels
        self.time_freq_convolutions = ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=smallest_kernel,
                padding=0,
                bias=False,
            )
            for i, kernel_size in enumerate(self.kernel_sizes)
        ])

    def forward(self, x):
        features = [
            conv1d(F.pad(x, (0, self.paddings[i])))
            for i, conv1d in enumerate(self.time_freq_convolutions)
        ]
        concatenated = torch.cat([f.unsqueeze(1) for f in features], 1)
        return concatenated


class TimeFreq2Signal(nn.Module):
    """
    X must be divisible by the smallest kernel. All kernels must be divisible by each other.
    """

    def __init__(self, in_channels: int, stride: int, kernel_sizes: List[int], output_channels=1):
        super(TimeFreq2Signal, self).__init__()
        self.kernel_sizes = sorted(kernel_sizes)
        self.stride = stride
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.time_freq_convolutions = ModuleList([
            nn.ConvTranspose1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=kernel_size,
                groups=self.in_channels,
                stride=self.stride,
                padding=0,
                bias=False,
            )
            for i, kernel_size in enumerate(self.kernel_sizes)
        ])
        self.join_conv = nn.Conv1d(
            in_channels=self.in_channels * len(self.kernel_sizes),
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        features = [
            conv_transpose1d(x[:, i])
            for i, conv_transpose1d in enumerate(self.time_freq_convolutions)
        ]
        smallest_len = min([f.shape[-1] for f in features])
        signalized = torch.cat([f[..., :smallest_len] for f in features], axis=1)

        return self.join_conv(signalized)


class TimeFreq2TimeFreqSimple(nn.Module):
    def __init__(self, in_channels, kernels, output_channels, activation_function=None):
        super(TimeFreq2TimeFreqSimple, self).__init__()
        self.in_channels = in_channels
        self.kernels = kernels
        self.output_channels = output_channels
        if activation_function is not None:
            self.activation_function = activation_function
        else:
            self.activation_function = lambda x: x

        self.con2D = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.kernels,
            kernel_size=7,
            padding='same',
            bias=True,
        )
        self.con2D2 = nn.Conv2d(
            in_channels=self.kernels,
            out_channels=self.kernels,
            kernel_size=7,
            padding='same',
            bias=True,
        )
        self.join_conv = nn.Conv2d(
            in_channels=self.kernels,
            out_channels=self.output_channels,
            kernel_size=1,
            padding='same',
            bias=True,
        )

    def forward(self, x):
        transformed = self.activation_function(self.con2D(x))
        transformed = self.activation_function(self.con2D2(transformed))
        return self.activation_function(self.join_conv(transformed))


class TimeFreq2TimeFreqResNeXtModule(nn.Module):
    def __init__(self, in_channels, kernels, output_channels, activation_function=None, inner_out_channels=4, residual=True):
        super(TimeFreq2TimeFreqResNeXtModule, self).__init__()
        self.in_channels = in_channels
        self.kernels = kernels
        self.output_channels = output_channels
        self.inner_out_channels = inner_out_channels
        self.residual = residual

        self.activation_function = activation_function
        if self.activation_function is None:
            self.activation_function = lambda x: x

        self.processing = ModuleList([Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inner_out_channels,
                kernel_size=1,
                padding='same',
                bias=True,
            ), self.activation_function,
            nn.Conv2d(
                in_channels=self.inner_out_channels,
                out_channels=self.inner_out_channels,
                kernel_size=3,
                padding='same',
                bias=True,
            ), self.activation_function,
            nn.Conv2d(
                in_channels=self.inner_out_channels,
                out_channels=self.output_channels,
                kernel_size=1,
                padding='same',
                bias=True,
            )
        ) for _ in range(32)
        ])
        # self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1)
        # self.max_pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1)
        if not self.residual:
            self.join_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.output_channels,
                kernel_size=1,
                padding='same',
                bias=True,
            )

    def forward(self, x):
        processed = [module(x) for module in self.processing]
        if self.residual:
            processed.append(x)
        else:
            processed.append(self.join_conv(x))  # todo: batchnorm

        processed = self.activation_function(torch.stack(processed, dim=0).sum(dim=0))
        return processed


class SpecResNeXt(nn.Module):
    def __init__(self, in_channels=256, processing_kernels=128, output_channels=1, attention=False,
                 activation='SELU', inner_out_channels=4, residual=(False,)):

        super(SpecResNeXt, self).__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.inner_out_channels = inner_out_channels
        self.residual = np.array(residual, dtype=bool)
        self.attention = attention

        if activation is None:
            self.activation_function = lambda x: x
        else:
            self.activation_function = get_activation(activation)

        # First and second columns represent input and output channels, respectively. Each row stands for a module.
        in_out = np.ones((len(residual), 2), dtype=int) * self.inner_out_channels
        where_false = np.where(~self.residual)[0]

        if len(where_false) < 2 and self.in_channels != self.output_channels:
            raise ValueError("Residual modules must have the same input and output shape, "
                             "there is no way how to build this network.")

        in_out[:where_false[0]+1, 0] = self.in_channels
        in_out[where_false[-1]+1:, 0] = self.output_channels

        in_out[:where_false[0], 1] = self.in_channels
        in_out[where_false[-1]:, 1] = self.output_channels

        self.processing = Sequential(*[
            TimeFreq2TimeFreqResNeXtModule(
                in_channels=in_out[i, 0], kernels=processing_kernels, output_channels=in_out[i, 1],
                activation_function=self.activation_function, inner_out_channels=inner_out_channels, residual=res,
            ) for i, res in enumerate(self.residual)
        ])

        if self.attention:
            self.attention_conv = nn.Conv2d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=5,
                padding='same',
                bias=True,
            )

    def forward(self, x):
        print(f'Input tensor has a shape of {x.shape}')
        processed = self.processing(x)

        if self.attention:
            processed = processed * self.attention_conv(processed)

        return self.activation_function(processed)


class SEModel(AutoParameterObject, nn.Module):
    def __init__(self, n_filters=256, processing_kernels=128, kernel_sizes=None, activation='SELU',
                 output_channels=1, inner_out_channels=4, attention=True, output_separately=False,
                 use_exception=True):
        super(SEModel, self).__init__()
        self.n_filters = n_filters
        self.processing_kernels = processing_kernels
        self.output_channels = output_channels
        self.attention = attention
        self.activation = activation
        self.inner_out_channels = inner_out_channels
        self.output_separately = output_separately
        self.use_exception = use_exception
        
        if kernel_sizes is None:
            self.kernel_sizes = [128, 256]
        else:
            self.kernel_sizes = kernel_sizes

        if activation is None:
            self.activation_function = lambda x: x
        else:
            self.activation_function = get_activation(activation)

        self.signal2time_freq = Signal2TimeFreq(kernel_sizes=self.kernel_sizes, output_channels=self.n_filters)

        if self.use_exception:
            self.processing = Sequential(
                TimeFreq2TimeFreqResNeXtModule(
                    in_channels=len(self.kernel_sizes), kernels=processing_kernels, output_channels=len(self.kernel_sizes),
                    activation_function=self.activation_function, inner_out_channels=inner_out_channels, residual=True,
                ),
                TimeFreq2TimeFreqResNeXtModule(
                    in_channels=len(self.kernel_sizes), kernels=processing_kernels, output_channels=inner_out_channels,
                    activation_function=self.activation_function, inner_out_channels=inner_out_channels, residual=False,
                ),
                TimeFreq2TimeFreqResNeXtModule(
                    in_channels=inner_out_channels, kernels=processing_kernels, output_channels=len(self.kernel_sizes),
                    activation_function=self.activation_function, inner_out_channels=len(self.kernel_sizes), residual=False,
                ),
                TimeFreq2TimeFreqResNeXtModule(
                    in_channels=len(self.kernel_sizes), kernels=processing_kernels, output_channels=len(self.kernel_sizes),
                    activation_function=self.activation_function, inner_out_channels=inner_out_channels, residual=True
                ),
            )

        else:
            self.processing = TimeFreq2TimeFreqSimple(
                in_channels=len(self.kernel_sizes), kernels=processing_kernels, output_channels=len(self.kernel_sizes),
                activation_function=self.activation_function,
            )

        if self.output_separately:
            self.time_freq2signal = ModuleList([
                TimeFreq2Signal(
                    in_channels=self.n_filters, stride=min(self.kernel_sizes), kernel_sizes=self.kernel_sizes,
                    output_channels=1,
                )
                for _ in range(output_channels)
            ])
        else:
            self.time_freq2signal = TimeFreq2Signal(
                in_channels=self.n_filters, stride=min(self.kernel_sizes), kernel_sizes=self.kernel_sizes,
                output_channels=output_channels,
            )

        if self.attention:
            self.attention_conv = self.con2D = nn.Conv2d(
                in_channels=len(self.kernel_sizes),
                out_channels=len(self.kernel_sizes),
                kernel_size=5,
                padding='same',
                bias=True,
            )

    def forward(self, x):
        print(f'Input tensor has a shape of {x.shape}')
        time_freq = self.signal2time_freq(x)
        print(f'Processed tensor has a shape of {time_freq.shape}')
        processed = self.processing(time_freq)

        if self.attention:
            processed = processed * self.attention_conv(processed)

        if self.output_separately:
            single_signals = [conv(processed) for conv in self.time_freq2signal]
            return torch.cat(single_signals, axis=1)
        
        return self.time_freq2signal(processed)
