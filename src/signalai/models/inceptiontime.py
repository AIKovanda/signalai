import torch
import torch.nn as nn
from torch.nn import ModuleList


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(X):
    return X


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=None, bottleneck_channels=32, activation=nn.SELU(),
                 return_indices=False):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(InceptionModule, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]
        self.return_indices = return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, x):
        # step 1
        Z_bottleneck = self.bottleneck(x)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(x)
        else:
            Z_maxpool = self.max_pool(x)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))
        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=None, bottleneck_channels=32, use_residual=True,
                 activation=nn.SELU(), return_indices=False):
        super(InceptionBlock, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = InceptionModule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_2 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_3 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )

    def forward(self, x):
        if self.return_indices:
            Z, i1 = self.inception_1(x)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
        else:
            Z = self.inception_1(x)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(x)
            Z = self.activation(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z


class InceptionModuleTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 activation=nn.ReLU()):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        """
        super(InceptionModuleTranspose, self).__init__()
        self.activation = activation
        self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.conv_to_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv1d(
            in_channels=3 * bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x, indices):
        Z1 = self.conv_to_bottleneck_1(x)
        Z2 = self.conv_to_bottleneck_2(x)
        Z3 = self.conv_to_bottleneck_3(x)
        Z4 = self.conv_to_maxpool(x)

        Z = torch.cat([Z1, Z2, Z3], axis=1)
        MUP = self.max_unpool(Z4, indices)
        BN = self.bottleneck(Z)
        # another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution

        return self.activation(self.batch_norm(BN + MUP))


class InceptionBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32,
                 use_residual=True, activation=nn.ReLU()):
        super(InceptionBlockTranspose, self).__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.inception_1 = InceptionModuleTranspose(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_2 = InceptionModuleTranspose(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_3 = InceptionModuleTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=out_channels
                )
            )

    def forward(self, x, indices):
        assert len(indices) == 3
        Z = self.inception_1(x, indices[2])
        Z = self.inception_2(Z, indices[1])
        Z = self.inception_3(Z, indices[0])
        if self.use_residual:
            Z = Z + self.residual(x)
            Z = self.activation(Z)
        return Z


class InceptionTime(nn.Module):

    def __init__(self, build_config, in_channels=1, outputs=1, activation="SELU"):
        """
        InceptionTime network
        :param build_config: list of dicts
        :param in_channels: integer
        :param outputs: None or integer as a number of output classes, negative means output signals
        """
        super().__init__()
        n_filters = [in_channels] + [node.get("n_filters", 32) for node in build_config]
        kernel_sizes = [node.get("kernel_sizes", [11, 21, 41]) for node in build_config]
        bottleneck_channels = [node.get("bottleneck_channels", 32) for node in build_config]
        use_residuals = [node.get("use_residual", True) for node in build_config]
        num_of_nodes = len(kernel_sizes)
        self.outputs = outputs
        if activation == "SELU":
            activation_function = nn.SELU()
        elif activation == "Mish":
            activation_function = nn.Mish()
        elif activation == "Tanh":
            activation_function = nn.Tanh()
        else:
            raise ValueError(f"Activation {activation} unknown!")

        self.inception_blocks = ModuleList([InceptionBlock(
            in_channels=n_filters[i] if i == 0 else n_filters[i] * 4,
            n_filters=n_filters[i + 1],
            kernel_sizes=kernel_sizes[i],
            bottleneck_channels=bottleneck_channels[i],
            use_residual=use_residuals[i],
            activation=activation_function
        ) for i in range(num_of_nodes)])
        self.in_features = (1 + len(kernel_sizes[-1])) * n_filters[-1] * 1
        if self.outputs > 0:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
            self.linear1 = nn.Linear(
                in_features=self.in_features,
                out_features=outputs)
            if self.outputs in [1, 2]:
                self.out_activation = nn.Sigmoid()
            else:
                self.out_activation = nn.Softmax()
        elif self.outputs < 0:
            self.final_conv = nn.Conv1d(
                in_channels=n_filters[-1] * 4,
                out_channels=-self.outputs,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        else:
            raise ValueError(f"Outputs cannot be 0.")

    def forward(self, x):
        for block in self.inception_blocks:
            x = block(x)
        if self.outputs > 0:
            x = self.adaptive_pool(x)
            x = x.view(-1, self.in_features)
            x = self.linear1(x)
            x = self.out_activation(x)
        else:
            x = self.final_conv(x)

        return x
