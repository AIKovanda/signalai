import torch
import torch.nn as nn
from taskchain.parameter import AutoParameterObject
from torch.nn import ModuleList

from models.tools import get_activation


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(x):
    return x


class InceptionModule(AutoParameterObject, nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=None, bottleneck_channels=32, activation=nn.SELU(),
                 use_batch_norm=True, return_indices=False):
        """
        in_channels				Number of input channels (input features)
        n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is necessary because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck won't be used if number of in_channels is equal to 1.
        activation				Activation function for output tensor (nn.ReLU()).
        return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(InceptionModule, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = bottleneck_channels
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.return_indices = return_indices

        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False,
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
            bias=False,
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False,
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False,
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
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
        if self.use_batch_norm:
            Z = self.batch_norm(Z)

        Z = self.activation(Z)
        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=None, bottleneck_channels=32, use_residual=True,
                 use_batch_norm=True, activation=nn.SELU(), return_indices=False):
        super(InceptionBlock, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]

        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.inception_1 = InceptionModule(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            use_batch_norm=self.use_batch_norm,
            return_indices=return_indices,
        )
        self.inception_2 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            use_batch_norm=self.use_batch_norm,
            return_indices=return_indices,
        )
        self.inception_3 = InceptionModule(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            use_batch_norm=self.use_batch_norm,
            return_indices=return_indices,
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters,
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
        in_channels				Number of input channels (input features)
        n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        activation				Activation function for output tensor (nn.ReLU()).
        """
        super(InceptionModuleTranspose, self).__init__()
        self.activation = activation
        self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False,
        )
        self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False,
        )
        self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False,
        )
        self.conv_to_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv1d(
            in_channels=3 * bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
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
                    padding=0,
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


class InceptionTime(AutoParameterObject, nn.Module):

    def __init__(self, build_config, out_activation=None, in_channels=1, outputs=1):
        """
        InceptionTime network
        :param build_config: list of dicts
        :param in_channels: integer
        :param outputs: None or integer as a number of output classes, negative means output signals
        """
        super().__init__()

        block_list = []
        self.poolings = []
        self.build_config = build_config
        self.in_channels = in_channels
        self.outputs = outputs

        last_kernel_size = None
        last_n_filters = in_channels
        for i, node in enumerate(build_config):
            last_kernel_size = node.get("kernel_sizes", [11, 21, 41])
            current_n_filters = node.get("n_filters", 32)

            bottleneck_channels = [node.get("bottleneck_channels", 32) for node in build_config]
            use_residuals = [node.get("use_residual", True) for node in build_config]

            activation_function = get_activation(node.get('activation', 'SELU'))
            block_list.append(InceptionBlock(
                in_channels=last_n_filters if i == 0 else last_n_filters * 4,
                n_filters=current_n_filters,
                kernel_sizes=last_kernel_size,
                bottleneck_channels=bottleneck_channels[i],
                use_residual=use_residuals[i],
                use_batch_norm=node.get("use_batch_norm", True),
                activation=activation_function
            ))

            last_n_filters = current_n_filters

            # pooling
            pooling_size = node.get("pooling_size", None)
            pooling_type = node.get("pooling_type", 'max')

            if pooling_size is None or pooling_type is None:
                self.poolings.append(lambda x: x)
            else:
                if pooling_type == 'max':
                    self.poolings.append(nn.MaxPool1d(pooling_size))
                else:
                    self.poolings.append(nn.AvgPool1d(pooling_size))

        self.out_activation = out_activation
        self.out_activation_function = get_activation(self.out_activation)

        self.inception_blocks = ModuleList(block_list)

        self.in_features = (1 + len(last_kernel_size)) * last_n_filters * 1
        if self.outputs > 0:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
            self.linear1 = nn.Linear(
                in_features=self.in_features,
                out_features=outputs)

        elif self.outputs < 0:
            self.final_conv = nn.Conv1d(
                in_channels=last_n_filters * 4,
                out_channels=-self.outputs,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            raise ValueError(f"Outputs cannot be 0.")

    def forward(self, x):
        for block, pooling in zip(self.inception_blocks, self.poolings):
            x = block(x)
            x = pooling(x)

        if self.outputs > 0:
            x = self.adaptive_pool(x)
            x = x.view(-1, self.in_features)
            x = self.linear1(x)
        else:
            x = self.final_conv(x)

        return self.out_activation_function(x)

    def weight_reset(self):
        return type(self)(
            build_config=self.build_config,
            out_activation=self.out_activation,
            in_channels=self.in_channels,
            outputs=self.outputs,
        )

