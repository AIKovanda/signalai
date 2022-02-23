from signalai.models.tools import get_activation
from taskchain.parameter import AutoParameterObject
from torch import nn


build_config = {

}



class SpectroMap(AutoParameterObject, nn.Module):
    def __init__(self, build_config, out_activation=None, in_channels=1, outputs=1):
        """
        InceptionTime network
        :param build_config: list of dicts
        :param in_channels: integer
        :param outputs: None or integer as a number of output classes, negative means output signals
        """
        super().__init__()

        module_list = []
        self.build_config = build_config
        self.in_channels = in_channels
        self.outputs = outputs

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

