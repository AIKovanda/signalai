from torch import nn as nn


def get_activation(activation=None):
    if activation is None:
        return lambda x: x

    activation = activation.lower()
    if activation == "selu":
        return nn.SELU()
    elif activation == "mish":
        return nn.Mish()
    elif activation == "softmax":
        return nn.Softmax()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Activation {activation} unknown!")
