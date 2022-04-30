import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, convs):
        super().__init__()

        self.activ_function = nn.Mish()
        self.convs = nn.Sequential(*[nn.Conv1d(**conv) for conv in convs])

    def forward(self, x):
        for conv in self.convs:
            x = self.activ_function(conv(x))
        return x


class Decoder(nn.Module):

    def __init__(self, decoder_convs):
        super().__init__()
        self.activ_function = nn.Mish()
        self.convs = nn.Sequential(*[nn.ConvTranspose1d(**conv) for conv in decoder_convs])

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = self.activ_function(conv(x))
        x = self.convs[-1](x)
        return x


class Simple1DAutoEncoder(nn.Module):

    def __init__(self, encoder_convs, decoder_convs):
        super().__init__()
        self.encoder = Encoder(encoder_convs)
        self.decoder = Decoder(decoder_convs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
