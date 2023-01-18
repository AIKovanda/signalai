from torch import nn


class Encoder(nn.Module):

    def __init__(self, convs, dim=2):
        super().__init__()

        self.activ_function = nn.Mish()
        if dim == 1:
            self.convs = nn.Sequential(*[nn.Conv1d(**conv) for conv in convs])
        elif dim == 2:
            self.convs = nn.Sequential(*[nn.Conv2d(**conv) for conv in convs])
        else:
            raise ValueError(f'Dim {dim} not supported!')

    def forward(self, x):
        for conv in self.convs:
            x = self.activ_function(conv(x))
        return x


class Decoder(nn.Module):

    def __init__(self, decoder_convs, dim=2):
        super().__init__()
        self.activ_function = nn.Mish()
        if dim == 1:
            self.convs = nn.Sequential(*[nn.ConvTranspose1d(**conv) for conv in decoder_convs])
        elif dim == 2:
            self.convs = nn.Sequential(*[nn.ConvTranspose2d(**conv) for conv in decoder_convs])
        else:
            raise ValueError(f'Dim {dim} not supported!')

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = self.activ_function(conv(x))
        x = self.convs[-1](x)
        return x


class SimpleAutoEncoder(nn.Module):

    def __init__(self, encoder_convs, decoder_convs, dim=2):
        super().__init__()
        self.encoder = Encoder(encoder_convs, dim=dim)
        self.decoder = Decoder(decoder_convs, dim=dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
