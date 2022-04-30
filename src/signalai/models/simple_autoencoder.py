import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super().__init__()

        self.activ_function = nn.Mish()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=(5, 5),
            stride=(2, 2),
        )
        self.conv2 = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(5, 4),
            stride=(2, 1),
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
        )
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
        )
        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
        )
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.activ_function(self.conv1(x))
        x = self.activ_function(self.conv2(x))
        # x = self.activ_function(self.conv3(x))
        # x = self.activ_function(self.conv4(x))
        # x = self.activ_function(self.conv5(x))
        # x = self.flatten(x)
        return x


class Decoder(nn.Module):

    def __init__(self, out_channels, latent_dim):
        super().__init__()
        self.activ_function = nn.Mish()

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 31, 3))

        self.conv2 = nn.ConvTranspose2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(3, 3),
            stride=(2, 2),
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 1),
        )
        self.conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(3, 3),
            stride=2,
        )
        self.conv5 = nn.ConvTranspose2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(5, 4),
            stride=(2, 1),
        )
        self.conv6 = nn.ConvTranspose2d(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(2, 2),
        )

    def forward(self, x):
        # x = self.unflatten(x)
        # x = self.activ_function(self.conv2(x))
        # x = self.activ_function(self.conv3(x))
        # x = self.activ_function(self.conv4(x))
        x = self.activ_function(self.conv5(x))
        x = self.activ_function(self.conv6(x))
        return x


class SimpleAutoEncoder(nn.Module):

    def __init__(self, in_channels=2, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
