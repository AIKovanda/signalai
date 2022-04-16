import torch
import torch.nn.functional as F
from torch import nn


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError('h1_shape[3] must be greater than h2_shape[3]')

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


class Conv2DBNActiv(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin, nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False),
            nn.BatchNorm2d(nout),
            activ()
        )

    def __call__(self, x):
        return self.conv(x)


class SeperableConv2DBNActiv(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin, nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False),
            nn.Conv2d(
                nin, nout,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(nout),
            activ()
        )

    def __call__(self, x):
        return self.conv(x)


class Encoder(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        skip = self.conv1(x)
        h = self.conv2(skip)

        return h, skip


class Decoder(nn.Module):

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super(Decoder, self).__init__()
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h


class ASPPModule(nn.Module):

    def __init__(self, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        )
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = SeperableConv2DBNActiv(
            nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 5, nout, 1, 1, 0, activ=activ),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        bottle = self.bottleneck(out)
        return bottle


class BaseASPPNet(nn.Module):

    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = Encoder(nin, ch, 3, 2, 1)
        self.enc2 = Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = ASPPModule(ch * 8, ch * 16, dilations)

        self.dec4 = Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):

    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(2, 16)
        self.stg1_high_band_net = BaseASPPNet(2, 16)

        self.stg2_bridge = Conv2DBNActiv(18, 8, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(8, 16)

        self.stg3_bridge = Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(16, 32)

        self.out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(16, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(16, 2, 1, bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, x):
        mix = x.detach()
        x = x.clone()

        x = x[:, :, :self.max_bin]

        bandw = x.size()[2] // 2
        aux1 = torch.cat([
            self.stg1_low_band_net(x[:, :, :bandw]),
            self.stg1_high_band_net(x[:, :, bandw:])
        ], dim=2)

        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate')

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode='replicate')
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode='replicate')
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            return mask * mix

    def predict(self, x_mag):
        h = self.forward(x_mag)

        if self.offset > 0:
            h = h[:, :, :, self.offset:-self.offset]
            assert h.size()[3] > 0

        return h
