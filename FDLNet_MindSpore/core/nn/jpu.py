"""Joint Pyramid Upsampling"""
from mindspore import nn
from mindspore import ops

__all__ = ['JPU']


class SeparableConv2d(nn.Cell):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride, pad_mode='pad', padding=padding, dilation=dilation,
                 group=inplanes, has_bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, has_bias=bias)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# copy from: https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py
class JPU(nn.Cell):
    def __init__(self, in_channels, width=512, norm_layer=nn.BatchNorm2d, **kwargs):
        super(JPU, self).__init__()

        self.conv5 = nn.SequentialCell(
            nn.Conv2d(in_channels[-1], width, 3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(width),
            nn.ReLU())
        self.conv4 = nn.SequentialCell(
            nn.Conv2d(in_channels[-2], width, 3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(width),
            nn.ReLU())
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels[-3], width, 3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(width),
            nn.ReLU())

        self.dilation1 = nn.SequentialCell(
            SeparableConv2d(3 * width, width, 3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU())
        self.dilation2 = nn.SequentialCell(
            SeparableConv2d(3 * width, width, 3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU())
        self.dilation3 = nn.SequentialCell(
            SeparableConv2d(3 * width, width, 3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU())
        self.dilation4 = nn.SequentialCell(
            SeparableConv2d(3 * width, width, 3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU())

    def construct(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        size = feats[-1].shape[2:]
        feats[-2] = ops.interpolate(feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = ops.interpolate(feats[-3], size, mode='bilinear', align_corners=True)
        feat = ops.cat(feats, axis=1)
        feat = ops.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         axis=1)

        return inputs[0], inputs[1], inputs[2], feat
