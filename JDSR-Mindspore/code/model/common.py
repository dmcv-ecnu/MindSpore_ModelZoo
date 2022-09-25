#! /usr/bin/env python

from mindspore import Parameter
import mindspore as ms
import mindspore.nn as nn
import numpy as np

def make_default_conv(in_channels, out_channels, kernel_size, has_bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, pad_mode="same",
        has_bias=has_bias) # padding为kernel的一半即为same，pytorch不支持same

class ResBlock(nn.Cell): # 残差块 移植实现
    def __init__(
        self, conv, n_feat, kernel_size,
        has_bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, has_bias=has_bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.SequentialCell(m)
        self.res_scale = res_scale

    def construct(self, x):
        res = self.body(x) * self.res_scale
        res += x

        return res

class PixelShuffle(nn.Cell): # 像素重组
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.upper = ms.ops.DepthToSpace(self.upscale_factor)

    def construct(self, x):
        return self.upper(x)

    def extend_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

class RgbNormal(nn.Cell): # 数据规范化
    def __init__(self, rgb_range, rgb_mean, rgb_std, inverse=False):
        super(RgbNormal, self).__init__()
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.inverse = inverse
        std = np.array(self.rgb_std, dtype=np.float32)
        mean = np.array(self.rgb_mean, dtype=np.float32)
        if not inverse:
            # y: (x / rgb_range - mean) / std <=> x * (1.0 / rgb_range / std) + (-mean) / std
            weight = (1.0 / self.rgb_range / std).reshape((1, -1, 1, 1))
            bias = (-mean / std).reshape((1, -1, 1, 1))
        else:
            # x: (y * std + mean) * rgb_range <=> y * (std * rgb_range) + mean * rgb_range
            weight = (self.rgb_range * std).reshape((1, -1, 1, 1))
            bias = (mean * rgb_range).reshape((1, -1, 1, 1))
        self.weight = Parameter(name='weight', default_input=weight, requires_grad=False)
        self.bias = Parameter(name='bias', default_input=bias, requires_grad=False)

    def construct(self, x):
        return x * self.weight + self.bias

    def extend_repr(self):
        s = 'rgb_range={}, rgb_mean={}, rgb_std={}, inverse = {}' \
            .format(
                self.rgb_range,
                self.rgb_mean,
                self.rgb_std,
                self.inverse,
            )
        return s