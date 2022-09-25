#! /usr/bin/env python
# 用于消融研究阶段的学生网络

import numpy as np
import mindspore as ms
from mindspore import Parameter
from mindspore import nn, ops
from mindspore.common.initializer import TruncatedNormal
from model.common import RgbNormal
import mindspore.ops as F

def make_conv2d(in_channels, out_channels, kernel_size, has_bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        pad_mode="same", has_bias=has_bias, weight_init=TruncatedNormal(0.02))

class ResBlock(nn.Cell):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, has_bias=True, res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = make_conv2d(in_channels, in_channels, kernel_size, has_bias)
        self.relu = nn.ReLU()
        self.conv2 = make_conv2d(in_channels, out_channels, kernel_size, has_bias)
        self.res_scale = res_scale

    def construct(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = res * self.res_scale
        x = x + res
        return x

class PixelShuffle(nn.Cell):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.upper = ops.DepthToSpace(self.upscale_factor)

    def construct(self, x):
        return self.upper(x)

    def extend_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

def UpsamplerBlockList(upscale_factor:int, n_feats:int, has_bias=True):
    if upscale_factor == 1:
        return []
    allow_sub_upscale_factor = [2, 3, None]
    for sub in allow_sub_upscale_factor:
        if sub is None:
            raise NotImplementedError(
                f"Only support \"scales\" that can be divisibled by {allow_sub_upscale_factor[:-1]}")
        if upscale_factor % sub == 0:
            break
    sub_block_list = [
        make_conv2d(n_feats, sub*sub*n_feats, 3, has_bias),
        PixelShuffle(sub),
    ]
    return sub_block_list + UpsamplerBlockList(upscale_factor // sub, n_feats, has_bias)


class Upsampler(nn.Cell):

    def __init__(self, scale, n_feats, has_bias=True):
        super(Upsampler, self).__init__()
        up = UpsamplerBlockList(scale, n_feats, has_bias)
        self.up = nn.SequentialCell(*up)

    def construct(self, x):
        x = self.up(x)
        return x


class EDSR(nn.Cell):
    """
    EDSR network
    """
    def __init__(self, scale=3, n_feats=64, kernel_size=3, n_resblocks=8,
                 n_colors=3,
                 res_scale=0.1,
                 rgb_range=255,
                 rgb_mean=(0.0, 0.0, 0.0),
                 rgb_std=(1.0, 1.0, 1.0)):
        super(EDSR, self).__init__()

        self.norm = RgbNormal(rgb_range, rgb_mean, rgb_std, inverse=False)
        self.de_norm = RgbNormal(rgb_range, rgb_mean, rgb_std, inverse=True)

        m_head = [make_conv2d(n_colors, n_feats, kernel_size)]

        m_body1 = [
            ResBlock(n_feats, n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks // 4)
            
        ]
        m_body1.append(make_conv2d(n_feats, n_feats, kernel_size))

        m_body2 = [
            ResBlock(n_feats, n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks // 4)
            
        ]
        m_body2.append(make_conv2d(n_feats, n_feats, kernel_size))

        m_body3 = [
            ResBlock(n_feats, n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks // 4)
            
        ]
        m_body3.append(make_conv2d(n_feats, n_feats, kernel_size))

        m_body4 = [
            ResBlock(n_feats, n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks // 4)
            
        ]
        m_body4.append(make_conv2d(n_feats, n_feats, kernel_size))

        m_tail1 = [
            Upsampler(scale, n_feats),
            make_conv2d(n_feats, n_colors, kernel_size)
        ]
        m_tail2 = [
            Upsampler(scale, n_feats),
            make_conv2d(n_feats, n_colors, kernel_size)
        ]
        m_tail3 = [
            Upsampler(scale, n_feats),
            make_conv2d(n_feats, n_colors, kernel_size)
        ]

        m_tail4 = [
            Upsampler(scale, n_feats),
            make_conv2d(n_feats, n_colors, kernel_size)
        ]

        m_tail = [
            Upsampler(scale, n_feats),
            make_conv2d(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.SequentialCell(m_head)
        self.body1 = nn.SequentialCell(m_body1)
        self.body2 = nn.SequentialCell(m_body2)
        self.body3 = nn.SequentialCell(m_body3)
        self.body4 = nn.SequentialCell(m_body4)
        self.tail1 = nn.SequentialCell(m_tail1)
        self.tail2 = nn.SequentialCell(m_tail2)
        self.tail3 = nn.SequentialCell(m_tail3)
        self.tail4 = nn.SequentialCell(m_tail4)
        self.tail = nn.SequentialCell(m_tail)

    def construct(self, x, predict=False):
        x = self.norm(x)
        x = self.head(x)

        res_0 = x

        # output1
        res_1 = self.body1(res_0)
        res1 = x + res_1
        out1 = self.tail1(res1)
        out1 = self.de_norm(out1)

        # output2
        res_2 = self.body2(res_1)
        res2 = x + res_2
        out2 = self.tail2(res2)
        out2 = self.de_norm(out2)

        # output3
        res_3 = self.body3(res_2)
        res3 = x + res_3
        out3 = self.tail3(res3)
        out3 = self.de_norm(out3)

        # output4
        res_4 = self.body4(res_3)
        res4 = x + res_4
        out4 = self.tail4(res4)
        out4 = self.de_norm(out4)

        # 推理
        if predict == True:
            return out4

        return [out1, out2, out3, out4], [res_0, res_1, res_2, res_3, res_4]

