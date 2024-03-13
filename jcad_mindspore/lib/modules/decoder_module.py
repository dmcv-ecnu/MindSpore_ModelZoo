import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .layers import *
# 假设 Conv 和 SelfAttn 已经按照前面的指示定义

class SimpleDecoder(nn.Cell):
    def __init__(self, c_in5, c_in4, c_in3, channel):
        super(SimpleDecoder, self).__init__()
        self.conv1 = Conv(channel * 3, channel, kernel_size=3)
        self.conv2 = Conv(channel, channel, kernel_size=3)
        self.conv3 = Conv(channel, 1, kernel_size=3, bn=False)

        # self.upsample = ops.ResizeBilinear()  # MindSpore 的插值函数
        self.upsample = lambda img, size: ops.interpolate(img, size=size, mode='bilinear', align_corners=True)
        # self.upsample = ops.interpolate(mode='bilinear', align_corners=True)

        self.reduce3 = Conv(c_in3, channel, kernel_size=3, relu=True)
        self.reduce4 = Conv(c_in4, channel, kernel_size=3, relu=True)
        self.reduce5 = Conv(c_in5, channel, kernel_size=3, relu=True)

    def upsample(self, img, size):
        # MindSpore 的 interpolate 需要一个 4D 张量
        return self.interpolate(img, size=size)

    def construct(self, f1, f2, f3):
        # 通道数减少
        f1 = self.reduce5(f1)
        f2 = self.reduce4(f2)
        f3 = self.reduce3(f3)

        # 解码器
        f1 = self.upsample(f1, (f3.shape[2], f3.shape[3]))
        f2 = self.upsample(f2, (f3.shape[2], f3.shape[3]))
        f3 = ops.concat((f1, f2, f3), axis=1)

        f3 = self.conv1(f3)
        f3 = self.conv2(f3)
        out = self.conv3(f3)

        return f3, out