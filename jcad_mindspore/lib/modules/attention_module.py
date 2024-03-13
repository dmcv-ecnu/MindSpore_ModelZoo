import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal
from .layers import *

class simple_attention(nn.Cell):
    def __init__(self, in_channel, channel, kernel_size=3, size=48):
        super(simple_attention, self).__init__()
        self.relu = nn.ReLU()
        self.conv_in = Conv(in_channel, channel, 1)
        self.conv1 = Conv(in_channel, channel, kernel_size, dilation=1, relu=True)
        self.conv2 = Conv(in_channel, channel, kernel_size, dilation=2, relu=True)
        self.conv3 = Conv(in_channel, channel, kernel_size, dilation=3, relu=True)
        self.conv4 = Conv(in_channel, channel, kernel_size, dilation=4, relu=True)
        self.concat = Conv(4 * channel, channel, kernel_size, relu=True)

        # self.res = ops.ResizeBilinear()
        self.res = lambda x, size: ops.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.cont_conv1 = Conv(channel, channel, kernel_size)
        self.fc1 = nn.Dense(size * size * channel, 1024)
        self.fc2 = nn.Dense(1024, 1024)
        self.predict = nn.Dense(1024, 3)

        self.conv_mid1 = Conv(channel, channel, kernel_size)
        self.conv_mid2 = Conv(channel, channel, kernel_size)
        self.conv_mid3 = Conv(channel, channel, kernel_size)

        self.conv_out = Conv(channel, 1, 1)

    def construct(self, x, map):
        B, _, H, W = x.shape
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        concat = self.concat(ops.Concat(1)((conv1, conv2, conv3, conv4)))
        x = self.conv_in(x)
        x = concat + x
        # mask attn
        map = self.res(map, (x.shape[2], x.shape[3]))
        amap = ops.Sigmoid()(map)
        amap = amap.expand_as(x)
        x = ops.Mul()(amap, x)  # 8*256*48*48

        cont_conv1 = self.cont_conv1(x)
        cont_down = self.res(cont_conv1, (H // 2, W // 2))
        f_re = cont_down.view(cont_down.shape[0], -1)
        f_re = self.relu(self.fc1(f_re))
        f_re = self.relu(self.fc2(f_re))
        f_re = self.predict(f_re)

        conv_mid1 = self.conv_mid1(x)
        conv_mid1 = conv_mid1 + cont_conv1
        conv_mid2 = self.conv_mid2(conv_mid1)
        conv_mid3 = self.conv_mid3(conv_mid2)

        out = self.conv_out(conv_mid3)
        out = out + map

        return x, out, f_re