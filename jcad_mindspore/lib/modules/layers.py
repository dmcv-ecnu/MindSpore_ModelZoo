import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter

class Conv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding=0, pad_mode="same",bias=False, bn=True, relu=False):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, pad_mode=pad_mode, padding=padding, dilation=dilation, group=groups, has_bias=bias)
        self.reset_parameters()

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        # MindSpore 的权重初始化方法可能有所不同，需要根据情况调整
        pass

class SelfAttn(nn.Cell):
    def __init__(self, in_channels, mode='hw'):
        super(SelfAttn, self).__init__()

        self.mode = mode
        self.query_conv = Conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(Tensor([0.0], mindspore.float32))
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        batch_size, channel, height, width = x.shape

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).transpose(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = ops.BatchMatMul(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = ops.BatchMatMul(projected_value, attention.transpose(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out