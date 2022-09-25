import mindspore as ms
import math
import mindspore.nn as nn
import mindspore.ops as ops


class BasicConv(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, has_bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.99, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, *inputs):
        x = self.conv(inputs)
        if self.bn is not None:
            x = self.bn(inputs)
        if self.relu is not None:
            x = self.relu(inputs)
        return x


class Flatten(nn.Cell):
    def construct(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Cell):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.SequentialCell(
            Flatten(),
            nn.Dense(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Dense(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def construct(self, *inputs, **kwargs):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = ops.AvgPool((inputs, inputs.size(2), inputs.size(3)), stride=(inputs.size(2), inputs.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = nn.MaxPool2d(inputs, (inputs.size(2), inputs.size(3)), stride=(inputs.size(2), inputs.size(3)), pad_mode="valid")
                channel_att_raw = self.mlp(max_pool)
            # elif pool_type == 'lp':

            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(inputs)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        expand_dims = ops.ExpandDims()
        scale = nn.Sigmoid(channel_att_sum).expand_dims(channel_att_sum, 2).expand_dims(channel_att_sum, 3).expand_as(x)

        return inputs * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    argmax = ops.ArgMaxWithValue()
    _, s = argmax(tensor_flatten, axis=2, keep_dims=True)
    outputs = s + (tensor_flatten - s).ops.Exp().ReduceSum(dim=2, keepdim=True).Log()
    return outputs

class ChannelPool(nn.Cell):
    def construct(self, *inputs, **kwargs):
        return ops.Concat((ops.ArgMaxWithValue(inputs, 1)[0].ExpandDims(1), ops.ReduceMean(inputs, 1).ExpandDims(1)), dim=1)


class SpatialGate(nn.Cell):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def construct(self, *inputs, **kwargs):
        x_compress = self.compress(inputs)
        x_out = self.spatial(x_compress)
        scale = nn.Sigmoid(x_out)  # broadcasting
        return inputs * scale


class CBAM(nn.Cell):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def construct(self, *inputs, **kwargs):
        x_out = self.ChannelGate(inputs)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
