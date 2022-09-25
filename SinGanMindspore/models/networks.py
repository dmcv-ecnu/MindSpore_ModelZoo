"""SinGAN-Dehaze network."""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import initializer as init
import numpy as np


###############################################################################
# Helper Functions
###############################################################################
def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    filt = ms.Tensor(a[:, None] * a[None, :])
    filt = filt / ops.ReduceSum()(filt)
    return filt


class Downsample(nn.Cell):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def construct(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return ops.Conv2D(stride=self.stride, group=inp.shape[1])(self.pad(inp), self.filt)


class Upsample2(nn.Cell):
    def __init__(self, scale_factor, mode='bilinear'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def construct(self, x):
        # return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        # 只实现用bilinear模式进行采样，其他计算模式需使用小算子拼接
        return nn.ResizeBilinear()(x, scale_factor=self.factor)


class Upsample(nn.Cell):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def construct(self, inp):
        ret_val = nn.Conv2dTranspose(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, group=inp.shape[1])[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    # 指定填充模式。取值为”CONSTANT”，”REFLECT”，”SYMMETRIC”。(找不到对应)
    """
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    """
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.Pad(mode="REFLECT")
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.Pad(mode="SYMMETRIC")
    elif pad_type == 'zero':
        PadLayer = nn.Pad(mode="CONSTANT")
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class ConvNormReLU(nn.Cell):
    """
    Convolution fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.
    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".
        use_relu (bool): Use relu or not. Default: True.
        padding (int): Pad size, if it is None, it will calculate by kernel_size. Default: None.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvNormReLU, self).__init__()
        norm = nn.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = nn.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad',
                             has_bias=has_bias, padding=padding)
            layers = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layers = [pad, conv, norm]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class ConvTransposeNormReLU(nn.Cell):
    """
    ConvTranspose2d fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.
    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
                        Default: "CONSTANT".
        use_relu (bool): use relu or not. Default: True.
        padding (int): pad size, if it is None, it will calculate by kernel_size. Default: None.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvTransposeNormReLU, self).__init__()
        conv = nn.Conv2dTranspose(in_planes, out_planes, kernel_size, stride=stride, pad_mode='same')
        norm = nn.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = nn.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = nn.Conv2dTranspose(in_planes, out_planes, kernel_size, stride, pad_mode='same', has_bias=has_bias)
            layers = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            conv = nn.Conv2dTranspose(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layers = [pad, conv, norm]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
