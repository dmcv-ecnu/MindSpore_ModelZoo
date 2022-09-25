from mindspore import ops

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F
from collections import OrderedDict
from model.common import *
from ..option import Args

def make_model(args:Args):
    return PAN(args)

class PA(nn.Cell):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = x * y

        return out

class PAConv(nn.Cell):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, pad_mode="same", has_bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, pad_mode="same", has_bias=False) # 3x3 convolution

    def construct(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = self.k3(x) * y
        out = self.k4(out)

        return out

class SCPA(nn.Cell):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, has_bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, has_bias=False)
        
        self.k1 = nn.SequentialCell(
                    [nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride, pad_mode="pad",
                        padding=dilation, dilation=dilation,
                        has_bias=False)]
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, has_bias=False)
        
        self.lrelu = nn.LeakyReLU(0.2)
        self.concat_op = ops.Concat(axis=1)


    def construct(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out = self.conv3(self.concat_op([out_a, out_b]))
        out += residual

        return out
    
class PAN(nn.Cell):
    def __init__(self, args, conv=make_default_conv):
        super(PAN, self).__init__()

        in_nc = 3
        out_nc = 3
        nf = 40
        unf = 24
        nb = 16
        # scale = 4
        self.num_block = [4, 4, 4, 4]
        n_feats = 40

        # SCPA
        # SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = args.scale
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, pad_mode="same", has_bias=True)
        
        ### main blocks
        # self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb)
        # define body module
        m_body_1 = [
            SCPA(nf, reduction=2) for _ in range(nb // 4)
        ]
        m_body_2 = [
            SCPA(nf, reduction=2) for _ in range(nb // 4)
        ]
        m_body_3 = [
            SCPA(nf, reduction=2) for _ in range(nb // 4)
        ]
        m_body_4 = [
            SCPA(nf, reduction=2) for _ in range(nb // 4)
        ]

        self.body_1 = nn.SequentialCell(m_body_1)
        self.body_2 = nn.SequentialCell(m_body_2)
        self.body_3 = nn.SequentialCell(m_body_3)
        self.body_4 = nn.SequentialCell(m_body_4)

        self.trunk_conv1 = nn.Conv2d(nf, nf, 3, 1, pad_mode="same", has_bias=True)
        self.trunk_conv2 = nn.Conv2d(nf, nf, 3, 1, pad_mode="same", has_bias=True)
        self.trunk_conv3 = nn.Conv2d(nf, nf, 3, 1, pad_mode="same", has_bias=True)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, pad_mode="same", has_bias=True)
        
        #### upsampling
        modules_tail1 = [nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode="same", has_bias=True),
                        nn.Conv2d(n_feats, 3 * (self.scale ** 2), kernel_size=3, pad_mode="same", has_bias=True),
                        PixelShuffle(self.scale)]
        # modules_tail1 = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]

        self.tail1 = nn.SequentialCell(*modules_tail1)

        modules_tail2 = [nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode="same", has_bias=True),
                        nn.Conv2d(n_feats, 3 * (self.scale ** 2), kernel_size=3, pad_mode="same", has_bias=True),
                        PixelShuffle(self.scale)]
        # modules_tail2 = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]
        self.tail2 = nn.SequentialCell(*modules_tail2)

        modules_tail3 = [nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode="same", has_bias=True),
                        nn.Conv2d(n_feats, 3 * (self.scale ** 2), kernel_size=3, pad_mode="same", has_bias=True),
                        PixelShuffle(self.scale)]
        # modules_tail3 = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]
        self.tail3 = nn.SequentialCell(*modules_tail3)

        # stage 4
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, pad_mode="same", has_bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, pad_mode="same", has_bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, pad_mode="same", has_bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, pad_mode="same", has_bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, pad_mode="same", has_bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, x):

        ILR = nn.ResizeBilinear()(x, scale_factor=self.scale, align_corners=False)

        # head
        x_0 = self.conv_first(x)
        res_0 = x_0

        # body
        # stage1
        res_1 = self.body_1(res_0)
        res1 = x_0 + res_1
        out1 = self.tail1(res1)
        out1 = out1 + ILR

        # stage2
        res_2 = self.body_2(res_1)
        res2 = x_0 + res_2
        out2 = self.tail2(res2)
        out2 = out2 + ILR

        # stage3
        res_3 = self.body_3(res_2)
        res3 = x_0 + res_3
        out3 = self.tail3(res3)
        out3 = out3 + ILR

        # stage4
        res_4 = self.body_4(res_3)
        res_4 = self.trunk_conv(res_4)
        res4 = x_0 + res_4
        # tail3
        fea = None
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(nn.ResizeBilinear()(res4, scale_factor=self.scale))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(nn.ResizeBilinear()(res4, scale_factor=2))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(nn.ResizeBilinear()(fea, scale_factor=2))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)
        out4 = out + ILR

        return [out1, out2, out3, out4], [res_0, res_1, res_2, res_3, res_4, ILR]