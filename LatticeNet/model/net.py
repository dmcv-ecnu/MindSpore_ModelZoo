import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
## 组合系数

def make_model(args, parent=False):
    return LatticeNet(args)


class MeanShift(mindspore.nn.Conv2d):
    """add or sub means of input data"""
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1, dtype=mindspore.float32):
        std = mindspore.Tensor(rgb_std, dtype)
        weight = mindspore.Tensor(np.eye(3), dtype).reshape(3, 3, 1, 1) / std.reshape(3, 1, 1, 1)
        bias = sign * rgb_range * mindspore.Tensor(rgb_mean, dtype) / std
        super(MeanShift, self).__init__(3, 3, kernel_size=1, has_bias=True, weight_init=weight, bias_init=bias)
        for p in self.get_parameters():
            p.requires_grad = False

# 由于像素随机采样不属于cell，故创造该类
class PixelShuffle(nn.Cell):
    """perform pixel shuffle"""
    def __init__(self, upscale_factor):
        super().__init__()
        #TODO:pixshuffle能用DepthToSpace替代吗？
        self.DepthToSpace = mindspore.ops.DepthToSpace(upscale_factor)

    def construct(self, x):
        return self.DepthToSpace(x)


class CC(nn.Cell):
    def __init__(self, channel):
        super(CC, self).__init__()
        # 定义所需计算
        # 全局均值,upbranch
        reduction = 16
        # 每个通道数的值数量为1x1
        self.avg_pool = ops.AdaptiveAvgPool2D(1)

        self.conv_mean = nn.SequentialCell(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, has_bias=True),
            #TODO:inplace? 可以 计算结果不受影响，只是节省内存
            nn.ReLU(),# 无法原地覆盖
            nn.Conv2d(channel // reduction, channel, 1, padding=0, has_bias=True),
            nn.Sigmoid()
        )

        # lowbranch
        self.conv_std = nn.SequentialCell(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, has_bias=True),
            nn.Sigmoid()
        )

    def construct(self, x):

        # cpu无法使用AdaptiveAvgPool2D
        # mean
        # ca_mean = self.avg_pool(x)
        # ca_mean = self.conv_mean(ca_mean)

        # std C? size -> shape 计算全局标准差
        m_batchsize, C, height, width = x.shape
        x_dense = x.view(m_batchsize, C, -1)
        # ca_std = Tensor.std(x_dense, axis=2, keepdims=True)#dim axis
        ca_std = x_dense.std(axis=2, keepdims=True)
        ca_std = ca_std.view(m_batchsize, C, 1, 1)

        ca_mean = x_dense.mean(axis=2, keep_dims=True)
        ca_mean = ca_mean.view(m_batchsize, C, 1, 1)

        ca_var = self.conv_std(ca_std)
        ca_mean = self.conv_mean(ca_mean)

        cc = (ca_mean + ca_var) / 2.0
        return cc


class LatticeBlock(nn.Cell):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(LatticeBlock, self).__init__()

        self.D3 = nFeat#初始通道数
        self.d = nDiff
        self.s = nFeat_slice
        #晶格块
        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-nDiff, nFeat, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.SequentialCell(*block_0)

        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)

        block_1 = []
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat, kernel_size=3, pad_mode='pad', padding=1, has_bias=True))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.SequentialCell(*block_1)

        self.fea_ca2 = CC(nFeat)
        self.x_ca2 = CC(nFeat)

        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, has_bias=True)

    def construct(self, x):
        # analyse unit
        x_feature_shot = self.conv_block0(x)
        fea_ca1 = self.fea_ca1(x_feature_shot)
        x_ca1 = self.x_ca1(x)
        #Pi-1， Qi-1
        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        # synthes_unit
        #Pi，Qi
        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        cat = ops.Concat(axis=1)
        out = cat((p3z, q3z))
        out = self.compress(out)

        return out

    ## LatticeNet 主体
class LatticeNet(nn.Cell):
    def __init__(self, args):
        super(LatticeNet, self).__init__()


        n_feats = args.n_feats
        scale = args.scale[0]

        nFeat = 64
        nDiff = 16
        nFeat_slice = 4
        nChannel = 3

        # RGB mean for DIV2K
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        # define head module
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)

        # define body module 定义四个晶格块
        self.body_unit1 = LatticeBlock(n_feats, nDiff, nFeat_slice)
        self.body_unit2 = LatticeBlock(n_feats, nDiff, nFeat_slice)
        self.body_unit3 = LatticeBlock(n_feats, nDiff, nFeat_slice)
        self.body_unit4 = LatticeBlock(n_feats, nDiff, nFeat_slice)

        #反向融合模块
        self.T_tdm1 = nn.SequentialCell(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, has_bias=True),
            nn.ReLU())
        self.L_tdm1 = nn.SequentialCell(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, has_bias=True),
            nn.ReLU())

        self.T_tdm2 = nn.SequentialCell(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, has_bias=True),
            nn.ReLU())
        self.L_tdm2 = nn.SequentialCell(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, has_bias=True),
            nn.ReLU())

        self.T_tdm3 = nn.SequentialCell(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, has_bias=True),
            nn.ReLU())
        self.L_tdm3 = nn.SequentialCell(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, has_bias=True),
            nn.ReLU())

        # define tail module 上采样层
        modules_tail = [nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
                        nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
                        PixelShuffle(scale)]
        self.tail = nn.SequentialCell(*modules_tail)

       # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def construct(self, x):
        x = self.sub_mean(x)

        x = self.conv1(x)
        x = self.conv2(x)

        res1 = self.body_unit1(x)
        res2 = self.body_unit2(res1)
        res3 = self.body_unit3(res2)
        res4 = self.body_unit4(res3)

        T_tdm1 = self.T_tdm1(res4)
        L_tdm1 = self.L_tdm1(res3)
        cat = ops.Concat(axis=1)
        out_TDM1 = cat((T_tdm1, L_tdm1))

        T_tdm2 = self.T_tdm2(out_TDM1)
        L_tdm2 = self.L_tdm2(res2)
        out_TDM2 = cat((T_tdm2, L_tdm2))

        T_tdm3 = self.T_tdm3(out_TDM2)
        L_tdm3 = self.L_tdm3(res1)
        out_TDM3 = cat((T_tdm3, L_tdm3))

        res = out_TDM3 + x
        out = self.tail(res)

        x = self.add_mean(out)

        return x


