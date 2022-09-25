# 互学习的教师网络，对应pytorch代码的edsr_lr2hrsl

from model.common import *

import mindspore as ms
from mindspore import nn

def make_model(args):
    return EDSR(args)

class EDSR(nn.Cell):
    def __init__(self, args, conv=make_default_conv):
        super().__init__()
        self.args = args
        n_resblock = 4 # 这个特殊，不用全局的参数
        n_colors = args.n_colors
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale
        self.res_scale = args.res_scale
        act = nn.ReLU()

        self.norm = RgbNormal(args.rgb_range, rgb_mean, rgb_std)
        self.de_norm = RgbNormal(args, rgb_mean, rgb_std, True)


        rgb_mean = (0.4488, 0.4371, 0.4040) #这个应该要改，看看怎么方便的计算数据集的这些数据，如果能实时计算那就更好了
        rgb_std = (1.0, 1.0, 1.0)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_body_1 = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=self.res_scale
            ) for _ in range(n_resblock // 4)
        ]

        m_body_2 = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=self.res_scale
            ) for _ in range(n_resblock // 4)
        ]

        

        self.head = nn.SequentialCell(m_head) # 为什么原来的代码里head有后缀2
        self.body_1 = nn.SequentialCell(m_body_1)
        self.body_2 = nn.SequentialCell(m_body_2)

        modules_tail =[
            nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, pad_mode="same", has_bias=True),
            nn.Conv2d(n_feats, 3 * (self.scale ** 2), kernel_size=kernel_size, pad_mode="same"),
            PixelShuffle(self.scale)
        ]

        self.tail = nn.SequentialCell(modules_tail)

    def construct(self, x):
        lr = self.norm(x)
        x = self.head(lr)

        res_0 = x
        res_1 = self.body_1(x)
        res_2 = self.body_2(res_1)

        res = res_2 + x
        x = self.tail(res)
        x = self.de_norm(x)

        return x
        