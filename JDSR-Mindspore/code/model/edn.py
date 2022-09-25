#! /usr/bin/env python

import os
from mindspore import nn
from mindspore import Parameter

from ..option import Args
from model.common import *

def is_trained(args:Args): # 判断模型是否已经训练好
    model_path = args.pre_train
    scale = args.scale
    model_path = os.path.join(model_path, f"EDN_x{scale}.ckpt")
    if os.path.exists(model_path):
        return True
    return False

def make_model(args):
    return EDN(args)

class EDN(nn.Cell): # 必须继承自Cell
    """
    Stage 1所使用的Encoder-Decoder网络
    """
    def __init__(self, args:Args, conv=make_default_conv):
        super().__init__() # 继承属性
        n_resblocks  = args.n_resblocks
        n_colors = args.n_colors
        n_feats = args.n_feats
        scale = args.scale
        res_scale = args.res_scale

        kernel_size = 3 # 卷积核大小
        rgb_range = 255 # rgb范围
        rgb_mean = (0.4488, 0.4371, 0.4040) # DIV2K数据集的rgb均值
        rgb_std = (1.0, 1.0, 1.0) # DIV2K数据集的rgb标准差
        
        self.norm = RgbNormal(rgb_range, rgb_mean, rgb_std) 
        self.de_norm = RgbNormal(rgb_range, rgb_mean, rgb_std, True)
        act = nn.ReLU() # mindspore的ReLu构造方法没有参数

        # Encoder:
        if scale == 2: # 2x
            m_head_lr = [
            nn.Conv2d(n_colors, n_feats, kernel_size=kernel_size, pad_mode="same", stride=2, has_bias=True)
        ]
        elif scale == 4: # 4x
            m_head_lr = [
            nn.Conv2d(args.n_colors, n_feats, kernel_size, pad_mode="same", stride=scale // 2, has_bias=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, pad_mode="same", stride=scale // 2, has_bias=True)]
        else: # 默认值为3，确保不会出现2, 3, 4之外的值
            scale = 3
            m_head_lr = [ # 3x
                nn.Conv2d(n_colors, n_feats, kernel_size=kernel_size, pad_mode="pad", padding=0, stride=3, has_bias=True)
            ]
            
        # 残差块8个
        m_body_1lr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale # conv 是一个函数
            ) for _ in range(n_resblocks // 4)
        ]

        m_body_2lr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]

        m_body_3lr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]

        m_body_4lr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]

        m_convlr = [
            conv(n_feats, n_feats, kernel_size)    
        ]

        m_end_convlr = [
            conv(n_feats, n_colors, kernel_size)
        ]

        self.headlr = nn.SequentialCell(m_head_lr)
        self.body_1lr = nn.SequentialCell(m_body_1lr)
        self.body_2lr = nn.SequentialCell(m_body_2lr)
        self.body_3lr = nn.SequentialCell(m_body_3lr)
        self.body_4lr = nn.SequentialCell(m_body_4lr)
        self.convlr = nn.SequentialCell(m_convlr)
        self.end_convlr = nn.SequentialCell(m_end_convlr)

        # Decoder

        # 3x3卷积
        m_head_hr = [conv(n_colors, n_feats, kernel_size)]  # args.n_colors  1

        # define body module
        m_body_1hr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]
        m_body_2hr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]
        m_body_3hr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]
        m_body_4hr = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks // 4)
        ]

        m_convhr = [
            conv(n_feats, n_feats, kernel_size)
        ]

        m_tail = [nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, pad_mode="pad", padding=1, has_bias=True),
                  nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=kernel_size, pad_mode="pad", padding=1, has_bias=True),
                  PixelShuffle(scale)]

        self.headhr = nn.SequentialCell(m_head_hr)
        self.body_1hr = nn.SequentialCell(m_body_1hr)
        self.body_2hr = nn.SequentialCell(m_body_2hr)
        self.body_3hr = nn.SequentialCell(m_body_3hr)
        self.body_4hr = nn.SequentialCell(m_body_4hr)
        self.convhr = nn.SequentialCell(m_convhr)
        self.up = nn.SequentialCell(m_tail)
        
        from typing import List
        ParamList = List[Parameter]
        params: ParamList = self.get_parameters()
        for v in params:
            v.requires_grad = True # 训练时需要优化
    
    def construct(self, x, predict=False): # 必须重写construct
        x_lr = self.norm(x) # 减去平均值，标准化
        x_lr = self.headlr(x_lr)  # 卷积
        res_1lr = self.body_1lr(x_lr) # 8个残差块 
        res_2lr = self.body_2lr(res_1lr)
        res_3lr = self.body_3lr(res_2lr)
        res_4lr = self.body_4lr(res_3lr)
        res_endlr = self.convlr(res_4lr)
        res_lr = res_endlr + x_lr # 加回输入值
        res_lr = self.end_convlr(res_lr) # 3x3卷积，子像素卷积
        x1 = self.de_norm(res_lr) # 加回平均值

        x_hr = self.norm(x1) # 减去平均值
        x_hr = self.headhr(x_hr)  # 3x3卷积
        res_1hr = self.body_1hr(x_hr)  # 8个残差块
        res_2hr = self.body_2hr(res_1hr)
        res_3hr = self.body_3hr(res_2hr)
        res_4hr = self.body_4hr(res_3hr)
        res_endhr = self.convhr(res_4hr)
        res_hr = res_endhr + x_hr # 加回输入值
        res_hr = self.up(res_hr) # 子像素网络上采样
        x2 = self.de_norm(res_hr) #加回平均值

        # 推理时仅需要前半段结果
        if predict == True:
            return x1

        return [x1, x2]

class LossEDN(nn.LossBase):
    """
    EDN网络的自定义损失函数
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
    
    def construct(self, outputlr, outputhr, lr, hr):
        x1 = self.loss(lr, outputlr)
        x2 = self.loss(hr, outputhr)
        return x1 + x2

class EDNWithLoss(nn.Cell):
    """
    EDN与Loss函数的封装
    """
    def __init__(self, backbone:nn.Cell, loss_fn:nn.LossBase, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, lr, hr):
        output = self._backbone(hr)
        return self._loss_fn(output[0], output[1], lr, hr)
