#! /usr/bin/env python

import numbers
import mindspore as ms
from mindspore import nn

class LossFn(nn.LossBase):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")

    def construct(self, ml_w, sr, hr, peer_sr):
        loss1 = self.loss(sr, hr)
        loss2 =  self.loss(sr, peer_sr) # 乘以系数变得太小了？
        return loss1 + loss2
        

class Loss(nn.Cell):
    def __init__(self, backbone: nn.Cell, loss_fn:nn.LossBase=LossFn(), reduction='mean'):
        super().__init__(reduction)
        self._backbone = backbone
        self._loss_fn = loss_fn
    
    from typing import List
    TensorList = List[ms.Tensor]
    def construct(self, ml_w:float, lr:ms.Tensor, hr:ms.Tensor, peer_sr:ms.Tensor):
        
        sr = self._backbone(lr)[0][-1]

        loss = self._loss_fn(ml_w, sr, hr, peer_sr)

        """
        if len(inter_sr) != 0:
            loss_isr = self._loss_fn(inter_sr[0], hr) + self._loss_fn(inter_sr[1], hr) + self._loss_fn(inter_sr[2], hr)

        if len(teacher_labels) != 0:
            lss_tea = self._loss_fn(inter_sr[2], teacher_labels[0]) + self._loss_fn(inter_sr[1], teacher_labels[1]) + self._loss_fn(inter_sr[2], teacher_labels[2])

        """
        """
        此处待实现
        """
        
        return loss