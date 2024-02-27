# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import torch
from .softmax_loss import CrossEntropyLabelSmooth
import mindspore

def make_loss(num_classes, ignore_index=-1):

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)

    def loss_func(i2tscore, target):

        mask = (target != ignore_index)
        # print(mask.shape)
        # print(i2tscore.shape)
        # print(target.shape)
        i2tscore = i2tscore[mask,:]
        target = target[mask]

        if mask.sum() == 0:
            return mindspore.tensor([0.0]).cuda()

        I2TLOSS = xent(i2tscore, target)

        return I2TLOSS

    return loss_func