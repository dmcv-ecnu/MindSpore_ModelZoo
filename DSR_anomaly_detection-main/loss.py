import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as mnp

class FocalLoss(nn.Cell):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def construct(self, logit, target):  # (8, 2, 256, 256) (8, 1, 256, 256)
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.shape[0], logit.shape[1], -1)    # (8, 2, 65536)
            logit = logit.permute(0, 2, 1)  # (8, 65536, 2)
            logit = logit.view(-1, logit.shape[-1])  # (524288, 2)
        target = ops.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha

        if alpha is None:
            alpha = ops.ones(num_class) # 在mindspore里面，这里默认是填充1
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            # alpha = torch.FloatTensor(alpha).view(num_class, 1) # question
            alpha = alpha_reshaped = Tensor(alpha, mindspore.float32).reshape((num_class, 1))
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = ops.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        idx = target.long()

        one_hot_key = mnp.zeros((target.shape[0], num_class), dtype=mnp.float32)
        for i in range(idx.shape[1]):
            one_hot_key[i, idx[i]] = 1
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)

        if self.smooth:
            one_hot_key = ops.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = ops.squeeze(alpha)
        loss = -1 * alpha * ops.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
if __name__ == "__main__":
    myNet = FocalLoss()
    logits = Tensor([0.7, 0.8, 0.9])
    label = Tensor([1, 1, 1])
    loss = myNet(logits, label)
