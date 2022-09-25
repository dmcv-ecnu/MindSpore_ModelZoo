import mindspore.numpy as msnp
import mindspore.nn as nn

from utils import data_process as dp
from models.common import SharedMLP


class EdgeConv(nn.Cell):
    def __init__(self, d_in):
        super(EdgeConv, self).__init__()
        self.net = SharedMLP(2 * d_in, d_in, use_bn=False, input_1d=False)

    def construct(self, feature, neighbor_idx):
        feature = self.relative_get_feature(feature, neighbor_idx)
        feature = self.net(feature)
        feature = msnp.max(feature, 3)
        return feature

    def relative_get_feature(self, feature, neighbor_idx):
        k = neighbor_idx.shape[-1]
        neighbor_feature = dp.gather_neighbour(feature, neighbor_idx)   # 1
        feature = msnp.expand_dims(feature, 3)
        """
        问题在下面这一行代码，详细原因见
        https://gitee.com/mindspore/mindspore/issues/I4ARHY
        反向传播中reducesum的维度过大而不支持
        其他代码中，也有一堆类似操作，如gather_naghbour，也用了expand + repeat
        """
        feature = msnp.repeat(feature, k, 3)
        relative_feature = neighbor_feature - feature   # 2
        relative_feature = msnp.concatenate([relative_feature, feature], 1)  # 3
        return relative_feature
