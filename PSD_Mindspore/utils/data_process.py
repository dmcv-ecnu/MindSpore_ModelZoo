import pdb

import mindspore.numpy as msnp
import mindspore.ops as ops
import numpy.random
import numpy as np
from mindspore import Tensor
from sklearn.neighbors import KDTree
#
# def knn(query, support, k):
#     # this implementation costs too much memory
#     s = support.shape[-1]
#     q = query.shape[-1]
#
#     query = msnp.expand_dims(query, 3)
#     query = msnp.repeat(query, s, 3)
#
#     support = msnp.expand_dims(support, 2)
#     support = msnp.repeat(support, q, 2)
#
#     diff_square = ops.square(query - support)
#     square_dist = ops.reduce_sum(diff_square, 1)
#     dist = ops.sqrt(square_dist)
#     # -dist mean the bigger the value is, the nearer the samples are
#     values, indices = ops.TopK()(-dist, k)
#     return indices
#     # search_tree = KDTree(support)
#     # search_tree.query(query)
#
#
#
# def data_aug(xyz, color, labels, num_out):
#     num_in = len(xyz)
#     dup = np.random.choice(num_in, num_out - num_in)
#     xyz_dup = xyz[dup, ...]
#     xyz_aug = np.concatenate([xyz, xyz_dup], 0)
#     color_dup = color[dup, ...]
#     color_aug = np.concatenate([color, color_dup], 0)
#     label_dup = labels[dup, ...]
#     label_aug = np.concatenate([labels, label_dup], 0)
#     return xyz_aug, color_aug, label_aug


def gather_neighbour(xyz, neighbor_idx):
    """
        xyz: [b, d, n]
        neighbor_idx: [b, n, k]
        return: [b, d, n, k]
    """
    # gather the coordinates or features of neighboring points
    B, N, K = neighbor_idx.shape
    D = xyz.shape[1]
    neighbor_idx = msnp.expand_dims(neighbor_idx, 1)
    neighbor_idx = msnp.repeat(neighbor_idx, D, 1)
    neighbor_idx = neighbor_idx.reshape(B, D, -1)
    r = ops.gather_d(xyz, 2, neighbor_idx)  # shape (B, D, N * K)
    r = r.reshape(B, D, N, K)
    return r