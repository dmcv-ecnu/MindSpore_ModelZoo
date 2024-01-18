# -*-coding:utf-8-*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase

from src.utils.tools import ConfigS3DIS as cfg


class WeightCEloss(LossBase):
    """weight ce loss"""

    def __init__(self, weights, num_classes):
        super(WeightCEloss, self).__init__()
        self.weights = weights
        self.num_classes = num_classes
        self.onehot = nn.OneHot(depth=num_classes)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, logits, labels, valid_idx):
        logits = logits.swapaxes(-2, -1).reshape((-1, self.num_classes))
        one_hot_label = self.onehot(labels)  # [2b*n, 13]
        weights = self.weights * one_hot_label  # [2b*n, 13]
        weights = P.ReduceSum()(weights, 1)*valid_idx  # [2b*n]
        logit = P.cast(logits, mstype.float32)
        one_hot_label = P.cast(one_hot_label, mstype.float32)
        unweighted_loss = self.ce(logit, one_hot_label)  # [2b*n]
        weighted_loss = unweighted_loss * weights * valid_idx  # [2b*n]
        cnt_valid = P.ReduceSum()(valid_idx.astype(mstype.float32))
        CE_loss = P.ReduceSum()(weighted_loss) / cnt_valid  # [1]

        return CE_loss


class JSLoss(LossBase):
    """Jensen-Shannon divergence"""

    def __init__(self, b):
        super(JSLoss, self).__init__()
        self.b = b
        self.softmax = nn.Softmax(axis=-1)
        self.norm = nn.Norm(axis=-1, keep_dims=False)

    def construct(self, logits):
        logits = logits.swapaxes(-2, -1)
        # logits = self.softmax(logits)
        logits_clean = logits[:self.b, :, :].reshape(-1, logits.shape[-1])
        logits_noise = logits[self.b:, :, :].reshape(-1, logits.shape[-1])
        p1 = P.cast(logits_clean, mstype.float32)
        p2 = P.cast(logits_noise, mstype.float32)
        # q = 1/2*(p1+p2)
        # loss_kl = p1 * P.Log()(p1/(q+1e-4)+1e-4) + p2 * P.Log()(p2/(q+1e-4)+1e-4)
        loss_cos = (1-P.ReduceSum()(p1*p2, -1)/(self.norm(p1)*self.norm(p2)))*10

        return P.ReduceMean(keep_dims=False)(loss_cos)


class CRLoss(LossBase):
    """CR loss"""

    def __init__(self, num_classes):
        super(CRLoss, self).__init__()
        self.onehot = nn.OneHot(depth=num_classes)
        self.relu = nn.ReLU()

    def construct(self, rs1, rs2, labels):
        label_pool_one_hot = self.onehot(labels)
        Afinite_hot = P.matmul(label_pool_one_hot, P.Transpose()(label_pool_one_hot, (1, 0)))

        rs_map_soft = P.matmul(rs1, P.Transpose()(rs2, (1, 0)))
        rs_map_soft = self.relu(rs_map_soft)
        rs_map_soft = P.clip_by_value(rs_map_soft, 1e-4, 1-(1e-4))
        Afinite = Afinite_hot.reshape([-1, 1])
        rs_map = rs_map_soft.reshape([-1, 1])
        loss_cr = -1.0 * P.ReduceMean()(Afinite * P.Log()(rs_map) +
                                        (1 - Afinite) * P.Log()(1 - rs_map))
        A_R = P.ReduceSum()(Afinite_hot * rs_map_soft, 1)
        loss_tjp = -1.0 * P.ReduceMean()(P.Log()(P.Div()(A_R, P.ReduceSum()(rs_map_soft, 1))))
        loss_tjr = -1.0 * P.ReduceMean()(P.Log()(P.Div()(A_R, P.ReduceSum()(Afinite_hot, 1))))

        return loss_cr + loss_tjp + loss_tjr


class PSDWithLoss(nn.Cell):
    """PSD-net with loss"""

    def __init__(self, network, weights, num_classes, ignored_label_inds, is_training):
        super(PSDWithLoss, self).__init__()
        self.network = network
        self.num_classes = num_classes
        self.ignored_label_inds = ignored_label_inds
        self.is_training = is_training
        self.b = cfg.batch_size
        self.ce_loss = WeightCEloss(weights, num_classes)
        self.kl_loss = JSLoss(cfg.batch_size)
        # self.cr_loss = CRLoss(num_classes)

    def construct(self, feature, aug_feature, labels, input_inds, cloud_inds,
                  p0, p1, p2, p3, p4, n0, n1, n2, n3, n4, pl0, pl1, pl2,
                  pl3, pl4, u0, u1, u2, u3, u4):
        # handle input
        xyz = [p0, p1, p2, p3, p4]
        neighbor_idx = [n0, n1, n2, n3, n4]
        sub_idx = [pl0, pl1, pl2, pl3, pl4]
        interp_idx = [u0, u1, u2, u3, u4]

        # forward
        logits, _, _ = self.network(
            xyz, feature, aug_feature, neighbor_idx, sub_idx, interp_idx)

        global_labels = P.Concat(0)([labels, labels])
        global_labels = global_labels.reshape((-1,))  # [2b, n] --> [2b*n]
        labels = labels.reshape((-1,))  # [b, n] --> [b*n]

        # generate valid index for valid logits and labels selection for loss compute. due to the lack operator of mindspore.
        # (B*N,)
        ignore_mask = P.zeros_like(global_labels).astype(mstype.bool_)  # [b*n]
        for ign_label in self.ignored_label_inds:
            ignore_mask = P.logical_or(
                ignore_mask, P.Equal()(global_labels, ign_label))

        # Collect logits and labels that are not ignored
        valid_idx = P.logical_not(ignore_mask).astype(mstype.int32)  # [b*n]

        # compute loss
        ce_loss = self.ce_loss(logits, global_labels, valid_idx)
        kl_loss = self.kl_loss(logits)
        # cr_loss = self.cr_loss(rs1, rs2, global_valid_labels)

        loss = ce_loss + kl_loss  # + cr_loss

        return loss


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]
