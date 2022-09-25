import mindspore
import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore as ms


class Loss(nn.Cell):
    def __init__(self, num_classes):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.one_hot = nn.OneHot(depth=num_classes)
        self.sft = nn.Softmax()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.class_weights = Tensor([4.97221634, 5.76365047, 3.51712115, 26.95745666, 24.88543334,
                                           21.88936609, 13.81691932, 18.19796956, 15.99252438, 40.37542513,
                                           11.23598421, 30.91618524, 7.03602697], dtype=mstype.float32)
        self.v_idx = msnp.arange(0, 200, dtype=mstype.int32)

    def construct(self, logit, embedding, labels):
        # not finished yet.
        ignored_label = self.num_classes
        b, c, n = logit.shape
        d = embedding.shape[1]

        logit = msnp.transpose(logit, [0, 2, 1])
        logit = logit.reshape(-1, c)
        embedding = msnp.transpose(embedding, [0, 2, 1])
        embedding = embedding.reshape(-1, d)
        labels = labels.reshape(-1)

        # sd loss
        l1 = logit[:b * n // 2]
        l2 = logit[b * n // 2:]
        loss_sd = self.sd_loss(l1, l2)
        # seg loss
        labels = msnp.tile(labels, (2, ))
        """
        mindspore不支持non_zero操作，无法graph_mode，不取元素，直接计算会大量占用显存,
        """
        nplabels = labels.asnumpy()
        v_idx = np.where(nplabels != ignored_label)[0]
        v_idx = Tensor(v_idx)
        loss_seg = self.seg_loss(logit[v_idx], labels[v_idx])
        # aff loss
        loss_cr = self.aff_loss(embedding[v_idx], labels[v_idx])
        return loss_seg + loss_sd + loss_cr

    def seg_loss(self, logit, labels):
        weights = ops.gather_d(self.class_weights, 0, labels)
        unweighted_losses = self.cross_entropy(logit, labels)
        weighted_losses = unweighted_losses * weights

        return msnp.mean(weighted_losses)

    def sd_loss(self, l1, l2):
        p1 = self.sft(l1)
        p2 = self.sft(l2)
        q = 1 / 2 * (p1 + p2)
        l = ops.ReduceMean()(p1 * msnp.log(p1 / (q + 1e-9) + 1e-9) + \
                      p2 * msnp.log(p2 / (q + 1e-9) + 1e-9))
        return l

    def aff_loss(self, embedding, labels):
        label_one_hot = self.one_hot(labels) # m * num_class
        aff = msnp.dot(label_one_hot, label_one_hot.T).astype(mstype.float32)

        similarity = msnp.dot(embedding, embedding.T)
        similarity = ops.ReLU()(similarity)
        """
        1.3版本用np.clip会导致报错，1.5版本没尝试过
        """
        # similarity = msnp.clip(similarity, 1e-4, 1-(1e-4))
        similarity = ops.clip_by_value(similarity, Tensor(1e-4, dtype=mstype.float32),
                                       Tensor(1-(1e-4), dtype=mstype.float32))
        # L_ce
        flat_aff = aff.reshape(-1, 1)
        flat_similarity = similarity.reshape(-1, 1)
        loss_cr = -msnp.mean(flat_aff * msnp.log(flat_similarity) + \
                                   (1 - flat_aff) * msnp.log(1 - flat_similarity))
        # L_p
        p_up = ops.ReduceSum()(aff * similarity, 1) # in paper 0, in code 1
        p_down = ops.ReduceSum()(similarity, 1)
        loss_p = -1.0 * msnp.mean(msnp.log(p_up / p_down))
        # L_r
        r_up = p_up
        r_down = ops.ReduceSum()(aff, 1) # difference
        loss_r = -1.0 * msnp.mean(msnp.log(r_up / r_down))
        # return loss_cr
        return loss_cr + loss_p + loss_r
        # return Tensor(0.0, mindspore.float32)