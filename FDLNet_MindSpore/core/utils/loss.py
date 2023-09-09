"""Custom losses."""
import mindspore
import numpy
from mindspore import nn, ops, Parameter, Tensor
__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']


# TODO: optim function
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_construct(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).construct(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).construct(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def construct(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_construct(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).construct(*inputs))


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def construct(self, *inputs):
        if not self.se_loss and not self.aux:
            preds, target = tuple(inputs)
            inputs = tuple(list(preds) + [target])
            return super(EncNetLoss, self).construct(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).construct(pred1, target)
            loss2 = super(EncNetLoss, self).construct(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass)
            se_target.set_dtype(pred.dtype)
            loss1 = super(EncNetLoss, self).construct(pred, target)
            loss2 = self.bceloss(ops.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass)
            se_target.set_dtype(pred1.dtype)
            loss1 = super(EncNetLoss, self).construct(pred1, target)
            loss2 = super(EncNetLoss, self).construct(pred2, target)
            loss3 = self.bceloss(ops.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.shape[0]
        tvect = Parameter(ops.zeros((batch, nclass)))
        for i in range(batch):
            hist = ops.histc(target[i].float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return Tensor(tvect)


# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, nclass, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight

    def construct(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = Tensor(ops.interpolate(target, pred_sub4.shape[2:], mode='bilinear', align_corners=True).squeeze(1),dtype=mindspore.int32)
        target_sub8 = Tensor(ops.interpolate(target, pred_sub8.shape[2:], mode='bilinear', align_corners=True).squeeze(1),dtype=mindspore.int32)
        target_sub16 = Tensor(ops.interpolate(target, pred_sub16.shape[2:], mode='bilinear', align_corners=True).squeeze(1),dtype=mindspore.int32)

        loss1 = super(ICNetLoss, self).construct(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).construct(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).construct(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)


class OhemCrossEntropy2d(nn.Cell):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.mask_mid = Tensor([0])
        if use_weight:
            weight = Tensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507], dtype=mindspore.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def construct(self, pred, target):
        n, c, h, w = pred.shape
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = Tensor(target * valid_mask, dtype=mindspore.int64)
        num_valid = valid_mask.sum()

        prob = ops.softmax(pred, axis=1)
        prob = prob.swapaxes(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = ops.masked_fill(prob, ~valid_mask, 1)
            # prob = prob.masked_fill_(~valid_mask, 1)
            if len(self.mask_mid)!=len(target):
                self.mask_mid = Tensor(numpy.arange(len(target)))
            mask_prob = prob[target, self.mask_mid]
            threshold = self.thresh
            if self.min_kept > 0:
                index = numpy.argsort(mask_prob.asnumpy())
                threshold_index = index[min(len(index), self.min_kept) - 1]
                maskk = mask_prob[Tensor(threshold_index)]
                if maskk > self.thresh:
                    threshold = maskk
            kept_mask = ops.le(mask_prob, threshold)
            valid_mask = valid_mask.astype(mindspore.int64) * kept_mask.astype(mindspore.int64)
            target = (target * kept_mask.astype(mindspore.int64)).astype(mindspore.int64)

        target = ops.masked_fill(target, ~valid_mask, self.ignore_index)
        # target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w).astype(mindspore.int32)


        return self.criterion(pred, target)

class FDLNetLoss(nn.Cell):
    def __init__(self, classes=19, ignore_index=255, norm=False, upper_bound=1.0, mode='train',
                aux_weight=0.4, seg_weight=1, att_weight=0.01, **kwargs):
        super(FDLNetLoss, self).__init__()
        self.num_classes = classes
        self.ignore_index = ignore_index
        if mode == 'train':
            self.seg_loss = OhemCrossEntropy2d(
                                               ignore_index=ignore_index)
        elif mode == 'val':
            self.seg_loss = OhemCrossEntropy2d(
                                               ignore_index=ignore_index)

        self.aux_loss = OhemCrossEntropy2d(ignore_index=ignore_index)
        self.att_loss = OhemCrossEntropy2d(min_kept=5000,ignore_index=ignore_index)
        self.aux_weight = aux_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight


    def edge_attention(self, input, target, edge):
        n, c, h, w = input.shape
        filler = ops.ones_like(target) * self.ignore_index
        return self.att_loss(input,
                             ops.where(edge.max(1, return_indices=True)[0] == 1., target, filler))

    def construct(self, inputs, targets):
        pred1, pred2 = inputs
        # print(pred1.shape, pred2.shape)
        target, edgemap =targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(pred1, target)
        losses['aux_loss'] = self.aux_weight * self.aux_loss(pred2, target)

        losses['att_loss'] = self.att_weight * self.edge_attention(pred1, target, edgemap)
        return losses

class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_construct(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).construct(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).construct(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def construct(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_construct(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).construct(*inputs))


def get_segmentation_loss(model, use_ohem=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)

    model = model.lower()
    if model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'icnet':
        return ICNetLoss(**kwargs)
    elif model == 'fdlnet':
        return FDLNetLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)
