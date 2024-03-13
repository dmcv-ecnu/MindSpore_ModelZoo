import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


def bce_iou_loss(pred, mask):
    weight = 1 + 5 * ops.abs(ops.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = ops.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = ops.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(axis=(2, 3)) / weight.sum(axis=(2, 3))
    weighted_iou = (weight * iou).sum(axis=(2, 3)) / weight.sum(axis=(2, 3))

    return (weighted_bce + weighted_iou).mean()


# class BCEIOULoss(nn.Cell):
#     def __init__(self):
#         super(BCEIOULoss, self).__init__()
#         self.avg_pol = nn.AvgPool2d(kernel_size=31, stride=1, pad_mode="pad", padding=15)
#         self.sigmoid = nn.Sigmoid()
#         self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
#
#     def construct(self, pred, mask):
#         weight = 1 + 5 * ops.abs(self.avg_pol(mask) - mask)
#
#         bce = self.binary_cross_entropy(pred, mask)
#
#         pred = self.sigmoid(pred)
#         inter = pred * mask
#         union = pred + mask
#         iou = 1 - (inter + 1) / (union - inter + 1)
#
#         weighted_bce = (weight * bce).sum(axis=(2, 3)) / weight.sum(axis=(2, 3))
#         weighted_iou = (weight * iou).sum(axis=(2, 3)) / weight.sum(axis=(2, 3))
#
#         return (weighted_bce + weighted_iou).mean()
#
#
# class DiceBCELoss(nn.Cell):
#     def __init__(self):
#         super(DiceBCELoss, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
#
#     def construct(self, pred, mask):
#         bce = self.binary_cross_entropy(pred, mask)
#
#         pred = self.sigmoid(pred)
#         inter = pred * mask
#         union = pred + mask
#         iou = 1 - (2. * inter + 1) / (union + 1)
#
#         return (bce + iou).mean()
#
#
# class TverskyLoss(nn.Cell):
#     def __init__(self, alpha=0.5, beta=0.5, gamma=2):
#         super(TverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.sigmoid = nn.Sigmoid()
#
#     def construct(self, pred, mask):
#         pred = self.sigmoid(pred)
#
#         # flatten label and prediction tensors
#         pred = pred.view(-1)
#         mask = mask.view(-1)
#
#         # True Positives, False Positives & False Negatives
#         TP = (pred * mask).sum()
#         FP = ((1 - mask) * pred).sum()
#         FN = (mask * (1 - pred)).sum()
#
#         Tversky = (TP + 1) / (TP + self.alpha * FP + self.beta * FN + 1)
#
#         return (1 - Tversky) ** self.gamma
#
#
# class TverskyBCELoss(nn.Cell):
#     def __init__(self, alpha=0.5, beta=0.5, gamma=2):
#         super(TverskyBCELoss, self).__init__()
#         self.tversky_loss = TverskyLoss(alpha, beta, gamma)
#         self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='mean')
#
#     def construct(self, pred, mask):
#         bce = self.binary_cross_entropy(pred, mask)
#         tversky = self.tversky_loss(pred, mask)
#         return bce + tversky