"""
Author: Qihang Ma
Date: Sep 2022
"""
import mindspore.numpy as msnp
import mindspore as ms

def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc

        Parameters
        ----------
        scores: ms.Tensor, dtype = float32, shape (B?, C, N)
            raw scores for each class
        labels: ms.Tensor, dtype = int64, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.shape[-2] # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = scores.argmax(axis=-2)

    accuracies = 0

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = msnp.logical_and(accuracy_mask , label_mask).astype(ms.float32).sum()
        per_class_accuracy /= label_mask.astype(ms.float32).sum()
        if label==0:
            accuracies = per_class_accuracy
        else:
            accuracies = msnp.append(accuracies, per_class_accuracy)
    # overall accuracy
    accuracies = msnp.append(accuracies, accuracy_mask.astype(ms.float32).mean())
    return accuracies

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc

        Parameters
        ----------
        scores: ms.Tensor, dtype = float32, shape (B?, C, N)
            raw scores for each class
        labels: ms.Tensor, dtype = int64, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.shape[-2] # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = scores.argmax(axis=-2)

    ious = 0

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = msnp.logical_and(pred_mask , labels_mask).astype(ms.float32).sum() / msnp.logical_or(pred_mask , labels_mask).astype(ms.float32).sum()
        if label==0:
            ious = iou
        else:
            ious = msnp.append(ious, iou)
    ious = msnp.append(ious, msnp.nanmean(ious))
    return ious
