from mindspore import ops
import mindspore as ms


def l1_norm(x, dim, eps=1e-10):
    # return x
    # return ops.L2Normalize(axis=dim, epsilon=eps)(x)
    out = ops.Abs()(x)
    out = out.sum(dim, keepdims=True)
    out = out + eps
    out = ops.Div()(x, out)
    return out


def label2edge(label):
    num_samples = label.shape[1]
    label_i = ops.ExpandDims()(label, -1)
    label_i = ops.Tile()(label_i, (1, 1, num_samples))
    label_j = label_i.swapaxes(1, 2)

    edge = ops.Equal()(label_i, label_j).astype(ms.float32)
    edge = ops.ExpandDims()(edge, 1)
    edge = ops.Concat(1)([edge, 1. - edge])
    return edge


def get_lr(lr):
    lrs = []
    for i in range(6):
        lrs.append(lr * (0.5 ** i))
    return lrs


def hit(logit, label):
    pred = ops.Argmax(1)(logit)
    hit = ops.Equal()(pred, label).astype(ms.float32)
    return hit


def one_hot_encode(num_classes, class_idx):
    return ops.Eye()(num_classes, num_classes, ms.float32)[class_idx]
