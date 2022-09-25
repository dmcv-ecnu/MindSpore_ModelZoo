from sklearn.metrics import confusion_matrix
import numpy as np
from config import cfg
import argparse
from dataset.s3dis_dataset import S3DIS, batch_map
import mindspore.dataset as ds
from mindspore import Tensor, dtype, ops
from models.psd import PSD
from utils import arg


def evaluate(FLAGS, model, num_classes):
    model.set_train(False)
    dataset = S3DIS(dataset_path=FLAGS.dataset_path,
                    area=FLAGS.test_area,
                    excl=False,
                    steps=cfg.val_steps * cfg.val_batch_size,
                    num_points=cfg.num_points,
                    noise_init=cfg.noise_init,
                    sub_sample_ratios=cfg.sub_sampling_ratio,
                    labeled_point=0, num_classes=cfg.num_classes)

    val_proportions = dataset.val_proportion
    val_batch_size = 16  # batch_size during validation and test

    val_dataloader = ds.GeneratorDataset(dataset,
                                         column_names=['xyz', 'color', 'label', 'pi', 'ci'])
    val_dataloader = val_dataloader.batch(cfg.batch_size)

    real_prob = [np.full((i.shape[0], num_classes), 1.0 / num_classes) for i in dataset.labels]
    smooth = 0.95

    for batch, (xyz, color, label, pi, ci) in enumerate(val_dataloader):
        s, u, n = batch_map(xyz, 16) # TODO: cfg.

        pred, _ = model(xyz, color, s, u, n)
        pred = ops.Softmax(1)(pred)
        pred = ops.transpose(pred, [0, 2, 1]).asnumpy()

        for i in range(val_batch_size):
            probs = pred[i, :, :]
            p_idx = pi[i]
            c_idx = ci[i]
            real_prob[c_idx][p_idx] = smooth * real_prob[c_idx][p_idx] + (1 - smooth) * probs
    real_pred = [np.argmax(i, 1) for i in real_prob]

    m = np.zeros((num_classes, num_classes))
    for i, v in enumerate(real_pred):
        m += confusion_matrix(dataset.labels[i], v, labels=np.arange(0, num_classes, 1))  # tensor to np.
    m *= np.expand_dims(val_proportions / (np.sum(m, axis=1) + 1e-6), 1) # scale.

    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)
    d = np.diagonal(m)
    iou = d.astype(np.float32) / (r + c - d + 1e-6).astype(np.float32)
    acc = d.astype(np.float32) / np.sum(r).astype(np.float32)
    return np.mean(iou), iou, acc


# if __name__ == '__main__':
#     FLAGS = arg.parse_arg()
#     os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#
#     net = PSD(3, cfg.num_points, cfg.num_classes, cfg.d_out)
#
#     miou, iou, acc = evaluate(FLAGS, net, cfg.num_classes)
#     rs = ('miou:{:.2f}%\n'
#           'iou:{}\n'
#           'acc:{:.2f}%\n').format(miou, np.array2string(iou, precision=3), acc)
#     print(rs)
