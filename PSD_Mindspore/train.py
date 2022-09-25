import time

import numpy as np

from models.psd import PSD
from loss import Loss
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor, Callback, LearningRateScheduler
from mindspore import context, Tensor, save_checkpoint, Model
from dataset.s3dis_dataset import S3DIS, batch_map
from config.s3dis_config import S3DISConfig as cfg
from eval import evaluate
from mindspore.train.callback import SummaryCollector
import datetime
from pathlib import Path
import os
from mindspore import ops, ParameterTuple
from utils import arg
import mindspore.numpy as msnp

# self.label_to_names = {0: 'ceiling',
#                                1: 'floor',
#                                2: 'wall',
#                                3: 'beam',
#                                4: 'column',
#                                5: 'window',
#                                6: 'door',
#                                7: 'table',
#                                8: 'chair',
#                                9: 'sofa',
#                                10: 'bookcase',
#                                11: 'board',
#                                12: 'clutter',
#                                13: 'unlabel'
#                                }

#

class ProgressCallback(Callback):
    def __init__(self):
        pass

    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        s = cb_params.cur_step_num
        print(s)


class SaveCallback(Callback):
    def __init__(self, save_dir):
        super(SaveCallback, self).__init__()
        self.save_dir = save_dir

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num > 30:
            file_name = self.save_dir / "epoch{}.ckpt".format(cb_params.cur_epoch_num)
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=str(file_name))


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, xyz, color, label, pi, ci):
        s, u, n = batch_map(xyz, 16)  # TODO: move to cfg.
        logit, embedding = self._backbone(xyz, color, s, u, n)
        self._loss_fn(logit, embedding, label)
        return self._loss_fn(logit, embedding, label)

# class GradWrap(nn.Cell):
#     """ GradWrap definition """
#     def __init__(self, network):
#         super(GradWrap, self).__init__(auto_prefix=False)
#         self.network = network
#         self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))
#
#     def construct(self, xyz, color, label, pi, ci):
#         weights = self.weights
#         grad = ops.GradOperation(get_by_list=True)(self.network, weights)(xyz, color, label, pi, ci)
#         return grad


# class TrainOneStepCell(nn.Cell):
#     def __init__(self, network, optimizer):
#         super(TrainOneStepCell, self).__init__(auto_prefix=False)
#         self.network = network
#         self.weights = ParameterTuple(network.trainable_params())
#         self.optimizer = optimizer
#         self.grad = ops.GradOperation(get_by_list=True) # sens
#
#     def construct(self, xyz, color, label, pi, ci):
#         weights = self.weights
#         loss = self.network(xyz, color, label, pi, ci) # don't understand why.
#         sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
#         grads = self.grad(self.network, weights)(xyz, color, label, pi, ci, sens) #
#         return ops.depend(loss, self.optimizer(grads))

def gen_lr_scheduler(step_num):
    def t(lr, step):
        if (step + 1) % step_num == 0:
            return lr * 0.95
        return lr
    return t


if __name__ == '__main__':
    FLAGS = arg.parse_arg()
    if FLAGS.target_platform == 0:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    if FLAGS.target_platform == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    if FLAGS.target_platform == 2:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    if FLAGS.use_modelart:
        import moxing as mox
        FLAGS.dataset_path = '/cache/dataset/'
        mox.file.copy_parallel(src_url=FLAGS.data_url, dst_url=FLAGS.dataset_path)
        

    dataset_gen = S3DIS(dataset_path=FLAGS.dataset_path,
                        area=FLAGS.test_area,
                        excl=True,
                        steps=cfg.batch_size * cfg.train_steps,
                        num_points=cfg.num_points,
                        noise_init=cfg.noise_init,
                        sub_sample_ratios=cfg.sub_sampling_ratio, labeled_point=0.01, num_classes=cfg.num_classes)
    dataset = ds.GeneratorDataset(dataset_gen,
                                  column_names=['xyz', 'color', 'label', 'pi', 'ci'])
    dataset = dataset.batch(cfg.batch_size)

    net = PSD(3, num_points=cfg.num_points, num_classes=cfg.num_classes, layer_dim=cfg.d_out)
    loss_fn = Loss(cfg.num_classes)
    loss_net = CustomWithLossCell(net, loss_fn)
    optimizer = nn.Adam(net.trainable_params(), cfg.learning_rate)

    rd = Path('cache/results/{}'.format(datetime.datetime.today().strftime('%y-%m-%d_%H-%M')))
    rd.mkdir(parents=True, exist_ok=True)

    model = Model(loss_net, loss_fn=None, optimizer=optimizer)
    model.train(cfg.max_epoch, dataset,
                callbacks=[LossMonitor(),
                           LearningRateScheduler(gen_lr_scheduler(cfg.batch_size * cfg.train_steps)),
                           SaveCallback(rd)], dataset_sink_mode=False)
    if FLAGS.use_modelart:
        mox.file.copy_parallel(src_url=str(rd), dst_url=FLAGS.train_url)

    # miou, iou, acc = evaluate(FLAGS, net, cfg.num_classes)
    # rs = ('miou:{:.2f}%\n'
    #       'iou:{}\n'
    #       'acc:{:.2f}%\n').format(miou, np.array2string(iou, precision=3), acc)
    # print(rs)

