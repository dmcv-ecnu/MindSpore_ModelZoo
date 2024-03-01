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

import datetime
import os
import argparse
import pickle
import shutil

import mindspore as ms
from mindspore import Model, Tensor, context, nn, ops
from mindspore.nn import Adam
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import SummaryCollector
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import  load_checkpoint, load_param_into_net
from src.data.dataset import ms_map
from src.utils.tools import ConfigS3DIS as cfg
from src.utils.logger import get_logger
from src.model.model import get_param_groups
import numpy as np
from pathlib import Path

def prepare_network(weights, args, logger):
    """Prepare Network"""

    d_in = 6
    bias = args.device_target == 'GPU'
    if args.mode == 'train':
        from src.model.model import RandLANet, RandLAWithLoss
        network = RandLANet(d_in, cfg.num_classes, bias=bias)
        network = RandLAWithLoss(network, weights, cfg.num_classes)
    elif args.mode == 'pretrain':
        from src.model.model_weak_pretrain import RandLANet, RandLAWithLoss
        network = RandLANet(d_in, cfg.num_classes, bias=bias)
        network = RandLAWithLoss(network, weights, cfg.num_classes, cfg.ignored_label_inds)
    elif args.mode == 'weak_train':
        from src.model.model_weak_train import RandLANet, RandLAWithLoss
        network = RandLANet(d_in, cfg.num_classes, bias=bias)
        # load ckpt
        ckpt_path = Path(os.path.join(args.model_path, 'ckpt'))
        ckpts = ckpt_path.glob('*.ckpt')
        ckpt = sorted(ckpts, key=lambda ckpt: int(ckpt.stem.split('-')[1].split('_')[0]), reverse=True)[0]
        logger.info('load ckpt from:{}'.format(str(ckpt)))
        param_dict = load_checkpoint(str(ckpt))
        param_not_load =load_param_into_net(network, param_dict)
        logger.info(param_not_load)
        network = RandLAWithLoss(network, weights, cfg.num_classes, cfg.ignored_label_inds)
        logger.info("training_epoch:{}".format(str(network.training_epoch.asnumpy())))
    return network

def prepare_dataloader(args):
    """Prepare Network"""

    if args.mode == 'train':
        from src.data.dataset import dataloader
        #data loader
        return dataloader(
            cfg.dataset,
            args,
            num_parallel_workers=8,
            shuffle=False
        )
        
    elif args.mode == 'pretrain':
        from src.data.dataset_weak_pretrain import dataloader
        #data loader
        return  dataloader(
            cfg.dataset,
            args,
            num_parallel_workers=8,
            shuffle=False
        )
       
    elif args.mode == 'weak_train':
        from src.data.dataset_weak_train import dataloader
        #data loader
        return dataloader(
            cfg.dataset,
            args,
            num_parallel_workers=8,
            shuffle=False
        )
    else:
        return None, None, None

def train(args):
#PYNATIVE_MODE GRAPH_MODE
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    logger = get_logger(args.outputs_dir, args.rank)

    for arg in vars(args):
        logger.info('%s: %s', str(arg), str(getattr(args, arg)))

    #data loader
    train_loader, _, _ = prepare_dataloader(args)

    train_loader = train_loader.batch(batch_size=args.batch_size,
                                      per_batch_map=ms_map,
                                      input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
                                      output_columns=["features", "labels", "input_inds", "cloud_inds",
                                                      "p0", "p1", "p2", "p3", "p4",
                                                      "n0", "n1", "n2", "n3", "n4",
                                                      "pl0", "pl1", "pl2", "pl3", "pl4",
                                                      "u0", "u1", "u2", "u3", "u4"],
                                      drop_remainder=True)

    logger.info('Computing weights...')

    n_samples = Tensor(cfg.class_weights, ms.float32)
    ratio_samples = n_samples / ops.ReduceSum()(n_samples)
    weights = 1 / (ratio_samples + 0.02)
    weights.expand_dims(axis=0)

    logger.info('Done')

    network = prepare_network(weights, args, logger)

    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps, is_stair=True)
    opt = Adam(
        params=get_param_groups(network),
        learning_rate=decay_lr
    )

    log = {'cur_epoch': 1, 'cur_step': 1, 'best_epoch': 1, 'besr_miou': 0.0}
    if not os.path.exists(args.outputs_dir + '/log.pkl'):
        f = open(args.outputs_dir + '/log.pkl', 'wb')
        pickle.dump(log, f)
        f.close()

    logger.info('==========begin training===============')

    #loss scale manager
    loss_scale_manager = DynamicLossScaleManager() if args.scale else None

    amp_level = 'O0' if args.device_target == 'GPU' else 'O3'
    if args.scale:
        model = Model(network,
                      amp_level=amp_level,
                      keep_batchnorm_fp32=True,
                      loss_scale_manager=loss_scale_manager,
                      loss_fn=None,
                      optimizer=opt)
    else:
        model = Model(network,
                      amp_level=amp_level,
                      keep_batchnorm_fp32=True,
                      loss_fn=None,
                      optimizer=opt)

    # callback for loss & time cost
    loss_cb = LossMonitor(50)
    time_cb = TimeMonitor(data_size=cfg.train_steps)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps=cfg.train_steps, keep_checkpoint_max=70)
    ckpt_cb = ModelCheckpoint(prefix='randla', directory=os.path.join(args.outputs_dir, 'ckpt'), config=config_ckpt)
    cbs += [ckpt_cb]

    #summary collector
    summary_collector = SummaryCollector(summary_dir=os.path.join(args.outputs_dir, 'summary'))
    cbs += [summary_collector]

    model.train(args.epochs,
            train_loader,
            callbacks=cbs,
            dataset_sink_mode=False)

    logger.info('==========end training===============')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RFCR_NL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    expr.add_argument('--epochs', type=int, help='max epochs',
                      default=100)
    expr.add_argument('--batch_size', type=int, help='batch size',
                      default=6)
    expr.add_argument('--val_area', type=str, help='area to validate',
                      default='Area_5')
    expr.add_argument('--labeled_point', type=str, default='1%', 
                      help='1, 1% or 10%')
    expr.add_argument('--scale', action='store_true', help='scale or not',
                      default=False)
    expr.add_argument('--scale_weight', type=float, help='scale weight',
                      default=1.0)
    dirs.add_argument('--outputs_dir', type=str, help='model to save',
                      default='./runs')
    misc.add_argument('--device_target', type=str, help='CPU or GPU',
                      default='GPU')
    misc.add_argument('--device_id', type=int, help='GPU id to use',
                      default=0)
    misc.add_argument('--rank', type=int, help='rank',
                      default=0)
    misc.add_argument('--name', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--mode', type=str, help='options: train, weak_pretrain, weak_train',
                    default='train')
    misc.add_argument('--gt_label_path', type=str, help='gt_label_path',
                    default=None)
    misc.add_argument('--pseudo_label_path', type=str, help='pseudo_label_path',
                    default=None)
    dirs.add_argument('--model_path', type=str, help='model saved path',
                    default='runs')
    arguments = parser.parse_args()

    if arguments.name is None:
        arguments.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    if not os.path.exists(arguments.outputs_dir):
        os.makedirs(arguments.outputs_dir)

    arguments.outputs_dir = os.path.join(arguments.outputs_dir, arguments.name)
    if not os.path.exists(arguments.outputs_dir):
        os.makedirs(arguments.outputs_dir)

    # start train
    train(arguments)
