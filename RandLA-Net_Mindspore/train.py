"""
Author: Qihang Ma
Date: Sep 2022
"""
import datetime
import os
import time
import argparse
import pickle
from pathlib import Path

import mindspore as ms
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net, nn
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import Adam
from mindspore.profiler.profiling import Profiler
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.ops import ops
from mindspore.train.callback import SummaryCollector
from mindspore.train.loss_scale_manager import DynamicLossScaleManager


from dataset import dataloader, ms_map
from utils.tools import ConfigS3DIS as cfg
from utils.logger import get_logger
from utils.metrics import accuracy, intersection_over_union
from model import RandLANet, RandLAWithLoss, TrainingWrapper, get_param_groups


def prepare_network(weights, cfg):
    """Prepare Network"""

    d_in = 6
    network = RandLANet(d_in, cfg.num_classes)
    network = RandLAWithLoss(network, weights, cfg.num_classes)

    return network


def evaluate(network, loader, logger):
    network = network.network
    network.set_train(False)
    loader = loader.batch(batch_size = cfg.val_batch_size,
                          per_batch_map=ms_map,
                          input_columns=["xyz","colors","labels","q_idx","c_idx"],
                          output_columns=["features","labels","input_inds","cloud_inds",
                                        "p0","p1","p2","p3","p4",
                                        "n0","n1","n2","n3","n4",
                                        "pl0","pl1","pl2","pl3","pl4",
                                        "u0","u1","u2","u3","u4"], 
                          drop_remainder=True)
    loader = loader.create_dict_iterator()
    losses = []
    accuracies = []
    ious = []
    logger.info('validating')
    for i, data in enumerate(loader):
        features = data['features']
        labels = data['labels']
        xyz = [data['p0'],data['p1'],data['p2'],data['p3'],data['p4']]
        neigh_idx = [data['n0'],data['n1'],data['n2'],data['n3'],data['n4']]
        sub_idx = [data['pl0'],data['pl1'],data['pl2'],data['pl3'],data['pl4']]
        interp_idx = [data['u0'],data['u1'],data['u2'],data['u3'],data['u4']]

        if i%50 == 0:
            logger.info(i, ' / ', cfg.val_batch_size)
        loss, logits = network(xyz, features, neigh_idx, sub_idx, interp_idx, labels)
        losses.append(loss.asnumpy())
        accuracies.append(accuracy(scores, labels))
        ious.append(intersection_over_union(scores, labels))
    
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)
        


def train(args):

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)

    logger = get_logger(args.outputs_dir, args.rank)

    logger.info('Computing weights...')

    n_samples = Tensor(cfg.class_weights, ms.float32)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples + 0.02)
    weights.expand_dims(axis=0)

    logger.info('Done')
    #logger.info('weights:',weights)

    network = prepare_network(weights, cfg)

    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps)
    opt = Adam(
        params = get_param_groups(network),
        learning_rate = decay_lr,
        loss_scale = cfg.loss_scale
    )

    #network = TrainingWrapper(network, opt, cfg.loss_scale)

    log = {'cur_epoch':1,'cur_step':1,'best_epoch':1,'best_miou':0.0}
    if not os.path.exists(args.outputs_dir + '/log.pkl'):
        f = open(args.outputs_dir + '/log.pkl', 'wb')
        pickle.dump(log, f)
        f.close()

    # resume checkpoint, cur_epoch, best_epoch, cur_step, best_step
    if args.resume:
        f = open(args.resume + '/log.pkl', 'rb')
        log = pickle.load(f)
        f.close()
        param = load_checkpoint(args.resume)
        load_param_into_net(network, args.resume)

    #data loader
    train_loader, val_loader, dataset = dataloader(
        cfg.dataset,
        args.val_area,
        num_parallel_workers=8,
        shuffle=False
    )

    train_loader = train_loader.batch(batch_size = args.batch_size,
                                      per_batch_map=ms_map,
                                      input_columns=["xyz","colors","labels","q_idx","c_idx"],
                                      output_columns=["features","labels","input_inds","cloud_inds",
                                                  "p0","p1","p2","p3","p4",
                                                  "n0","n1","n2","n3","n4",
                                                  "pl0","pl1","pl2","pl3","pl4",
                                                  "u0","u1","u2","u3","u4"],
                                      drop_remainder=True)
    train_loader = train_loader.create_dict_iterator()
    
    begin_epoch = log['cur_epoch']
    logger.info('==========begin training===============')
    model = Model(network, loss_fn=None, optimizer=opt)

    # callback for loss & time cost
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=cfg.train_steps)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps= cfg.train_steps, keep_checkpoint_max=40)
    ckpt_cb = ModelCheckpoint(prefix='randla', directory=os.path.join(args.outputs_dir, 'ckpt'), config=config_ckpt)
    cbs += [ckpt_cb]

    #summary collector
    summary_collector = SummaryCollector(summary_dir=os.path.join(args.outputs_dir, 'summary'))
    cbs += [summary_collector]

    #loss scale manager
    loss_scale_manager = DynamicLossScaleManager()
    model.train(args.epochs, 
                train_loader, 
                loss_scale_manager=loss_scale_manager,
                callbacks=cbs, 
                dataset_sink_mode=False)

    logger.info('==========end training===============')


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    expr.add_argument('--epochs', type=int, help='max epochs',
                        default=100)

    expr.add_argument('--batch_size', type=int, help='batch size',
                        default=4)

    expr.add_argument('--val_area', type=str, help='area to validate',
                        default='Area_5')

    expr.add_argument('--resume', type=str, help='model to resume',
                        default=None)

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

    args = parser.parse_args()

    if args.name is None:
        if args.resume:
            args.name = Path(args.resume).split('/')[-1]
        else:
            args.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    args.outputs_dir = os.path.join(args.outputs_dir, args.name)
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    
    if args.resume:
        args.outputs_dir = args.resume
    
    # start train
    train(args)






    




