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
import mindspore.numpy as msnp
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, nn
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import Adam
from mindspore.profiler.profiling import Profiler
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import RunContext
from mindspore.train.callback import _InternalCallbackParam


from dataset import dataloader, ms_map
from utils.tools import ConfigS3DIS as cfg
from utils.logger import get_logger
from utils.metrics import accuracy, intersection_over_union
from model_train_val import RandLANet, weight_ce_loss, RandLAWithLoss, TrainingWrapper, get_param_groups


def evaluate(network, loader, logger):
    network = network.network_logits
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
    accuracies = 0
    ious = 0
    print('validating')
    for i, data in enumerate(loader):
        features = data['features']
        labels = data['labels']
        xyz = [data['p0'],data['p1'],data['p2'],data['p3'],data['p4']]
        neigh_idx = [data['n0'],data['n1'],data['n2'],data['n3'],data['n4']]
        sub_idx = [data['pl0'],data['pl1'],data['pl2'],data['pl3'],data['pl4']]
        interp_idx = [data['u0'],data['u1'],data['u2'],data['u3'],data['u4']]

        if i%50 == 0:
            print(str(i), ' / ', str(cfg.val_steps))
        logits = network(xyz, features, neigh_idx, sub_idx, interp_idx)
        if i==0:
            accuracies = accuracy(logits, labels).expand_dims(0)
            ious = intersection_over_union(logits, labels).expand_dims(0)
        else:
            accuracies = msnp.append(accuracies, accuracy(logits, labels).expand_dims(0), axis=0)
            ious = msnp.append(ious, intersection_over_union(logits, labels).expand_dims(0), axis=0)
    
    return msnp.nanmean(accuracies, axis=0).asnumpy(), msnp.nanmean(ious, axis=0).asnumpy()
        


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

    d_in = 6
    network = RandLANet(d_in, cfg.num_classes)

    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps)
    opt = Adam(
        params = get_param_groups(network),
        learning_rate = decay_lr,
        loss_scale = cfg.loss_scale
    )

    loss_fn = weight_ce_loss(weights, cfg.num_classes)
    network = RandLAWithLoss(network, loss_fn)
    network = TrainingWrapper(network, opt, cfg.loss_scale)

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
    for epoch in range(begin_epoch, cfg.max_epoch+1):
        if epoch is not 1:
            logger.info('best epoch {} , best mIou {:.3f}\n'.format(log['best_epoch'], log['best_miou']))
        logger.info('=== EPOCH {:d}/{:d} ==='.format(log['cur_epoch'], cfg.max_epoch))
        t0 = time.time()
        
        #train
        network.set_train()

        # iterate over dataset
        begin_step = log['cur_step']
        for i, data in enumerate(train_loader, begin_step):
            features = data['features']
            labels = data['labels']
            xyz = [data['p0'],data['p1'],data['p2'],data['p3'],data['p4']]
            neigh_idx = [data['n0'],data['n1'],data['n2'],data['n3'],data['n4']]
            sub_idx = [data['pl0'],data['pl1'],data['pl2'],data['pl3'],data['pl4']]
            interp_idx = [data['u0'],data['u1'],data['u2'],data['u3'],data['u4']]

            loss, logits = network(xyz, features, neigh_idx, sub_idx, interp_idx, labels)
            if i%50==0:
                logger.info('step {:d} loss {}'.format(log['cur_step'], str(loss)))
            
            log['cur_step'] += 1
            save_path = os.path.join(args.outputs_dir, 'cur_model.ckpt')
            ms.save_checkpoint(network, save_path)

        val_accs, val_ious = evaluate(
            network,
            val_loader,
            logger
        )
        
        # save best val_iou ckpt
        cur_miou = val_ious[-1]
        if cur_miou > log['best_miou']:
            log['best_epoch'] = log['cur_epoch']
            log['best_miou'] = cur_miou*100
            best_save_path = os.path.join(args.outputs_dir, 'best_epoch_{:d}_miou_{:.1f}.ckpt'.format(log['best_epoch'], log['best_miou']))
            ms.save_checkpoint(network, best_save_path)

        t1 = time.time()
        d = t1 - t0
        # Display results
        mean_iou = cur_miou*100
        
        logger.info('eval accuracy: {:f}'.format(val_accs[-1]))
        logger.info('mean IOU: {:f}'.format(cur_miou))
        logger.info('Mean IoU: {:.1f}%'.format(mean_iou))
        s = '{:5.2f} | '.format(mean_iou)
        for i, IoU in enumerate(val_ious):
            if i < cfg.num_classes:
                s += '{:5.2f} '.format(100 * IoU)
        logger.info('-' * len(s))
        logger.info(s)
        logger.info('-' * len(s) + '\n')
        logger.info('Time elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)))

        log['cur_step'] = 0
        log['cur_epoch'] += 1

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






    




