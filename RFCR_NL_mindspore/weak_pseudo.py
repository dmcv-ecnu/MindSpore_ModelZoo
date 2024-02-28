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
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from os.path import join
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, ops

from src.data.dataset_weak_pseudo import dataloader, ms_map
from src.utils.tools import ConfigS3DIS as cfg
from src.utils.tools import DataProcessing as DP
from src.utils.logger import get_logger
from src.utils.helper_ply import write_ply
from src.model.model_weak_pretrain import RandLANet


def run_eval(args):
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    logger = get_logger(args.outputs_dir, args.rank)
    logger.info('Computing weights...')

    n_samples = Tensor(cfg.class_weights, ms.float32)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples + 0.02)
    weights.expand_dims(axis=0)

    logger.info('Done')

    # data loader
    _, val_ds, dataset = dataloader(
        cfg.dataset,
        args,
        num_parallel_workers=8,
        shuffle=False
    )

    val_loader = val_ds.batch(batch_size=args.batch_size,
                              per_batch_map=ms_map,
                              input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
                              output_columns=["features", "labels", "input_inds", "cloud_inds",
                                              "p0", "p1", "p2", "p3", "p4",
                                              "n0", "n1", "n2", "n3", "n4",
                                              "pl0", "pl1", "pl2", "pl3", "pl4",
                                              "u0", "u1", "u2", "u3", "u4"],
                              drop_remainder=True)
    val_loader = val_loader.create_dict_iterator()
    # print(val_ds.source.__dict__)
    # exit()

    # load ckpt, iterate ckpts to find the best
    d_in = 6
    bias = args.device_target == 'GPU'
    network = RandLANet(d_in, cfg.num_classes, bias=bias)
    network.set_train(False)
    if not bias:
        network.to_float(ms.float16)

    ckpt_path = Path(os.path.join(args.model_path, 'ckpt'))
    ckpts = ckpt_path.glob('*.ckpt')
    ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.stem.split('-')[1].split('_')[0]), reverse=True)[:1]
    best_miou = 0.0
    best_ckpt = ckpts[-1]
    best_names = []
    best_preds = []
    best_labels = []
    for _, ckpt in enumerate(ckpts): 
        # load current ckpt
        logger.info('load ckpt from:{}'.format(str(ckpt)))
        param_dict = load_checkpoint(str(ckpt))
        param_not_load =load_param_into_net(network, param_dict)
        logger.info(param_not_load)

        # Number of points per class in validation set
        val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1
        test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32)
                      for l in dataset.input_labels['validation']]

        # Smoothing parameter for votes
        test_smooth = 0.95
        
        logger.info('==========begin test===============')

        for step_i, data in enumerate(val_loader):
            print('step:', str(step_i))
            features = data['features']
            labels = data['labels']
            xyz = [data['p0'], data['p1'], data['p2'], data['p3'], data['p4']]
            neigh_idx = [data['n0'], data['n1'], data['n2'], data['n3'], data['n4']]
            sub_idx = [data['pl0'], data['pl1'], data['pl2'], data['pl3'], data['pl4']]
            interp_idx = [data['u0'], data['u1'], data['u2'], data['u3'], data['u4']]
            point_idx = data['input_inds'].asnumpy()
            cloud_idx = data['cloud_inds'].asnumpy()

            logits, _, = network(xyz, features, neigh_idx, sub_idx, interp_idx, labels)  # [b, num_classes, N]
            logits = logits.swapaxes(-2, -1)  # [b, num_classes, N] --> [b, N, num_classes]
            stacked_probs = ops.Softmax(-1)(logits).asnumpy()
            for j in range(np.shape(stacked_probs)[0]):
                probs = stacked_probs[j, :, :]
                p_idx = point_idx[j, :]
                c_i = cloud_idx[j][0]
                test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs
        best_ckpt, best_miou, best_names, best_preds, best_labels = cal_metric(best_ckpt, best_miou, best_names,
                                                                               best_preds, best_labels, ckpt,
                                                                               dataset, logger, test_probs,
                                                                               val_ds, val_proportions, cfg.threshold, cfg.span, args.outputs_dir)

    

def cal_metric(b_c, b_miou, b_n, b_p, b_l, ckpt, dataset, logger, test_probs, val_ds, val_proportions, threshold, span, test_path):
    last_min = -0.5
    num_votes = 100

    while last_min < num_votes:
        new_min = np.min(val_ds.source.min_possibility['validation'])

        if last_min + 1 < new_min:

            # Update last_min
            last_min += 1

            # Show vote results (On subcloud so it is not the good values here)
            logger.info('Confusion on sub clouds')
            confusion_list = []

            num_val = len(dataset.input_labels['validation'])

            sum_point = 0  #记录取伪标签的个数及比例
            pseudo_point = 0
            sum_correct = 0
            sum_all = 0
            for i_test in range(num_val):
                probs = test_probs[i_test]
                probs2 = np_normalized(probs)
                # dynamic threshold
                p_t = np.max(probs2, axis=0) - span
                # print(p_t.min(),p_t.max())
                p_t[np.argwhere(p_t < threshold - span)] = threshold - span
                # print(p_t.min(),p_t.max())
                pt_c = np.repeat(np.reshape(p_t, [1, -1]), repeats=probs2.shape[0], axis=0)
                probs2 = probs2 - pt_c
                # print(probs2.min(),probs2.max())
                uncertainty_idx = np.argwhere(np.max(probs2, axis=1) < 0)
                # uncertainty_idx = np.argwhere(np.max(probs2, axis=1) < 0.7)
                uncertainty_idx = np.reshape(uncertainty_idx, [-1])
                sum_point += probs2.shape[0]
                pseudo_point += uncertainty_idx.shape[0]

                preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                # 对于置信度不够高的点，仍置为unlabel
                for id in uncertainty_idx :
                    preds[id] = 13

                labels = dataset.input_labels['validation'][i_test]
                labels_p  = np.delete(labels, uncertainty_idx)
                preds_p = np.delete(preds, uncertainty_idx)
                correct_preds = np.sum(labels_p == preds_p)
                sum_preds = np.prod(np.shape(preds_p))
                acc_preds = correct_preds / float(sum_preds)
                logger.info('proportion = {:f}, acc_pred = {:.3f}'.format(1 - uncertainty_idx.shape[0] / probs2.shape[0], acc_preds))
                sum_correct += correct_preds
                sum_all += sum_preds
                # Confs
                confusion_list += [confusion_matrix(labels_p, preds_p, labels=dataset.label_values)]
                
                cloud_name = dataset.input_names['validation'][i_test]
                # ascii_name = join(test_path, 'pseudo_label', cloud_name + '.txt')
                # np.savetxt(ascii_name, preds, fmt='%d')
                ascii_name = join(test_path, 'pseudo_label', cloud_name)
                np.save(ascii_name, preds)
                logger.info(ascii_name + ' has saved')
            # Regroup confusions
            C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

            # Rescale with the right number of point per class
            C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

            # Compute IoUs
            IoUs = DP.IoU_from_confusions(C)
            m_IoU = np.mean(IoUs)
            s = '{:5.2f} | '.format(100 * m_IoU)
            for IoU in IoUs:
                s += '{:5.2f} '.format(100 * IoU)
            logger.info(s + '\n')
            pseudo_point = sum_point - pseudo_point
            logger.info('sum: {:d}, pseudo labels: {:d}, proportion = {:f}'.format(sum_point, pseudo_point, pseudo_point / sum_point))
            logger.info('sum correct: {:d}, all: {:d}, correct acc:  = {:.3f}'.format(sum_correct, sum_all, sum_correct / sum_all))
            return b_c, b_miou, b_n, b_p, b_l
    return b_c, b_miou, b_n, b_p, b_l

def np_normalized(z):
    a = z / np.reshape(np.sum(z, axis=1), [-1,1])
    return a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RFCR_NL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    expr.add_argument('--batch_size', type=int, help='val batch size',
                      default=20)
    expr.add_argument('--val_area', type=str, help='area to validate',
                      default='Area_5')
    dirs.add_argument('--model_path', type=str, help='model saved path',
                      default='runs')
    misc.add_argument('--device_target', type=str, help='CPU or GPU',
                      default='GPU')
    misc.add_argument('--device_id', type=int, help='GPU id to use',
                      default=0)
    misc.add_argument('--rank', type=int, help='rank',
                      default=0)
    expr.add_argument('--labeled_point', type=str, default='1%', help='1, 1% or 10%')
    arguments = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    arguments.model_path = os.path.join(base_dir, arguments.model_path)

    # test output dir
    test_log_path = os.path.join(arguments.model_path, 'prediction')
    test_save_path = os.path.join(test_log_path, 'val_preds')
    arguments.outputs_dir = test_log_path
    os.makedirs(test_save_path) if not os.path.exists(test_save_path) else None
    # val pred path
    if not os.path.exists(os.path.join(arguments.outputs_dir, 'pseudo_label')):
        os.makedirs(os.path.join(arguments.outputs_dir, 'pseudo_label'))
    # start test
    run_eval(arguments)
    