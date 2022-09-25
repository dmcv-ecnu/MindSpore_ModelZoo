# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/06/23
    Description:

"""
import datetime, os, argparse, pickle, shutil
from pathlib import Path
import numpy as np

cu111_path = Path("/usr/local/cuda-11.1")
os.environ['CUDA_HOME'] = str(cu111_path)

print(f"Hello! Have a nice day! CUDA_HOME:{os.environ['CUDA_HOME']}")

os.environ['PATH'] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['PATH']}"

from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net, nn, ops, set_seed
from mindspore.nn import Adam
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager

from dataset.S3DIS_dataset import dataloader, ms_map
from dataset.tools import ConfigS3DIS as cfg
from dataset.tools import DataProcessing as DP
from models.base_model import get_param_groups
from models.model_s3dis import RandLANet_S3DIS, RandLA_S3DIS_WithLoss
from utils.logger import get_logger


class UpdateLossEpoch(Callback):
    def __init__(self, num_training_ep0=30, logger=None):
        super(UpdateLossEpoch, self).__init__()
        self.training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
        self.training_ep.update({i: 0 for i in range(0, num_training_ep0)})
        self.logger = logger

    def on_train_epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        train_network_with_loss = cb_params.network
        cur_epoch_num = cb_params.cur_epoch_num  # 从1开始
        train_network_with_loss.c_epoch_k += self.training_ep[cur_epoch_num - 1]
        self.logger.info(
            f"UpdateLossEpoch ==>  cur_epoch_num:{cur_epoch_num}, "
            f"cur_training_ep:{self.training_ep[cur_epoch_num]}, "
            f"loss_fn.c_epoch_k:{train_network_with_loss.c_epoch_k}")


class S3DISLossMonitor(Callback):
    def __init__(self, per_print_times=1, logger=None):
        super(S3DISLossMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.logger = logger

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss {}, terminating training."
                             "CE Loss {}; SP Loss {}".format(cb_params.cur_epoch_num, cur_step_in_epoch, loss,
                                                             cb_params.network.CE_LOSS.asnumpy(),
                                                             cb_params.network.SP_LOSS.asnumpy()))

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -= \
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            self.train_network_with_loss = cb_params.network

            if isinstance(self.train_network_with_loss, Tensor):
                msg = f"epoch: {cb_params.cur_epoch_num} step: {cur_step_in_epoch}, " \
                      f"loss is {loss} (CE Loss:{self.train_network_with_loss.CE_LOSS.asnumpy()}; SP Loss:{self.train_network_with_loss.SP_LOSS.asnumpy()})"
            else:
                msg = f"epoch: {cb_params.cur_epoch_num} step: {cur_step_in_epoch}, " \
                      f"loss is {loss} (CE Loss:{self.train_network_with_loss.CE_LOSS}; SP Loss:{self.train_network_with_loss.SP_LOSS})"
                # f"loss is {loss} (CE Loss:{self.train_network_with_loss.CE_LOSS.asnumpy()}; SP Loss:{self.train_network_with_loss.SP_LOSS.asnumpy()})"
            #  self.train_network_with_loss.CE_LOSS.dtype == Parameters
            #  self.train_network_with_loss.SP_LOSS.dtype == Parameters
            self.logger.info(msg)


def prepare_network(weights, cfg, args):
    """Prepare Network"""

    d_in = 6  # xyzrgb
    network = RandLANet_S3DIS(d_in, cfg.num_classes)
    if args.ss_pretrain:
        print(f"Load scannet pretrained ckpt from {args.ss_pretrain}")
        param_dict = load_checkpoint(args.ss_pretrain)
        whitelist = ["encoder"]
        load_all = True
        new_param_dict = dict()
        for key, val in param_dict.items():
            if key.split(".")[0] == 'network' and key.split(".")[1] in whitelist:
                new_key = ".".join(key.split(".")[1:])
                new_param_dict[new_key] = val
        load_param_into_net(network, new_param_dict, strict_load=True)

    network = RandLA_S3DIS_WithLoss(network, weights, cfg.num_classes, cfg.ignored_label_indexs, cfg.c_epoch, cfg.loss3_type, cfg.topk)

    if args.retrain_model:
        print(f"Load S3DIS pretrained ckpt from {args.retrain_model}")
        param_dict = load_checkpoint(args.retrain_model)
        load_param_into_net(network, param_dict, strict_load=True)

    return network


def train(cfg, args):
    if cfg.graph_mode:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)

    logger = get_logger(args.outputs_dir, args.rank)

    logger.info("============ Args =================")
    for arg in vars(args):
        logger.info('%s: %s' % (arg, getattr(args, arg)))
    logger.info("============ Cfg =================")
    for c in vars(cfg):
        logger.info('%s: %s' % (c, getattr(cfg, c)))

    train_loader, val_loader, dataset = dataloader(cfg, shuffle=False, num_parallel_workers=8)
    ignored_label_indexs = [getattr(dataset, 'label_to_idx')[ign_label] for ign_label in getattr(dataset, 'ignored_labels')]
    cfg.ignored_label_indexs = ignored_label_indexs
    weights = DP.get_class_weights("S3DIS")
    network = prepare_network(weights, cfg, args)

    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps, is_stair=True)
    opt = Adam(
        params=get_param_groups(network),
        learning_rate=decay_lr,
        loss_scale=cfg.loss_scale
    )

    log = {'cur_epoch': 1, 'cur_step': 1, 'best_epoch': 1, 'besr_miou': 0.0}
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

    # data loader

    train_loader = train_loader.batch(batch_size=cfg.batch_size,
                                      per_batch_map=ms_map,
                                      input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
                                      output_columns=["features", "features2", "labels", "input_inds", "cloud_inds",
                                                      "p0", "p1", "p2", "p3", "p4",
                                                      "n0", "n1", "n2", "n3", "n4",
                                                      "pl0", "pl1", "pl2", "pl3", "pl4",
                                                      "u0", "u1", "u2", "u3", "u4"],
                                      drop_remainder=True)

    logger.info('==========begin training===============')

    # loss scale manager
    loss_scale = cfg.loss_scale
    # loss_scale = args.scale_weight
    loss_scale_manager = FixedLossScaleManager(loss_scale) if args.scale or loss_scale != 1.0 else None
    print('loss_scale:', loss_scale)

    if args.scale:
        model = Model(network,
                      loss_scale_manager=loss_scale_manager,
                      loss_fn=None,
                      optimizer=opt)
    else:
        model = Model(network,
                      loss_fn=None,
                      optimizer=opt)

    # callback for loss & time cost
    loss_cb = S3DISLossMonitor(20, logger)
    time_cb = TimeMonitor(data_size=cfg.train_steps)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps=cfg.train_steps, keep_checkpoint_max=100)
    ckpt_cb = ModelCheckpoint(prefix='randla', directory=os.path.join(args.outputs_dir, 'ckpt'), config=config_ckpt)
    cbs += [ckpt_cb]

    update_loss_epoch_cb = UpdateLossEpoch(args.num_training_ep0, logger)
    cbs += [update_loss_epoch_cb]

    logger.info(f"Outputs_dir:{args.outputs_dir}")
    logger.info(f"Total number of epoch: {cfg.max_epoch}; "
                f"Dataset capacity: {train_loader.get_dataset_size()}")

    model.train(cfg.max_epoch,
                train_loader,
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

    expr.add_argument('--epochs', type=int, help='max epochs', default=100)

    expr.add_argument('--batch_size', type=int, help='batch size', default=6)

    expr.add_argument('--val_area', type=str, help='area to validate', default='Area_5')

    expr.add_argument('--resume', type=str, help='model to resume', default=None)

    expr.add_argument('--scale', type=bool, help='scale or not', default=False)

    # expr.add_argument('--scale_weight', type=float, help='scale weight', default=1.0)

    misc.add_argument('--device_target', type=str, help='CPU or GPU', default='GPU')

    misc.add_argument('--device_id', type=int, help='GPU id to use', default=0)

    misc.add_argument('--rank', type=int, help='rank', default=0)

    misc.add_argument('--name', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--ss_pretrain', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--retrain_model', type=str, help='name of the experiment',
                      default=None)
    misc.add_argument('--train_steps', type=int, default=500)
    misc.add_argument('--learning_rate', type=float, default=0.01)
    misc.add_argument('--lr_decays', type=float, default=0.95)
    misc.add_argument('--loss_scale', type=float, default=1.0)
    misc.add_argument('--topk', type=int, default=500)
    misc.add_argument('--num_training_ep0', type=int, default=30)
    misc.add_argument('--labeled_percent', type=int, default=1)  # range in [1,100]
    misc.add_argument('--random_seed', type=int, default=888)
    misc.add_argument('--graph_mode', action='store_true', default=False)

    args = parser.parse_args()

    cfg.batch_size = args.batch_size
    cfg.max_epoch = args.epochs
    cfg.train_steps = args.train_steps
    cfg.learning_rate = args.learning_rate
    cfg.lr_decays = args.lr_decays
    cfg.loss_scale = args.loss_scale
    cfg.topk = args.topk
    num_training_ep0 = args.num_training_ep0
    cfg.training_ep0 = {i: 0 for i in range(0, num_training_ep0)}
    cfg.training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
    cfg.training_ep.update(cfg.training_ep0)
    cfg.labeled_percent = args.labeled_percent
    cfg.random_seed = args.random_seed
    cfg.graph_mode = args.graph_mode

    if args.name is None:
        if args.resume:
            args.name = Path(args.resume).split('/')[-1]
        else:
            time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
            args.name = f'TSteps{cfg.train_steps}_MaxEpoch{cfg.max_epoch}_BatchS{cfg.batch_size}_lr{cfg.learning_rate}' \
                        f'_lrd{cfg.lr_decays}_ls{cfg.loss_scale}_Topk{cfg.topk}_NumTrainEp0{num_training_ep0}_LP_{cfg.labeled_percent}_RS_{cfg.random_seed}'
            if cfg.graph_mode:
                args.name += "_GraphM"
            else:
                args.name += "_PyNateiveM"
            args.name += f'_{time_str}'

    np.random.seed(cfg.random_seed)
    set_seed(cfg.random_seed)  # https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.set_seed.html?highlight=set_seed

    output_dir = f"./runs/s3dis_model"
    args.outputs_dir = os.path.join(output_dir, args.name)

    print(f"outputs_dir:{args.outputs_dir}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    if args.resume:
        args.outputs_dir = args.resume

    # copy file
    shutil.copy('dataset/tools.py', str(args.outputs_dir))
    shutil.copy('models/model_s3dis.py', str(args.outputs_dir))
    shutil.copy('dataset/S3DIS_dataset.py', str(args.outputs_dir))
    shutil.copy('train_s3dis.py', str(args.outputs_dir))

    # start train
    train(cfg, args)
