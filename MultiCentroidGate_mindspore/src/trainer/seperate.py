import logging
from args import IncrementalConfig 
import factory
import numpy as np
from ds.incremental import IncrementalDataset
# from models.loss import AutoKD, BCEWithLogitTarget
# from models.loss.t import DivLoss
from rehearsal.memory_size import MemorySize
from scipy.spatial.distance import cdist
from tqdm import tqdm

import utils
# from utils.model_utils import extract_features

from .utils import build_exemplar_broadcast, collate_result
# from .utils import build_exemplar, build_exemplar_broadcast, collate_result

from utils import AverageMeter, MultiAverageMeter  # 下行待补充，未做修改
# from utils import ClassErrorMeter, AverageMeter, MultiAverageMeter, log_dict

# from models import *
from collections import defaultdict
# from ds.incremental import WeakStrongDataset
from ds.dataset_transform import dataset_transform
import os.path
from pathlib import Path
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.train import Model
from mindspore.profiler import Profiler
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from models.loss import aux_loss

# Constants
EPSILON = 1e-8
logger = logging.getLogger() 

class IncModel():
    def __init__(self, cfg: IncrementalConfig, inc_dataset):
        super(IncModel, self).__init__()
        self.cfg = cfg
        self._inc_dataset: IncrementalDataset = inc_dataset
        self._n_epochs = None
        self._memory_size = MemorySize(cfg["mem_size_mode"], cfg["memory_size"], cfg["fixed_memory_per_cls"])
        self._coreset_strategy = cfg["coreset_strategy"]
        self._network = factory.create_network(cfg)
        self._parallel_network = None
        self._old_model = None
        try:
            global experts_trainer_cfg, gate_trainer_cfg 
            self.et, self.ef = experts_trainer_cfg[self.cfg.subtrainer] 
        except Exception as e:
            print(e)         

    
    def before_task(self):
        # dist.barrier()  # 分布式，先放弃  Soap
        self._memory_size.update_nb_classes(self.cfg.nb_seen_classes)
        self._network.add_classes(self.cfg.nb_task_classes)
        if self.cfg.pretrain_model_dir is not None:  # 先跳过，因为没有预训练模型，无法load Soap
            self.load() 
        # if self.cfg["syncbn"]:  # 算子疑似缺失，先不转换成syncbn  Soap
        #     self._network = nn.SyncBatchNorm.convert_sync_batchnorm(self._network)
        # dist.barrier()  # 分布式，先放弃  Soap
            
        '''self._parallel_network = DDP(self._network,
                                    device_ids=[self.cfg.gpu],
                                    output_device=self.cfg.gpu,
                                    find_unused_parameters=True,
                                    broadcast_buffers=True) 
        self._parallel_network = self._parallel_network.cuda()'''
        self._parallel_network = self._network  # 没有分布式，直接恒等
        self._parallel_network.to_float(mindspore.float16)
        self._parallel_network.der.classifier.to_float(mindspore.float32)
        if self.cfg.idx_task > 0:
            self._parallel_network.der.aux_classifier.to_float(mindspore.float32)

        # dist.barrier()  # 分布式，先放弃  Soap

    def train_task(self, train_loader, val_loader):  
        if self.cfg.part_enable[0] == 1: self.train_experts(train_loader)
        if self.cfg.part_enable[1] == 1 and (self.cfg.idx_task != 0 or self.cfg.force_ft_first): self.ft_experts(train_loader)
    
    def train_experts(self, train_loader):
        self._parallel_network.set_train()  # ms的.train()
        # 冻结老模型的backbone
        # self._parallel_network.module.freeze("old")  torch分布式写法
        self._parallel_network.freeze("old")

        # 设置老模型的backbone为eval，新模型的为train
        # 自定义的set_train，修改为set_model_train
        # self._parallel_network.module.set_train("experts")  torch分布式写法
        self._parallel_network.set_model_train("experts")

        # dist.barrier()  # 分布式，先放弃  Soap
        
        scaler = None
        if self.cfg.amp:
            scaler = DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)

        # parameters = self._parallel_network.module.param_groups()['experts']  torch分布式写法
        parameters = self._parallel_network.param_groups()['experts']
        
        parameters = filter(lambda x: x.requires_grad, parameters)

        optimizer = nn.SGD(parameters,
                           learning_rate=self.cfg.lr,
                           momentum=self.cfg.momentum,
                           weight_decay=self.cfg.weight_decay)
        
        # 疑似不需要
        # scheduler, n_epochs = create_scheduler(self.cfg, optimizer) 
        n_epochs = self.cfg.epochs

        # with tqdm(range(n_epochs), disable=not utils.is_main_process()) as pbar:
        with tqdm(range(n_epochs)) as pbar:  # 没有进行分布式计算  Soap
            for e in pbar: 
                # train_loader.sampler.set_epoch(e)    # 没有进行分布式计算  Soap
                loss = self.et(self.cfg,
                               self._parallel_network,
                               self._old_model,
                               train_loader,
                               optimizer, 
                               scaler)
                # scheduler.step(e)  # 暂时不进行混合精度计算  Soap
                # if utils.is_main_process():  # 没有进行分布式计算  Soap
                pbar.set_description(f"E {e} expert loss: {loss:.3f}") 

    def ft_experts(self, train_loader):
        dataset = self._inc_dataset.get_custom_dataset("train", "train", True)  
        train_loader = factory.create_dataloader(self.cfg, dataset, self.cfg.distributed, True)

        # self._network.reset("experts_classifier")  # 原  Soap
        self._parallel_network.reset("experts_classifier")
        # self._network.freeze("backbone")  # 原  Soap
        self._parallel_network.freeze("backbone")

        # 不进行分布式  Soap
        # self._parallel_network = DDP(self._network.cuda(),
        #                         device_ids=[self.cfg.gpu],
        #                         output_device=self.cfg.gpu,
        #                         find_unused_parameters=True).cuda()
        # dist.barrier()

        self._parallel_network.set_train(False)
        optim = nn.SGD(self._parallel_network.param_groups()["experts_classifier"],
                       learning_rate=self.cfg.ft.lr,
                       momentum=self.cfg.ft.momentum,
                       weight_decay=self.cfg.ft.weight_decay)        

        print(self.cfg.ft.epochs)
        scaler = None
        if self.cfg.amp:
            scaler = DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)
        # with tqdm(range(self.cfg.ft.epochs), disable=not utils.is_main_process()) as pbar:
        with tqdm(range(self.cfg.ft.epochs)) as pbar:  # 没有进行分布式计算  Soap
            for i in pbar:
                # train_loader.sampler.set_epoch(i)  # 不进行分布式训练  Soap
                loss = self.ef(
                    self.cfg,
                    self._parallel_network,
                    train_loader,
                    optim,
                    scaler)
                # if utils.is_main_process():  # 不进行分布式训练  Soap
                pbar.set_description(f"Epoch {i} expert finetuning loss {loss.asnumpy().item(): .3f}")
                # sched.step(i)  # 不进行混合精度  Soap

    def after_task(self): 
        self._parallel_network.set_train(False)
        if self.cfg.coreset_feature == "last":
            # fn = lambda x: self._network(x)['feature'][:, -512:]  # 不懂为什么这么做，可能和分布式有关  Soap
            fn = lambda x: self._parallel_network(x)['feature'][:, -512:]
        elif self.cfg.coreset_feature == "all":
            # fn = lambda x: self._network(x, "experts")['feature']  # 不懂为什么这么做，可能和分布式有关  Soap
            fn = lambda x: self._parallel_network(x, "experts")['feature']
        
        try:
            idx = np.load(Path(self.cfg.pretrain_model_dir) / f"mem/step{self.cfg.idx_task}.npy")
            self._inc_dataset.data_memory = self._inc_dataset.data_inc[idx]
            self._inc_dataset.targets_memory = self._inc_dataset.targets_inc[idx]
            print("use mem idx cache")
        except Exception as e:
            idx = build_exemplar_broadcast(self.cfg,
                                           fn,
                                           self._inc_dataset,
                                           self._memory_size.mem_per_cls)
            pretrain_has_mem = False
        if self.cfg.save_model:
        # if utils.is_main_process() and self.cfg.save_model:  # 不进行分布式训练  Soap
            np.save(self.cfg.mem_folder / f"step{self.cfg.idx_task}.npy", idx)
            mindspore.save_checkpoint(self._parallel_network.get_dict(self.cfg.idx_task), 
                                      self.cfg.ckpt_folder / f"step{self.cfg.idx_task}.ckpt")

    def eval_task(self, data_loader, cfg):
        self._parallel_network.set_train(False)
        r, t = collate_result(lambda x: self._parallel_network(x, "eval"), data_loader, cfg)
        return r['logit'], t


global_step = 0
global_kd_step = 0
def forward_experts(cfg: IncrementalConfig, model, old_model, train_loader, optimizer, scaler): 
    meter = MultiAverageMeter()  
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean').to_float(mindspore.float32)
    def forward_fn(data, label):
        output = model(data, "experts")
        loss = loss_fn(output['logit'], label)
        loss_aux = ops.zeros([1])
        if output['aux_logit'] is not None and cfg.idx_task > 0:
            # 还未进行编写，先略过，怀疑是下一个阶段的事情  Soap
            loss_aux = aux_loss(cfg.aux_cls_type,
                                cfg.aux_cls_num,
                                cfg.nb_seen_classes,
                                output['aux_logit'],
                                label,
                                nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                                 reduction='mean').to_float(mindspore.float32))
        loss_bwd = loss + loss_aux
        loss_bwd = scaler.scale(loss_bwd)
        return loss_bwd, loss, loss_aux

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    def train_step(inputs, targets):
        (loss_bwd, loss, loss_aux), grads = grad_fn(inputs, targets)
        optimizer(grads)
        return loss_bwd, loss, loss_aux

    for batch in train_loader.create_dict_iterator():
        loss_bwd, loss, loss_aux = train_step(batch['data'], batch['label'])
        loss_bwd = float(loss_bwd.asnumpy())
        loss = float(loss.asnumpy())
        loss_aux = float(loss_aux.asnumpy())
        meter.update("clf_loss", loss)
        meter.update("aux_loss", loss_aux)

        if cfg.debug:
            break

    # if utils.is_main_process():   # 分布式，先放弃 Soap
    global global_step 
    # log_dict(utils.get_tensorboard(), meter.avg_per, global_step)  # log与tb有关，先放弃
    global_step += 1
    return meter.avg_all


def finetune_experts(cfg, model, train_loader, optimizer, scaler): 
    _loss = AverageMeter()  

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean').to_float(mindspore.float32)
    def forward_fn(data, label, temperature):
        output = model(data, "experts")
        loss = loss_fn(output['logit'] / temperature, label)
        loss = scaler.scale(loss)
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    def train_step(inputs, targets, temperature):
        loss, grads = grad_fn(inputs, targets, temperature)
        optimizer(grads)
        return loss

    for inputs, targets in train_loader:
        loss = train_step(inputs, targets, cfg.ft.temperature)
        _loss.update(loss)
        if cfg.debug:
            break

    return _loss.avg


experts_trainer_cfg = {
    "baseline": [forward_experts, finetune_experts],
}