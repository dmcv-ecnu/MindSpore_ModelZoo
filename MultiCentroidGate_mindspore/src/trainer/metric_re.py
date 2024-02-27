import importlib
import itertools
import logging
from copy import deepcopy
import math
from multiprocessing import reduction
from args import IncrementalConfig 

import factory
import numpy as np
import timm.utils.cuda
import torch
import torch.cuda.amp
import torch.distributed as dist  
from ds.incremental import IncrementalDataset
# from models import der, moenet, network, resnetatt, share
from models.loss import AutoKD, BCEWithLogitTarget
from models.loss.imbalanced import compute_adjustment
from models.loss.t import DivLoss
from models.loss.distillation import SSIL, MetricKD 
from rehearsal.memory_size import MemorySize
from scipy.spatial.distance import cdist
from timm.optim import create_optimizer, create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from torch import div, nn
from torch.nn import DataParallel  # , DistributedDataParallel
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timm.data import Mixup
from torchvision import transforms as T
from torch.utils.data import TensorDataset

# from tools import factory, utils
import utils
from utils.distributed import is_main_process
from utils.metrics import accuracy
from utils.model_utils import extract_features

from .utils import build_exemplar, build_exemplar_broadcast, collate_result
from utils import ClassErrorMeter, AverageMeter, MultiAverageMeter, log_dict
from models import *
from collections import defaultdict
from ds.incremental import WeakStrongDataset, DummyDataset, DummyDataset3
import os.path
from pathlib import Path

from torchvision import transforms as T
from models.loss import aux_loss
from scipy.optimize import linear_sum_assignment
from easydict import EasyDict

from models.layers.mcm import DifferMCM
from typing import Tuple, Union
        
class IncModel():
    def __init__(self, cfg: IncrementalConfig, inc_dataset):
        super(IncModel, self).__init__()
        self.cfg = cfg
        # Data
        self._inc_dataset: IncrementalDataset = inc_dataset

        # Optimizer paras
        self._n_epochs = None

        # memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], cfg["memory_size"], cfg["fixed_memory_per_cls"])
        self._coreset_strategy = cfg["coreset_strategy"]

        # Model
        self._network = factory.create_network(cfg)
        self._parallel_network = None
        self._old_model = None
  
        self.old_mcm = None

        self.mlp = nn.Linear(512 * 0, 512, False).cuda() 

        self.old_mcm_idx = np.empty([0], dtype=np.int64)

        # training routing
        try:
            global gate_trainer_cfg 
            self.gt, self.gf = gate_trainer_cfg[cfg.subtrainer]
        except Exception as e:
            print(str(e))
 
    def before_task(self):
        dist.barrier()
        self._memory_size.update_nb_classes(self.cfg.nb_seen_classes)
        self._network.add_classes(self.cfg.nb_task_classes)
        if self.cfg.pretrain_model_dir is not None:
            self.load()
        if self.cfg["syncbn"]:
            self._network = nn.SyncBatchNorm.convert_sync_batchnorm(self._network)
        self._network = self._network.cuda()
        del self._parallel_network
        self._parallel_network = DDP(self._network,
                                    device_ids=[self.cfg.gpu],
                                    output_device=self.cfg.gpu,
                                    find_unused_parameters=True,
                                    broadcast_buffers=True # if set, call model twice will trigger inplace opt.
                                    ) 
        self._parallel_network = self._parallel_network.cuda()
        dist.barrier() # required.

    def save(self):
        if utils.is_main_process() and self.cfg.save_model:
            torch.save(self._network.get_dict(self.cfg.idx_task), self.cfg.ckpt_folder / f"step{self.cfg.idx_task}.ckpt")
    
    def load(self):
        assert self.cfg.part_enable[0] == 0
        model_file = (Path(self.cfg.pretrain_model_dir) / Path(f"ckpt/step{self.cfg.idx_task}.ckpt")).expanduser() 
        state_dict = torch.load(model_file, map_location=torch.device("cpu"))
        self._network.set_dict(state_dict, self.cfg.idx_task, False)  
 
    def train_task(self, train_loader, val_loader):
        self.metric_learning(train_loader)
        if self.cfg.md_ft_bn: self.get_old_mcm_idx(None)
    
    def get_old_mcm_idx(self, train_loader):
        assert "ImageNet" not in self.cfg.dataset
        dataset = self._inc_dataset.get_custom_dataset("train_cur", "test") # don't change me...
        train_loader = factory.create_dataloader(self.cfg, dataset, False, False)
        # self._parallel_network.eval()
        r, t = collate_result(lambda x: self._parallel_network(x, "gate"), train_loader) 
        self.old_mcm_idx = np.concatenate([self.old_mcm_idx, r['gate_logit'].argmax(1).numpy()], 0)
  
    def compute_la(self, y, copy_y=None):
        # y remapped. copy_y not remapped.
        # MCM_OVLP: #2
        # t = utils.target_to_task(y, self.cfg.increments)
        t = utils.target_to_task(copy_y, self.cfg.unique_increments)
        ut, uc = np.unique(t, return_counts=True)
        uc = uc / np.array(self._network.mcm.centroids_task)
        label_freq_array = uc  
        label_freq_array = label_freq_array / label_freq_array.sum()
        log_label_freq_array = np.log(label_freq_array ** 1.0 + 1e-5) 
        adjustments = torch.from_numpy(log_label_freq_array)\
                           .repeat_interleave(torch.LongTensor(self._network.mcm.centroids_task), 0).cuda() 
        # adjustments_for_ce = torch.from_numpy(log_label_freq_array)\
        #                           .repeat_interleave(torch.LongTensor(self.cfg.increments), 0).cuda()
        # ... TODO: seems impl but top used....
        # if self.cfg.idx_task > 0:
        #     import pdb; pdb.set_trace()
        _y = copy_y if copy_y is not None else y
        _ut, _uc = np.unique(_y, return_counts=True)
        _label_freq_array = _uc  
        _label_freq_array = _label_freq_array / _label_freq_array.sum()
        _log_label_freq_array = np.log(_label_freq_array ** 1.0 + 1e-5) 
        adjustments_for_ce = torch.from_numpy(_log_label_freq_array).cuda()
        return adjustments, adjustments_for_ce

    def metric_learning(self, train_loader):  
        # not useful... lower than random batch sampler.
        if self.cfg.trainset_sampling == "maxk":
            assert self.cfg.increment == self.cfg.base_classes 
            train_loader = factory.create_dataloader(self.cfg, train_loader.dataset, self.cfg.distributed, True, True, "maxk")

        if self.cfg.overlap_dataset: # decoding process.
            copy_y = utils.decode_targets(train_loader.dataset.y,
                                            self.cfg.increments,
                                            self.cfg.overlap_class)
        else:
            copy_y = train_loader.dataset.y

        train_loader = factory.replace_loader_transforms(self.cfg.new_transform, self.cfg, train_loader)
        if self.cfg.md_pos_old: # don't use. really low acc.
            nb_new_cls_sample = len(self._inc_dataset.data_cur)
            # concat sequence matters.
            pos_target = np.concatenate([np.full([nb_new_cls_sample], -1), self.old_mcm_idx], 0) 
            ds = train_loader.dataset
            new_trainset = DummyDataset3(ds.x, ds.y, pos_target, ds.transform) 
            train_loader = factory.create_dataloader(self.cfg, new_trainset, self.cfg.distributed, True)
        
        if self.cfg.md_use_la:
            adjustments_tuple = self.compute_la(train_loader.dataset.y, copy_y)
        else:
            adjustments_tuple = (0, 0)

        scaler = utils.Scaler(self.cfg.amp)  
        self._parallel_network.train()
        self._parallel_network.module.set_train("gate")
        self._parallel_network.module.freeze("experts")
        dist.barrier()

        parameters = self._parallel_network.module.param_groups()["gate"]
        parameters = filter(lambda x: x.requires_grad, parameters) 
        optimizer = create_optimizer(self.cfg, parameters) 
        scheduler, n_epochs = create_scheduler(self.cfg, optimizer)

        with tqdm(range(n_epochs), disable=not utils.is_main_process()) as pbar:
            for e in pbar: 
                train_loader.sampler.set_epoch(e)   
                with torch.autograd.set_detect_anomaly(False):
                    loss = self.gt(cfg=self.cfg,
                                        model=self._parallel_network,
                                        old_model=self._old_model,
                                        train_loader=train_loader, 
                                        mlp=self.mlp,
                                        sim_fn=lambda x, y: sim_fn(x, y, norm=self.cfg.md_cos_dis),
                                        optimizer=optimizer, 
                                        adjustments=adjustments_tuple, 
                                        scaler=scaler) 
                scheduler.step(e)
                if utils.is_main_process():
                    pbar.set_description(f"E {e} metric learning loss: {loss:.3f}") 

    def after_task(self): 
        self._parallel_network.eval()
        if self.cfg.coreset_feature == "last":
            fn = lambda x: self._network(x)['feature'][:, -512:]
        elif self.cfg.coreset_feature == "all":
            fn = lambda x: self._network(x, "experts")['feature']
            
        pretrain_has_mem = True
        try:
            # raise Exception()
            idx = np.load(Path(self.cfg.pretrain_model_dir) / f"mem/step{self.cfg.idx_task}.npy")
            print("use mem idx cache")
        except Exception as e:
            idx = build_exemplar_broadcast(self.cfg,
                            fn,
                            self._inc_dataset,
                            self._memory_size.mem_per_cls)
            pretrain_has_mem = False

        self._inc_dataset.data_memory = self._inc_dataset.data_inc[idx]
        self._inc_dataset.targets_memory = self._inc_dataset.targets_inc[idx]

        if self.cfg.md_pos_old:
            self.old_mcm_idx = self.old_mcm_idx[idx]
        # self._old_model = deepcopy(self._parallel_network)
        if self.cfg.save_model:
            self.save()
            if utils.is_main_process() and not pretrain_has_mem:
                np.save(self.cfg.mem_folder / f"step{self.cfg.idx_task}.npy", idx) 
        # utils.switch_grad(self._old_model.parameters(), False)
        
    def eval_task(self, data_loader): 
        self._parallel_network.eval()  
        r, t = collate_result(lambda x: self._network(x, "eval"), data_loader) 
        # return r['logit'], r['gate_task_logit'], r['final_logit'], r['final_logit_gcl'], t
        return r, t

global_step = 0
global_kd_step = 0

def find_pos_2(cfg: IncrementalConfig, feature, targets, centroid: Union[torch.Tensor, DifferMCM], sim_fn):
    idx_centroid = torch.arange(centroid.out_dim, dtype=torch.long).cuda()\
                        .split(centroid.centroids_task) 
    centroid = centroid.mcm 
 
    mean_feature, mean_targets = utils.per_target_mean(feature, targets)
    mean_tasks = utils.target_to_task(mean_targets, cfg.unique_increments)  
    
    # task centroid list
    a = [centroid[t] for t in mean_tasks] 
    ad = [idx_centroid[t] for t in mean_tasks]
    
    # select in each task the best centroid using mean class prototype
    bb = per_task_map(mean_feature, mean_tasks, centroid, sim_fn) ###########
    c = [a[i][b] for i, b in enumerate(bb)] #a[torch.arange(mb), bb] 
    cd = [ad[i][b] for i, b in enumerate(bb)] #ad[torch.arange(mb), bb] 

    # assign back to original batch
    indices = ((mean_targets - targets.unsqueeze(0).T) == 0).nonzero()[:, 1]
    c = torch.stack([c[i] for i in indices])
    cd = torch.stack([cd[i] for i in indices])
    return c, cd 

def sim_fn(x, y, norm=True):
    if len(x.shape) == 2:
        x = x[None, ...]
        y = y[None, ...]
        if norm:
            x = F.normalize(x, 2, 2)
            y = F.normalize(y, 2, 2)
        return torch.bmm(x, y.transpose(1, 2))[0]
    if norm:
        x = F.normalize(x, 2, 2)
        y = F.normalize(y, 2, 2)
    return torch.bmm(x, y.transpose(1, 2))

def forward_gate_logit_map(cfg: IncrementalConfig, model, old_model, train_loader, sim_fn,
 optimizer, scaler, adjustments: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs):  
    adjustments, adjustments_for_ce = adjustments
    meter = MultiAverageMeter() 
    acc = AverageMeter()
    centroid = model.module.mcm.clone()  
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(cfg.gpu, non_blocking=True), targets.to(cfg.gpu, non_blocking=True) 
        tasks = utils.target_to_task(targets, cfg.unique_increments)   
        # print(inputs.shape)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            output = model(inputs, "experts gate")
            feature = output['gate_feature']
            b = inputs.shape[0] 
            c, cd = find_pos_2(cfg, feature, targets, centroid, sim_fn)  
            loss = F.cross_entropy((output['gate_logit'] + adjustments) , cd)
            lbd = 1
            if cfg.distillation == "experts":
                use_logit = output['logit_map']
                if cfg.md_use_la and cfg.md_use_la_on_cls:
                    loss_kd_t, lbd = AutoKD(cfg.tau)(use_logit + adjustments_for_ce, output['logit'])
                    # comment this improve acc to 67.77(+0.4)
                    # loss_kd_t += F.cross_entropy(use_logit + adjustments_for_ce, targets)
            
            if cfg.md_div_k:
                loss_bwd = loss / cfg.md_k + loss_kd_t  
            else:
                loss_bwd = loss + loss_kd_t  
            with torch.no_grad(): 
                acc.update(accuracy(output['gate_task_logit'], tasks)[0], b) 

        optimizer.zero_grad()
        scaler(loss_bwd, optimizer) 
        # assert model.module.mcm.requires_grad == True
        meter.update("clf_loss", loss.item())  
        meter.update("kd_t_loss", loss_kd_t.item())

    if utils.is_main_process(): 
        global global_step 
        log_dict(utils.get_tensorboard(), meter.avg_per, global_step)
        global_step += 1
    return acc.avg


def per_task_map(features, tasks, centroid, sim_fn):
    """
    mean feature(per task), task(class->task)
    """
    with torch.no_grad():
        ut = torch.unique(tasks)
        bb = torch.zeros_like(tasks)
        for t in ut:
            st = tasks[tasks == t]
            sf = features[tasks == t] # ?b d 
            sidx = torch.where(tasks == t)[0]
            assert st.shape[0] <= centroid[t].shape[0]
            tmp = centroid[t] # k d
            try:
                row_ind, col_ind = linear_sum_assignment(sim_fn(sf, tmp).cpu().numpy(), maximize=True)
            except:
                import pdb; pdb.set_trace()
            bb[sidx[row_ind]] = torch.from_numpy(col_ind).cuda()
            # sbb[row_inx] = torch.from_numpy(col_ind).cuda() # wrong implements.... always bb == 0. but good result 76.
        return bb

global_step = 0
global_kd_step = 0

gate_trainer_cfg = {
    # "baseline": (forward_gate, ft),
    # "ft_all": (forward_gate, ft2),
    # "mixup": (forward_gate_mixup, ft),
    "logit_map": (forward_gate_logit_map, None),
    # "logit_map_rev": (forward_gate_logit_map_rev, ft514),
    # "logit_map_contrastive": (forward_gate_logit_map_contrastive, ft514),
    # "old_mcm": (forward_gate_logit_map_old, ft514),
    # "unicentroid": (forward_gate_unicentroid, None)
}