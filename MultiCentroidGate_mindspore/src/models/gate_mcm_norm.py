import copy
import itertools
import math 
from turtle import forward
 
from args import IncrementalConfig

import factory
import torch
import torch.nn.functional as F
import yaml
from torch import native_norm
import utils
from utils import TaskInfoMixin 
from .ensemble import Model as EnModel
import numpy as np
from .layers.mcm import DifferMCM
import torch.nn as nn

class Model(nn.Module,):
    def __init__(self, cfg: IncrementalConfig):
        super(Model, self).__init__()
        self.cfg: IncrementalConfig = cfg

        self.der = EnModel(cfg)
        self.gate, self.out_dim = factory.create_convnet(cfg) 
        self.embedding = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False)
        )
        self.logit_map = nn.Linear(0, 0, False) # from mcm to task ? (cls_num, self.k * n)
        # self.mcm = None
        self.mcm = DifferMCM(in_channel=self.out_dim, expansion_factor=self.cfg.md_k, reuse=self.cfg.md_reuse)
        self.cls_head = nn.Linear(self.out_dim, 0, False)
        # self.tt = nn.Identity()
        # self.tt_type = -1
        self.use_norm = False
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.mcm_temperature = nn.Parameter(torch.tensor(self.cfg.task_temperature), True)
        self.gcl_temperature = nn.Parameter(torch.tensor(self.cfg.task_temperature), True)

        self.cache_repeat = None
    
    def sim_fn(self, x, y, norm=True):
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
    
    def forward(self, x, mode="train", *args): 
        if mode == "experts last":
            return self.der(x, "last")  
        do = self.der(x)
        if mode == "experts": return do   

        gof = self.gate(x)
        # norm = utils.per_task_norm(do['feature'], self.cfg.nb_tasks)

        # gf = gof.detach()
        # print("Warning: detach gf gradient.")
        gf = gof
        # gf = self.embedding(gf) 
  
        # gate_logit = self.sim_fn(gf, self.mcm.view(n * k, d), self.use_norm)
        gate_logit, gol = self.mcm(gf, lambda x, y: self.sim_fn(x, y, self.use_norm))

        gate_cls_logit = self.cls_head(gof)
        logit_map = self.logit_map(gate_logit) # softmax?
 
        if mode == "gate":
            return {
                "gate_feature": gf, 
                "gate_logit": gate_logit,
                "gate_cls_logit": gate_cls_logit,
                "logit_map": logit_map,
                'gate_task_logit': gol, # not gol.  
            } 
        elif "experts gate" in mode:
            return {
                **do,
                "gate_feature": gf, 
                "gate_logit": gate_logit,
                "gate_cls_logit": gate_cls_logit,
                "logit_map": logit_map,
                'gate_task_logit': gol, # not gol.  
            } 
        b = gf.shape[0]  
        # gol = gate_logit.reshape(b, n, k).amax(2)
        gocls = gate_cls_logit
        gocl = utils.opt_by_task(gocls, self.cfg.unique_increments, "amax")

        logit_mcm = self.masked(gol, do, self.cfg.task_temperature)
        logit_gcl = self.masked(gocl, do, self.cfg.task_temperature)

        lamb = self.alpha
        logit_mix = lamb * logit_mcm + (1 - lamb) * logit_gcl
        return {
            'logit': do['logit'],
            'feature': do['feature'],
            'gate_feature': gf,
            "gate_logit": gate_logit,
            'gate_task_logit': gol, # not gol. 
            'final_logit': logit_mcm,
            'final_logit_gcl': logit_gcl,
            'final_logit_mix': logit_mix,
            "gate_cls_logit": gocls,
            "logit_map": logit_map
        }

    def masked(self, gol_use, do, temperature):
        mask_ = gol_use.div(temperature).softmax(1).repeat_interleave(self.cache_repeat, dim=1) 
        if self.cfg.topk != -1:
            _, tk = gol_use.topk(min(gol_use.shape[1], self.cfg.topk), 1) 
            mask = torch.zeros([gol_use.shape[0], self.cfg.nb_tasks], dtype=torch.long).cuda()  
            mask.scatter_(dim=1, index=tk, value=1)
            mask = torch.repeat_interleave(mask, self.cache_repeat, dim=1)
            mask = mask * mask_
        else :
            mask = mask_
        logit = do['logit'] * mask
        return logit
    
    def param_groups(self):
        return { 
            "experts": self.der.parameters(),
            "experts_classifier": self.der.classifier.parameters(),
            "gate": itertools.chain(self.gate.parameters(),
                                    self.mcm.parameters(),
                                    self.embedding.parameters(),
                                    self.cls_head.parameters(),
                                    [self.mcm_temperature, self.gcl_temperature],
                                    self.logit_map.parameters()
                                    ),
            "mcm": self.mcm.parameters(),
            "gate_classifier": self.cls_head.parameters(),
            "gate_classifier_mcm": itertools.chain(self.cls_head.parameters(), self.mcm.parameters()),
            "gate_temp": itertools.chain([self.mcm_temperature, self.gcl_temperature], 
                                    self.cls_head.parameters()),
            "alpha": [self.alpha]
        }
    
    def set_train(self, mode):
        if mode == "experts":
            self.der.set_train(mode)
            self.gate.eval()
        elif mode == "gate":
            self.der.eval()
            self.gate.train()
        else:
            raise ValueError() 

    def freeze(self, mode="old"):
        utils.switch_grad(self.parameters(), True)
        if mode == "old":
            self.der.freeze(mode)
        elif mode == "backbone":
            self.der.freeze("backbone")
        elif mode == "experts":
            utils.switch_grad(self.der.parameters(), False)
        elif mode == "except t2t":
            utils.switch_grad(self.der.parameters(), False)
            utils.switch_grad(self.gate.parameters(), False)
        else:
            raise ValueError() 
    
    def reset(self, mode): 
        if mode == "classifier":
            self.der.reset_classifier() 
        elif mode == "experts_classifier":
            self.der.reset("classifier")
        elif mode == "gate_classifier":
            # self.cls_head.reset_parameters()
            nn.init.kaiming_normal_(self.cls_head.weight, nonlinearity="linear")
        elif mode == "mcm":
            # no...
            assert 1 == 0
            nn.init.kaiming_normal_(self.mcm, nonlinearity="linear")
            nn.init.kaiming_uniform_(self.mcm, a=math.sqrt(5))
        elif mode == "alpha":
            self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        elif mode == "temperature":
            self.mcm_temperature = nn.Parameter(torch.tensor(self.cfg.task_temperature), True)
            self.gcl_temperature = nn.Parameter(torch.tensor(self.cfg.task_temperature), True)
        else:
            raise ValueError() 

    def add_classes(self, n_classes): 
        self.der.add_classes(n_classes)  
        self.mcm.add_classes(n_classes)

        copy_map = torch.empty([0, self.out_dim]) if self.logit_map is None else self.logit_map.weight.data.clone()
        self.logit_map = nn.Linear(self.mcm.out_dim, self.cfg.nb_seen_unique_classes)
        # print(self.logit_map.weight.shape) # bug: shape[1] = 0 可能导致broadcast错误
        self.logit_map.weight.data[:copy_map.shape[0], :copy_map.shape[1]] = copy_map 

        copy_cls_head = torch.empty([0, self.out_dim]) if self.cls_head is None else self.cls_head.weight.data.clone()
        self.cls_head = nn.Linear(self.out_dim, self.cfg.nb_seen_unique_classes, False)
        self.cls_head.weight.data[:copy_cls_head.shape[0]] = copy_cls_head # small acc down.

        self.cache_repeat = torch.tensor(self.cfg.increments).cuda()
  
      
    
    def get_dict(self, i):
        return {
            "der": self.der.get_dict(i),
            "gate": utils.remove_component_from_state_dict(
                        self.state_dict(), ["der"], False)
        }
    
    def set_dict(self, dict, i, load_gate=True):
        self.der.set_dict(dict["der"], i)
        if load_gate:
            self.load_state_dict(dict["gate"], strict=False)