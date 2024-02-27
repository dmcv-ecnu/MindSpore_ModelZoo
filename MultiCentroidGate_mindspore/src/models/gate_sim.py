import copy
import itertools
 
from args import IncrementalConfig

import factory
import yaml
import utils
import mindspore.nn as nn
from utils import TaskInfoMixin 
from .ensemble import Model as EnModel
import numpy as np
import mindspore

class Model(nn.Cell):
    def __init__(self, cfg: IncrementalConfig):
        super(Model, self).__init__()
        self.cfg: IncrementalConfig = cfg

        self.der = EnModel(cfg) 
        self.cache_repeat = None

    def construct(self, x, mode='train'):
        return self.der(x)

    def param_groups(self):
        return { 
            "classifier": itertools.chain(self.der.param_groups()['classifier']),
            "experts": list(self.der.trainable_params()),
            "experts_classifier": list(self.der.classifier.trainable_params()),
        }

    def set_model_train(self, mode):
        if mode == "experts":
            self.der.set_model_train(mode)
        else:
            raise ValueError() 

    def freeze(self, mode="old"):
        utils.switch_grad(self.get_parameters(), True)
        if mode == "old":
            self.der.freeze(mode)
        elif mode == "backbone":
            self.der.freeze("backbone")
        elif mode == "experts":
            utils.switch_grad(self.der.get_parameters(), False)
        else:
            raise ValueError() 

    def reset(self, mode):
        if mode == "classifier":
            self.der.reset_classifier()
        elif mode == "experts_classifier":
            self.der.reset("classifier")
        else:
            raise ValueError()

    def add_classes(self, n_classes): 
        self.der.add_classes(n_classes)

    def get_dict(self, i):
        return {
            "der": self.der.get_dict(i),
            "gate": utils.remove_component_from_state_dict(self.parameters_dict(), ["der"], False)
        }
    
    def set_dict(self, dict, i, load_gate=True):
        self.der.set_dict(dict["der"], i)
        if load_gate:
            mindspore.load_param_into_net(self, dict["gate"])