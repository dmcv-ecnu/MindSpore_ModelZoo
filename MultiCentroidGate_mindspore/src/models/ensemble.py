import copy
import itertools
from args import IncrementalConfig
import math
import factory
import yaml
import utils
import mindspore
import mindspore.nn as nn
from utils import TaskInfoMixin 
from mindspore.common.initializer import initializer, HeNormal, HeUniform
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P


class Model(nn.Cell):
    def __init__(self, cfg: IncrementalConfig):
        super(Model, self).__init__()
        self.cfg = cfg 
        self.convnets = nn.CellList()
        self.classifier = nn.CellList()
        self.remove_last_relu = False
        self.use_bias = False 
        self.aux_classifier = None
        self.out_dim = None
        self.init = "kaiming"

    def construct(self, x, method=''):
        if method == "last":
            lf = self.convnets[-1](x)
            return {
                'logit': self.classifier[-1](lf),
                'aux_logit': self.aux_classifier(lf) if self.aux_classifier is not None else None 
            }
        fl = [conv(x) for conv in self.convnets]
        l = [head(fl[i]) for i, head in enumerate(self.classifier)]
        l = mindspore.ops.cat(l, 1)

        return {
            "logit": l,
            "feature": mindspore.ops.cat(fl, 1),
            "aux_logit": self.aux_classifier(fl[-1]) if self.aux_classifier is not None else None
        } 
    
    def set_model_train(self, mode):
        self.convnets[:-1].set_train(False)
        self.convnets[-1].set_train() 
    
    def param_groups(self):
        return { 
            "classifier": list(self.classifier.trainable_params()),
        }

    def freeze(self, mode="old"):
        utils.switch_grad(self.get_parameters(), True)
        if mode == "old":
            utils.switch_grad(self.convnets[:-1].get_parameters(), False)  # 把除了老模型的backbone都冻结
        elif mode == "backbone":
            utils.switch_grad(self.convnets.get_parameters(), False)
        else:
            raise ValueError()
        
    def reset(self, mode):
        assert mode in ["classifier"]
        if mode == "classifier":
            for cell in self.classifier:
                cell.weight.set_data(initializer(HeUniform(negative_slope=math.sqrt(5)), 
                                                 cell.weight.shape, cell.weight.dtype))

    def reset_classifier(self):
        self.reset("classifier")

    def add_classes(self, n_classes): 
        self._add_classes_multi_fc(n_classes)

    def _add_classes_multi_fc(self, n_classes):
        new_clf, out_dim = factory.create_convnet(self.cfg)
        self.out_dim = out_dim

        if self.cfg.idx_task > 0:
            mindspore.load_param_into_net(new_clf, self.convnets[-1].parameters_dict())
            if self.cfg.aux_cls_type == "1-n":
                self.aux_classifier = self._gen_classifier(self.out_dim,
                                             self.cfg.nb_task_classes + self.cfg.aux_cls_num) 
            elif self.cfg.aux_cls_type == "n-n":
                self.aux_classifier = self._gen_classifier(self.out_dim, self.cfg.nb_seen_classes)

        self.convnets.append(new_clf)
        self.classifier.append(self._gen_classifier(self.out_dim, self.cfg.nb_task_classes))

    def _gen_classifier(self, in_features, n_classes):
        # 分类头采用FP32  Soap
        classifier = nn.Dense(in_features, n_classes, has_bias=self.use_bias)
        if self.init == "kaiming": 
            pass
        if self.use_bias:
            classifier.bias.set_data(initializer("zeros", classifier.bias.shape, classifier.bias.dtype))

        return classifier
    
    def get_dict(self, i):
        return {
            "fe": self.convnets[i].parameters_dict(),
            "fc": self.classifier.parameters_dict()
        }
 
    def set_dict(self, state_dict, i):
        mindspore.load_param_into_net(self.convnets[i], state_dict["fe"])
        mindspore.load_param_into_net(self.classifier, state_dict["fc"])