import math
from collections import OrderedDict

import mindspore as ms

from mindspore import nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import load_checkpoint, load_param_into_net
from option import Args
import utility
from model.label_generator import labelGenerator
from model import edn

def cal_consitency_weight(epoch, init_ep=0, end_ep=1000, init_w=0.0, end_w=1.0):
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        # weight_mse = T * (end_w - init_w) + init_w # linear
        weight_cl = (math.exp(-0.5*(1.0 - T)*(1.0 - T))) * (end_w - init_w) + init_w
    return weight_cl

class Trainer():
    def __init__(self, args:Args, dataset, my_model, my_loss):
        self.args = args
        self.scale = args.scale
        
        self.model1:nn.Cell = my_model[0] # 教师网络
        self.model2:nn.Cell = my_model[1] # 学生网络
        
        self.ds_train:ms.dataset.GeneratorDataset = dataset
        """
        self.ds_train:ms.dataset.GeneratorDataset = dataset.ds_train
        self.ds_meta:ms.dataset.GeneratorDataset = dataset.ds_meta # 元学习
        self.ds_test:ms.dataset.GeneratorDataset = dataset.ds_test # 测试集
        """

        self.labelGenerator = labelGenerator(args)

        self.loss = my_loss
        self.criterion = nn.L1Loss()
        
        lr = nn.CosineDecayLR(0.0000125, 0.0002, args.epochs * 800 // args.batch_size)
        self.optimizer = [
            nn.Adam(self.model1.trainable_params(), learning_rate=lr),
            nn.Adam(self.model2.trainable_params(), learning_rate=lr)
        ]

        self.meta_optimizer = nn.Adam(self.labelGenerator.trainable_params(), learning_rate=lr)

        
        self.params1 = self.optimizer[0].parameters # 教师网络的参数
        self.params2 = self.optimizer[1].parameters # 学生网络的参数
        self.params_mata = self.meta_optimizer.parameters

        self.grad_op = ops.GradOperation(get_by_list=True) # 梯度算子
        self.grad_reducer = ms.ops.functional.identity 
        

        self.lambda_KD = 0.5
        self.error_last = 1e8
        self.T_net = edn.make_model(args)
        param_dict = load_checkpoint('/home/hyacinthe/graduation-dissertation/mycode/src/EDN-new.ckpt')
        load_param_into_net(self.T_net, param_dict)


    def validate(self):
        eval_acc1 = 0
        eval_acc2 = 0
        # 这个要换掉
        iterator_ds = self.ds_train.create_dict_iterator()
        for column in iterator_ds:
            lr = column['lr']
            hr = column['hr']
            slr = self.T_net(hr)[0]
            slr = utility.quantize(slr, self.args.rgb_range)

            sr0 = self.model1(slr)[0][-1]
            sr = self.model2(lr)[0][-1]

            sr0 = utility.quantize(sr0, self.args.rgb_range)
            sr = utility.quantize(sr, self.args.rgb_range)

            eval_acc1 += utility.calc_psnr( # 学生网络
                sr0, hr, self.scale, self.args.rgb_range
            )

            eval_acc2 += utility.calc_psnr(
                sr, hr, self.scale, self.args.rgb_range
            )

        psnr_sr1 = eval_acc1 / self.args.n_train # 解释网络的
        psnr_sr2 = eval_acc2 / self.args.n_train #学生网络的
        delta = psnr_sr1 - psnr_sr2 # 教师减去学生的

        return delta