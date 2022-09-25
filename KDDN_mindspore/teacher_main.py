from mindspore.dataset import GeneratorDataset
from data import dehazeDataloader
import mindspore.nn as nn
from Teacher import endeFUINT2_1
import mindspore
from PIL import Image
from mindspore.ops import stop_gradient
from option import args
from mindspore import ops
import mindspore.dataset.transforms.c_transforms as C
from mindspore import ParameterTuple
from mindspore import Tensor, Model
from mindspore import dtype as mstype
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset as ds
import os
import moxing as mox

mindspore.context.set_context(device_target = "CPU")
dataset = dehazeDataloader(args)
dataloader = GeneratorDataset(source=dataset, column_names=["image", "label"])
dataloader = dataloader.batch(20)
#(10,3,256,256)

#定义损失网络
class MyWithLossCell(nn.Cell):
    """定义损失网络"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out, _ = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        return self.backbone


#定义损失函数
criterion = nn.L1Loss()
modelG = endeFUINT2_1()
#定义优化器
optimizer_G = mindspore.nn.Adam(filter(lambda p: p.requires_grad, modelG.get_parameters()), learning_rate=1e-4,
                                            beta1=0.9, beta2 = 0.999)
net_with_criterion = MyWithLossCell(modelG, loss_fn=criterion)

#尝试一下不使用单步训练
# type_cast_op_image = C.TypeCast(mstype.float32)
# type_cast_op_label = C.TypeCast(mstype.float32)
# HWC2CHW = CV.HWC2CHW()
# dataloader = dataloader.map(operations=[type_cast_op_image, HWC2CHW], input_columns="image")
# dataloader = dataloader.map(operations=type_cast_op_label, input_columns="label")
# model = Model(modelG, loss_fn=criterion, optimizer=optimizer_G)
# model.train(epoch=100, train_dataset=dataloader)
#尝试结束


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        """参数初始化"""
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        # 使用tuple包装weight
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        # 定义梯度函数
        self.grad = mindspore.ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        # 为反向传播设定系数
        sens = mindspore.ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        return loss, self.optimizer(grads)


#train_net = MyTrainStep(net_with_criterion,optimizer_G)
train_net = TrainOneStepCell(net_with_criterion,optimizer_G)


epoch = 30
for epoch in range(epoch):
    iteraretor = 1
    for item in dataloader.create_dict_iterator():
        Ix,Jx = item['label'],item['label']#教师网络为自编码器，所以输入输出相同
        #print("这只是一个标记！！！！！！！！！！！！！！！！！！！！！")
        #print(Ix.shape,Jx.shape)
        #print(type(Ix))
        train_net(Ix, Jx)
        loss_val = net_with_criterion(Ix, Jx)#源码里jx有detach
        print(loss_val)
        iteraretor = iteraretor + 1
        if iteraretor%10 == 0:
            #保存模型
            print("这里保存了！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
            premodel = "/cache/premodel/"
            os.makedirs(premodel, exist_ok=True)
            mindspore.save_checkpoint(modelG, premodel + "teacherNet.ckpt")
            mox.file.copy(premodel + "teacherNet.ckpt", args.train_url + "TeacherResult.ckpt")

# for data in dataloader.create_dict_iterator():
#     print(data["image"].shape, data["label"].shape)

