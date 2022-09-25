from mindspore.dataset import GeneratorDataset
import numpy as np
from data import dehazeDataloader
#from loss import *
import os
#from SSIM import SSIM
import imageio
from Student import *
from Teacher import endeFUINT2_1
from mindspore.ops import stop_gradient
from option import args
import moxing as mox
from mindspore.train import Model
#from mindvision.engine.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, SummaryCollector

mindspore.context.set_context(device_target = "CPU")
#数据集
dataset = dehazeDataloader(args)
dataloader = GeneratorDataset(source=dataset, column_names=["image", "label"],num_parallel_workers=48)
dataloader = dataloader.batch(20)

#定义模型
modelG = Student2()
#定义优化器
optimizer_G = mindspore.nn.Adam(filter(lambda p: p.requires_grad, modelG.get_parameters()), learning_rate=1e-4,
                                             beta1=0.9, beta2 = 0.999)
#加载教师网络
teacher = endeFUINT2_1()

premodel = "/cache/premodel/"
os.makedirs(premodel, exist_ok=True)
mox.file.copy(args.train_url + "TeacherResult.ckpt", premodel + "teacherNet.ckpt")
mox.file.copy(args.train_url + "studentNet.ckpt", premodel + "studentNet.ckpt")

param_dict = mindspore.load_checkpoint(premodel + "teacherNet.ckpt")
#stu_parm = mindspore.load_checkpoint(premodel + "studentNet.ckpt")
mindspore.load_param_into_net(teacher, param_dict)
#mindspore.load_param_into_net(modelG,param_dict)


#定义损失
class StudentLoss(nn.LossBase):
    def __init__(self,reduction="mean"):
        super(StudentLoss, self).__init__(reduction)
        self.criterion = nn.L1Loss()
        #self.VGGLoss = VGGLoss()

    def construct(self, data, target):
        #Ix = data
        Jx = target
        PT, feaT = teacher(Jx)
        #Jx = stop_gradient(Jx)#替代detach
        fake, feaS = data
        g_loss_MSE0 = self.criterion(fake, Jx) #+ self.VGGLoss(fake, Jx)
        lossRM = 0
        for i in range(len(feaS)):
            #feaT[i] = stop_gradient(feaT[i])
            lossRM += self.criterion(feaS[i], feaT[i])
        loss = g_loss_MSE0 + lossRM
        return self.get_loss(loss)

#定义带损失的网络
class StuWithLossCell(nn.Cell):
    def __init__(self,backbone,loss_fn):
        super(StuWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def backbone_network(self):
        return self.backbone

    def construct(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)

#开始训练
stuloss = StudentLoss()
net_with_criterion = StuWithLossCell(modelG, loss_fn=stuloss)
# train_net = nn.TrainOneStepCell(net_with_criterion,optimizer_G)
# epoch = 30
# for epoch in range(epoch):
#     iteraretor = 0
#     for item in dataloader.create_dict_iterator():
#         Ix,Jx = item['image'],item['label']#教师网络为自编码器，所以输入输出相同
#         #print(Ix.shape,Jx.shape)
#         #print(type(Ix))
#         train_net(Ix,Jx)
#         loss_val = net_with_criterion(Ix,Jx)
#         print(loss_val)
#         iteraretor = iteraretor + 1
#         if iteraretor%10 == 0:
#             print("这里保存了模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
#             mindspore.save_checkpoint(modelG, premodel + "studentNet.ckpt")
#             mox.file.copy(premodel + "studentNet.ckpt", args.train_url + "studentNet.ckpt")

#这里尝试多核并行
model = Model(network=modelG,loss_fn=stuloss, optimizer=optimizer_G, metrics={'acc'})
epoch = 60

# 定义回调类
step_size = dataloader.get_dataset_size()
time_cb = TimeMonitor(data_size=step_size)
loss_cb = LossMonitor()
cb = [time_cb, loss_cb]
config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=10)
ckpt_cb = ModelCheckpoint(prefix="hhllog", directory=premodel, config=config_ck)
cb += [ckpt_cb]
model.train(epoch, dataloader, callbacks=cb, dataset_sink_mode=False)

mindspore.save_checkpoint(modelG, premodel + "studentNet.ckpt")
mox.file.copy(premodel + "studentNet.ckpt", args.train_url + "studentNet.ckpt")

#start test
# def normImage(image, num=1.):
#     if len(image.shape) > 2:
#         for i in range(3):
#             img = image[:, :, i]
#             max = np.max(img)
#             min = np.min(img)
#             image[:, :, i] = (img - min) / (max - min + 1e-8)
#     else:
#         max = np.max(image)
#         min = np.min(image)
#         image = (image - min) / (max - min + 1e-8) * num
#     return image

# param_dict = mindspore.load_checkpoint(premodel + "studentNet.ckpt")
# mindspore.load_param_into_net(modelG, param_dict)
# import matplotlib.pyplot as plt

# for item in dataloader.create_dict_iterator():
#     testin = item['image']
#     testlabel = item['label']
#     testout,_ = modelG(testin)
#     testin = testin.reshape((3, 256, 256)).transpose([1, 2, 0]).asnumpy()
#     testout = testout.reshape((3,256,256)).transpose([1, 2, 0]).asnumpy()
#     testlabel = testlabel.reshape((3, 256, 256)).transpose([1, 2, 0]).asnumpy()
#     testout = normImage(testout)
#     plt.subplot(221)
#     plt.imshow(testin)
#     plt.subplot(222)
#     plt.imshow(testout)
#     plt.subplot(223)
#     plt.imshow(testlabel)
#     plt.show()
#     break

