import mindspore as ms
import mindspore.ops as ops
import os
# import utility
# import data
# import model
import numpy as np
from .losses import LossNetwork
# from option import args
# from trainer import Trainer
# from model import rcan,rcan_stu,smallsr,smallsr_8PB
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
# from torchvision.models import vgg16

"""
从mindspore加载VGG16
import mindspore
import mindspore_hub as mshub
from mindspore import Tensor
from mindspore import nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import py_transforms

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    device_id=0)

model = "mindspore/ascend/1.3/vgg16_v1.3_imagenet2012"
# initialize the number of classes based on the pre-trained model
network = mshub.load(model, num_classes=1000, dataset="imagenet2012")
network.set_train(False)
"""


def statistics(map,save_path,save_name):
    m = []
    v = []
    for i in range(map.shape[2]):
        img_single = map[:, :, i]
        m.append(np.mean(img_single))
        v.append(np.std(img_single))
    plt.figure()
    plt.plot(m, label='mean')
    plt.title('mean')
    # plt.savefig('../feats/m.png')
    plt.plot(v, label='s. d.')
    plt.title('mean and standard deviation')
    plt.legend(['mean', 's. d.'])
    plt.savefig(os.path.join(save_path, '{} m&v.png'.format(save_name)))


def draw_features(width, height, x, savename, id):

    fig = plt.figure(figsize=(width, height))
    p1 = '/media/gjk/ywj/US-dehaze/lookkkkkk/'+str(id)+'/'
    if not os.path.exists(p1):
        os.mkdir(p1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :].detach().cpu().numpy()

        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
    fig.savefig(p1+savename+'.png', dpi=100)
    fig.clf()
    plt.close()
    fm = x[0, :, :, :]
    fm = fm.byte().permute(1, 2, 0).cpu().numpy()
    statistics(fm, save_path=p1, save_name=savename)


def draw_features_average(width, height, x, savename, id):
    fig = plt.figure(figsize=(1, 1))
    p1 = '/media/gjk/ywj/US-dehaze/lookkkkkk/'+str(id)+'/'
    if not os.path.exists(p1):
        os.mkdir(p1)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # print(x.shape)

    plt.subplot(1, 1, 1)
    plt.axis('off')
    img = x[0, :, :, :].detach().cpu().numpy()
    img = np.mean(img, axis=0)
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
    plt.imshow(img)
        #print("{}/{}".format(i, width * height))
    fig.savefig(p1+savename+'.png', dpi=100)
    fig.clf()
    plt.close()

    fm = x[0, :, :, :]
    fm = fm.byte().permute(1, 2, 0).cpu().numpy()

    # statistics(fm,save_path=p1,save_name=savename)

# model = smallsr_8PB.SMALLSR(args)
# model =  rcan.RCAN(args)
# model_path ='/home/zjt/Desktop/KDLW_SR/T_model/RCAN_BIX4.pt'
# model_path='/home/zjt/Desktop/KDLW_SR/experiment/到 lstmsr_log 的链接/test8PB/model/model_best.pt'

"""
image_path  = '/home/zjt/Desktop/my_storge/benchmark/Set5/LR_bicubic/X4/butterfly_GT.png'
vgg_model = vgg16(pretrained=True).features[:16]
# vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
model = LossNetwork1(vgg_model)
model.eval()
"""

"""
# model = model.cuda()
# model.load_state_dict(torch.load(model_path))
#
img = misc.imread(image_path)
img = img.transpose(2, 0, 1)
img_tensor = ms.context(ops.ExpandDims(ms.FloatTensor(img), 0))
fm = model(img_tensor)
# fm, _ = model(img_tensor)

for i in range(0, 8):
    for j in range(0, 5):
        if j == 2:
            w = 4
            h = 4
        elif j == 1 or j == 3:  # or j==4:
            w = 4
            h = 8
        else:
            w=8
            h=8
        draw_features_average(width=w, height=h, x=fm[i][j], savename=str(i)+'_'+str(j), id=j)

# draw_features(width=8, height=8, x=fm,savename='head',id=1)
"""
