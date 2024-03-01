# from data_loader_test import TestMVTecDataset
# import mindspore.numpy as mnp
# from data_loader import TrainImageOnlyDataset, TrainWholeImageDataset, MVTecImageAnomTrainDataset
# import math
# import numpy as np
# from dsr_model import FeatureEncoder, FeatureDecoder, ResidualStack, ImageReconstructionNetwork, UnetEncoder, UnetModel
# from mindspore import Tensor
# import mindspore
# from mindspore.common.initializer import One, Normal
# import mindspore.nn as nn
# import mindspore.ops as ops
# from discrete_model import DiscreteLatentModel, VectorQuantizerEMA, EncoderBot, EncoderTop, DecoderBot
# import subprocess

# mvtec_path = "/home/yyk/datasets/mvtec_anomaly_detection/"
# obj_name = "zipper"
# img_dim = 224
# dataset = TestMVTecDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim,img_dim])

# print(type(dataset))
# print(len(dataset))
# print(type(dataset[0]))
# ex = dataset[0]
# image = ex["image"]
# has_anomaly = ex["has_anomaly"]
# mask = ex["mask"]
# idx = ex["idx"]
# print(type(idx))
# print(mask.shape)


# path = "/home/yyk/datasets/mvtec_anomaly_detection/zipper/train/good"
# myDataset = TrainImageOnlyDataset(path, (224, 224))
# print(len(myDataset))
# print(myDataset[0]["image"].shape)
# print(myDataset[0]["idx"])

# new_idx = mnp.randint(0, 10).asnumpy()[0]
# print(type(mnp.randint(0, 10)))
# print(new_idx)

# do_aug_orig = mnp.rand() > 0.6
# print(do_aug_orig)
# if do_aug_orig:
#     print("ok")

# perlin_scalex = 2 ** mnp.randint(0, 2).item()
# perlin_scaley = 2 ** mnp.randint(0, 2).item()
# print(perlin_scalex)

# myDataset = TrainWholeImageDataset("/home/yyk/datasets/mvtec_anomaly_detection/zipper/train/good", (224, 224), True)
# print(len(myDataset))
# print(myDataset[0])

# perlin_scalex = 2 ** mnp.randint(0, 2).astype('int').item()
# perlin_scaley = 2 ** mnp.randint(0, 2).astype('int').item()
# print(perlin_scalex)
# beta = mnp.rand() * 0.8
# print(beta)
# idx = mnp.randint(0, 10).item()
# print(idx)

# question
# myDataset = MVTecImageAnomTrainDataset("/home/yyk/datasets/mvtec_anomaly_detection/zipper/train/good", (224, 224))
# print(len(myDataset))
# temp = myDataset[0]

# no_anomaly = mnp.rand()
# print(no_anomaly)
# if no_anomaly > 0.5:
#     print("no")

# ex = UnetModel()
# print(ex)
# t = Tensor(shape = (1, 64, 224, 224), dtype=mindspore.float32, init=One())
# print(t)
# out = ex(t)
# print(type(out))
# print(out.shape)

# embedding_weight = mindspore.Tensor(np.random.randn(100, 10), dtype=mindspore.float32)
# embedding = nn.Embedding(100, 10, embedding_table=embedding_weight)
# print(type(embedding))

# a = mindspore.numpy.randn((5, 5))
# print(type(a))

# b = mindspore.Parameter(a) 
# print(type(b))


# num_hiddens = 128
# num_residual_hiddens = 64
# num_residual_layers = 2
# embedding_dim = 128
# num_embeddings = 4096
# commitment_cost = 0.25
# decay = 0.99
# anom_par = 0.2

# model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay)
# myModel = model._vq_vae_top
# youModel = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
#                                 commitment_cost, decay)
# a = Tensor(shape = (8, 128, 32, 32), dtype=mindspore.float32, init=One())
# e_1, e_2, _, _ = myModel(a)
# # print(type(e_1))
# # print(b.shape)

# optimizer = mindspore.nn.Adam([
#     {"params": model.trainable_params(), "lr": 0.01},
#     {"params": youModel.trainable_params(), "lr": 0.01},
# ])

# optimizer = mindspore.nn.Adam(myModel.trainable_params(), learning_rate=0.01)

# a = Tensor(shape = (1, 3, 224, 224), dtype=mindspore.float32, init=One())
# b = mindspore.nn.Softmax(axis=1)
# a = b(a)
# a = a.float()
# print(ytpe(a))

# path = "/home/yyk/codes/DSR_anomaly_detection-main/train_dsr.py"
# # Use: python temp_train.py --gpu_id 0 --obj_id 0 --lr 0.0002 --bs 8 --epochs 100 --data_path "/home/yyk/datasets/mvtec_anomaly_detection/" --out_path "/home/yyk/codes/DSR_anomaly_detection-main/temp_dir"
# additional_args = [0, 1, 0.0002, 100, 0, "/home/yyk/datasets/mvtec_anomaly_detection/", "/home/yyk/codes/DSR_anomaly_detection-main/temp_dir"]
# subprocess.run(['python', path] + additional_args)


import torch
# 创建一个需要梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 对张量进行操作，得到新的张量y
y = x * 2

# 使用detach()函数分离y，得到一个不需要梯度的张量z
z = y.detach()

# 打印y和z的地址
print(id(y))  # 输出张量y的地址
print(id(z))  # 输出张量z的地址
print()