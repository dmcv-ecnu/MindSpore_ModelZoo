from ast import arg
from cProfile import run
import mindspore as ms

import dataset
import model
import loss

from option import Args, args

from mindspore import nn
from mindspore.train.callback import LossMonitor
from mindspore import Model

from option import args
import dataset
from model import edn

def run_test_stage1(args:Args): # 低阶API，仅用作开发过程中的简单测试
    from mindspore import load_checkpoint, load_param_into_net
    net = edn.make_model(args)
    param_dict = load_checkpoint("/home/hyacinthe/Downloads/EDN.ckpt")
    load_param_into_net(net, param_dict)
    psnr = nn.PSNR()

    ds_train = dataset.create_dataset_DIV2K(args)
    iterator_ds = ds_train.create_dict_iterator()
    i = 0
    for column in iterator_ds:
        if i > 0: # 只测试一张图片
            break
        data = column['hr']
        lr = column[f'lrx{args.scale}']
        result = net(data) # 数组[batch_size * n_color * height * width, batch_size * n_color * height * width]

        slr = result[0][0]
        sr = result[1][0]
        
        #print(ms.Tensor(slr[0][0]).shape())
        import numpy as np
        from PIL import Image
        img:ms.Tensor = slr.transpose(1, 2, 0)
        import utility
        print(type(data))
        img = utility.quantize(img, args.rgb_range)
        value = utility.calc_psnr(result[0],lr, args.scale, args.rgb_range)
        
        print(value)
        print(img.shape)
        #Image.fromarray((img.asnumpy()).astype(np.uint8)).save(f"slr2.png")
        #img:ms.Tensor = sr.transpose(1, 2, 0)
        #Image.fromarray((img.asnumpy()).astype(np.uint8)).save(f"hr2.png")

        i = i + 1

def low_level(args:Args): # 仅用于开发过程中的测试
    epoch = 10 # args.epoch
    net = edn.make_model(args)
    loss = edn.LossEDN()
    loss_net = edn.EDNWithLoss(net, loss) # 构造模型与Loss的封装

    lr = nn.CosineDecayLR(0.0000125, 0.0002, epoch * 800 // args.batch_size)
    
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    weights = optimizer.parameters
    grad_op = ms.ops.GradOperation(get_by_list=True)
    grad_reducer = ms.ops.functional.identity
    ds_train = dataset.create_dataset_DIV2K(args)
        

    i = 1
    while i <= epoch:
        iterator_ds = ds_train.create_dict_iterator()
        j = 1
        for columns in iterator_ds:
            lr = columns['lrx3']
            hr = columns['hr']
            #output = net(hr)
            #loss1 = loss(output[0], output[1], lr, hr)
            loss = loss_net(lr, hr)
            #lr_rate = optimizer.get_lr()
            #print(f'学习率：{lr_rate}')
            grads = grad_op(loss_net, weights)( lr, hr)
            grads = grad_reducer(grads)
            loss = ms.ops.functional.depend(loss, optimizer(grads))
            print(f"epoch: {i}, step: {j}, loss: {loss}")
            j = j + 1

        i = i + 1
    ms.save_checkpoint(net, './EDN-new.ckpt')

def run_test_stage2(args:Args):
    from mindspore import load_checkpoint, load_param_into_net
    from model import edsr
    net = edsr.EDSR()
    param_dict = load_checkpoint("/home/hyacinthe/EDSR.ckpt")
    load_param_into_net(net, param_dict)
    
    ds_train = dataset.create_dataset_DIV2K_test(args)
    iterator_ds = ds_train.create_dict_iterator()
    i = 0
    for column in iterator_ds:
        if i > 0:
            break
        data = column['lrx3']
        hr:ms.Tensor = net(data)[0][-1][0] # 数组[batch_size * n_color * height * width, batch_size * n_color * height * width]
        print(hr.shape)
        #print(ms.Tensor(slr[0][0]).shape())
        import numpy as np
        from PIL import Image
        img:ms.Tensor = hr.transpose(1, 2, 0)
        #print(img.shape)
        Image.fromarray((img.asnumpy()).astype(np.uint8)).save("hr.png")
        i = i + 1

run_test_stage2()