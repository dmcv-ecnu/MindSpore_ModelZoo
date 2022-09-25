import numpy as np
import argparse
import option as op
import mindspore.dataset as ds
from model.net import LatticeNet
from mindspore import context, load_checkpoint, load_param_into_net
import math
import matplotlib.pyplot as plt
from data.benchmark import Benchmark


def calc_psnr(sr, hr, scale, rgb_range):
    """calculate psnr"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
    mse = np.mean(pow(valid, 2))
    return -10 * math.log10(mse)

def calc_ssim(img1, img2, scale):
    """calculate ssim value"""
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    border = 0
    if scale != 1:
        border = scale
    else:
        border = scale + 6
    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]
    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == "__main__":
    # 设置硬件参数
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"

    # 加载数据库
    op.args.test_only = True
    train_dataset = Benchmark(op.args, name=op.args.data_test)
    train_dataset.set_scale(op.args.task_id)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"],  shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)


    # print("Init data successfully")

    #加载模型

    # net = WDSR()
    net = LatticeNet(op.args)
    # net = LeNet5()
    # net = nn.Conv2d(3, 64, 1, has_bias=False, weight_init='normal')
    # print("Init net successfully")
    param_dict = load_checkpoint("./premodel/LatticeNet_12-1_500.ckpt")
    # 将参数加载到网络中
    load_param_into_net(net, param_dict)

    def imgshow(result, x, y, id, name):
        result = result.squeeze()
        result = result.transpose()
        result = result.asnumpy()
        result = result.astype('uint8')
        plt.title(name)
        plt.subplot(x, y, id)
        plt.imshow(result)

    avgPsnr = 0

    for data in train_de_dataset.create_dict_iterator():
        # print("LR shape: {}".format(data['LR'].shape), ", HR: {}".format(data['HR'].shape))
        plt.figure()
        imgshow(data['HR'], 2, 2, 1, 'HR')
        imgshow(data['LR'], 2, 2, 2, 'LR')
        resultLr = net(data['LR'])
        imgshow(resultLr, 2, 2, 3, 'result')
        plt.show()

        psnr = calc_psnr(resultLr.asnumpy(), data['HR'].asnumpy(), 2, 255.0)
        avgPsnr += psnr
        print("current psnr: ", psnr)
    # 测试网络是否正确输出channels数
    # input_x = Tensor(np.ones([1, 3, 48, 48]), mindspore.float32)
    # print(net(input_x).shape)
    print("average psnr : ", avgPsnr/5)


    # step_size = train_de_dataset.get_dataset_size()
    #
    # # 优化器 损失函数
    # opt = nn.Adam(params=net.trainable_params(), learning_rate=2e-4)
    # loss = nn.L1Loss()
    # print("Init opt and lossfunction successfully")
    #
    # time_cb = TimeMonitor(data_size=step_size)
    # loss_cb = LossMonitor()
    # cb = [time_cb, loss_cb]
    # config_ck = CheckpointConfig(save_checkpoint_steps=50,keep_checkpoint_max=40)
    # ckpt_cb = ModelCheckpoint(prefix="LatticeNet", directory='./premodel', config=config_ck)
    # cb += [ckpt_cb]
    #
    # model = Model(net, loss_fn=loss, optimizer=opt)
    #
    # model.train(op.args.epochs, train_de_dataset, dataset_sink_mode=True, callbacks=cb)