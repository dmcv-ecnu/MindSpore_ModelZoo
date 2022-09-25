import numpy as np
import mindspore as ms
from . import networks
import mindspore.nn as nn
import mindspore.ops as ops
import cv2
from .losses import LossNetwork, PatchNCELoss
import os
import scipy.misc as misc
import matplotlib.pyplot as plt
import argparse
from argparse import Namespace


def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = ms.tensor.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


def GaussianBlur(batch_img, ksize, sigma=None):
    kernel = getGaussianKernel(ksize, sigma)  # 生成权重
    B, C, H, W = batch_img.shape  # C：图像通道数，group convolution 要用到
    # 生成 group convolution 的卷积核
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1).cuda()
    pad = (ksize - 1) // 2  # 保持卷积前后图像尺寸不变
    # mode=relfect 更适合计算边缘像素的权重
    batch_img_pad = nn.Pad(paddings=[pad, pad, pad, pad], mode='REFLECT')(batch_img)
    weighted_pix = ops.Conv2D(batch_img_pad, weight=kernel, bias=None, stride=1, padding=0, groups=C)
    return weighted_pix


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差

		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[	K_ss K_st
							K_ts K_tt ]
    """

    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = ops.Concat([source, target])  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = ops.ReduceSum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [ops.Exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = ops.ReduceMean(keep_dims=True)(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CUTModel:
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

        parser.add_argument('--nce_idt', type=str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--G_layers', type=str, default='16', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'perceptual', 'G', 'NCE', 'MMD']
        self.visual_names = ['real_A', 'fake_B',  'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.G_layers = [int(i) for i in self.opt.G_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        """
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.loss_network = LossNetwork(vgg_model)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        """

    def data_dependent_initialize(self):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = self.real_A.size(0) // len(self.opt.gpu_ids)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.construct()                    # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = nn.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.construct()                  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
        # real_A1 = self.real_A
        # self.real_A1 = cv2.resize(real_A1,[128,128])
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def construct(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = ops.Concat((self.real_A, self.real_B)) if self.opt.nce_idt else self.real_A
        self.real1 = ops.Concat((self.real_A1, self.real_B1)) if self.opt.nce_idt else self.real_A1
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = ops.ReverseV2(self.real, [3])
                self.real1 = ops.ReverseV2(self.real1, [3])
        # image_path = '/media/gjk/ywj/US-dehaze/contrastive-unpaired-translation-master/datasets/dehaze/b/486_Hazy.png'
        # img = misc.imread(image_path)
        # img = img.transpose(2, 0, 1)
        # img_tensor = torch.FloatTensor(img).unsqueeze(dim=0).cuda()
        # fm, _ = self.netG(img_tensor, self.G_layers, fup=True, encode_only=True)
        # feat_qblur = GaussianBlur(fm,25,25)
        # fm = fm - feat_qblur
        # # fm, _ = model(img_tensor)
        #
        # for i in range(0, 1):
        #     j = 0
        #     w = 8
        #     h = 8
        #     draw_features_average(width=w, height=h, x=fm, savename=str(i) + '_' + str(j), id=j)

        self.fea, self.fake = self.netG(self.real, self.G_layers, fup=True, encode_only=False)
        # self.fea1, self.fake1 = self.netG(self.real, self.G_layers,fup=True,encode_only=False)
        self.fake_B = self.fake[:self.real_A.size(0)]

        if self.opt.nce_idt:
            # idtfea_B = []
            # idtfea_B1 = []
            self.idt_B = self.fake[self.real_A.size(0):]

    def backward_D(self):
        if self.opt.lambda_GAN > 0.0:
            """Calculate GAN loss for the discriminator"""
            fake = self.fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            pred_real = self.netD(self.real_B)
            loss_D_real_unweighted = self.criterionGAN(pred_real, True)

            #print('xxxx',pred_real.shape)
            self.loss_D_real = loss_D_real_unweighted.mean()

            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        else:
            self.loss_D_real, self.loss_D_fake, self.loss_D = 0.0, 0.0, 0.0

    def backward_G(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE+self.loss_NCE_Y)*0.5
        else:
            loss_NCE_both = self.loss_NCE
        # self.loss_L1 = self.criterionIdt(self.fake_B,self.real_B)
        # atl1loss = self.criterionIdt(self.fake_B,self.fake_B1)
        # for i in range(len(self.feaB)):
            # atl1loss += self.criterionIdt(self.feaB[i],self.feaB1[i])
        # self.loss_MMD = 0
        self.loss_MMD = self.MMD_loss(self.fake_B, self.real_B)
        # self.loss_ATloss_L1 = atl1loss
        self.loss_network.eval()
        self.loss_perceptual = self.loss_network(self.real_A, self.fake_B)
        self.loss_G = self.loss_G_GAN + self.loss_perceptual + loss_NCE_both + self.loss_MMD
        # torch.autograd.set_detect_anomaly(True)
        self.loss_G.backward()

    def MMD_loss(self, src, tgt):
        # n_layers = len(self.G_layers)
        feat_q,_ = self.netG(tgt, self.G_layers,fup=True, encode_only=True)
        feat_qblur = GaussianBlur(feat_q,25,25)
        feat_q = feat_q - feat_qblur
        feat_q = ops.Squeeze(feat_q, dim=0)

        feat_k, _ = self.netG(src, self.G_layers, fup=True, encode_only=True)
        feat_kblur = GaussianBlur(feat_k,25,25)
        feat_k = feat_k - feat_kblur
        feat_k = ops.Squeeze(feat_k, dim=0)
        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        MMD_loss = 0.0
        numc = feat_k.shape[0]
        for i in range(numc):
            MMD_loss += mmd(feat_k[i],feat_q[i])

        # feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        # feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        # for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
        #     loss = crit(f_q, f_k) * self.opt.lambda_NCE
        #     self.G_layers

        return MMD_loss / numc

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers,fup=True, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [ops.ReverseV2(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers,fup=True,encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        #     MMD_loss =
        # else:
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
