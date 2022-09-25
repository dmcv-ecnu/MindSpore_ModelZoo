"""Cycle GAN network."""
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
import mindspore.ops as ops
from .networks import ConvNormReLU, init_weights, Upsample2
from .resnet import ResNetGenerator
from .depth_resnet import DepthResNetGenerator
from .unet import UnetGenerator


def get_generator_G(args):
    """
    This class implements the SinGAN-Dehaze model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    """
    if args.model == "ResNet":
        netG = ResNetGenerator(in_planes=args.in_planes, ngf=args.ngf, n_layers=args.gl_num,
                              alpha=args.slope, norm_mode=args.norm_mode, dropout=args.need_dropout,
                              pad_mode=args.pad_mode)
        init_weights(netG, args.init_type, args.init_gain)
    elif args.model == "DepthResNet":
        netG = DepthResNetGenerator(in_planes=args.in_planes, ngf=args.ngf, n_layers=args.gl_num,
                                   alpha=args.slope, norm_mode=args.norm_mode, dropout=args.need_dropout,
                                   pad_mode=args.pad_mode)
        init_weights(netG, args.init_type, args.init_gain)
    elif args.model == "UNet":
        netG = UnetGenerator(in_planes=args.in_planes, out_planes=args.in_planes, ngf=args.ngf, n_layers=args.gl_num,
                            alpha=args.slope, norm_mode=args.norm_mode, dropout=args.need_dropout)
        init_weights(netG, args.init_type, args.init_gain)
    else:
        raise NotImplementedError(f'Model {args.model} not recognized.')
    return netG


def get_generator_F(args):
    if args.model == 'global_pool':
        netF = PoolingF()
    elif args.model == 'reshape':
        netF = ReshapeF()
    elif args.model == 'sample':
        netF = PatchSampleF(use_mlp=False, init_type=args.init_type, init_gain=args.init_gain, nc=args.netF_nc)
    elif args.model == 'mlp_sample':
        netF = PatchSampleF(use_mlp=True, init_type=args.init_type, init_gain=args.init_gain, nc=args.opt.netF_nc)
    elif args.model == 'strided_conv':
        netF = StridedConvF(init_type=args.init_type, init_gain=args.init_gain)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % args.netF)
    return init_net(netF, args.init_type, args.init_gain, args.gpu_ids)


def get_discriminator(args):
    """Return discriminator by args."""
    net = Discriminator(in_planes=args.in_planes, ndf=args.ndf, n_layers=args.dl_num,
                        alpha=args.slope, norm_mode=args.norm_mode)
    init_weights(net, args.init_type, args.init_gain)
    return net


class Discriminator(nn.Cell):
    """
    Discriminator of GAN.
    Args:
        in_planes (int): Input channel.
        ndf (int): Output channel.
        n_layers (int): The number of ConvNormReLU blocks.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
    Returns:
        Tensor, output tensor.
    Examples:
        >>> Discriminator(3, 64, 3)
    """
    def __init__(self, in_planes=3, ndf=64, n_layers=3, alpha=0.2, norm_mode='batch'):
        super(Discriminator, self).__init__()
        kernel_size = 4
        layers = [
            nn.Conv2d(in_planes, ndf, kernel_size, 2, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha)
        ]
        nf_mult = ndf
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8) * ndf
            layers.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 2, alpha, norm_mode, padding=1))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) * ndf
        layers.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 1, alpha, norm_mode, padding=1))
        layers.append(nn.Conv2d(nf_mult, 1, kernel_size, 1, pad_mode='pad', padding=1))
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class Generator(nn.Cell):
    """
    Generator of CycleGAN, return fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.
    Args:
        G_A (Cell): The generator network of domain A to domain B.
        G_B (Cell): The generator network of domain B to domain A.
        use_identity (bool): Use identity loss or not. Default: True.
    Returns:
        Tensors, fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.
    Examples:
        >>> Generator(G_A, G_B)
    """
    def __init__(self, G_A, G_B, use_identity=True):
        super(Generator, self).__init__()
        self.G_A = G_A
        self.G_B = G_B
        self.ones = ops.OnesLike()
        self.use_identity = use_identity

    def construct(self, img_A, img_B):
        """If use_identity, identity loss will be used."""
        fake_A = self.G_B(img_B)
        fake_B = self.G_A(img_A)
        rec_A = self.G_B(fake_B)
        rec_B = self.G_A(fake_A)
        if self.use_identity:
            identity_A = self.G_B(img_A)
            identity_B = self.G_A(img_B)
        else:
            identity_A = self.ones(img_A)
            identity_B = self.ones(img_B)
        return fake_A, fake_B, rec_A, rec_B, identity_A, identity_B


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.
    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B):
        _, _, lg, _, _, _, _, _, _ = self.network(img_A, img_B)
        return lg


class TrainOneStepG(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        generator (Cell): Generator of CycleGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, G, generator, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.G.D_A.set_grad(False)
        self.G.D_A.set_train(False)
        self.G.D_B.set_grad(False)
        self.G.D_B.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(generator.trainable_params())
        self.net = WithLossCell(G)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_A, img_B):
        weights = self.weights
        fake_A, fake_B, lg, lga, lgb, lca, lcb, lia, lib = self.G(img_A, img_B)
        sens = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.net, weights)(img_A, img_B, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)

        return fake_A, fake_B, ops.depend(lg, self.optimizer(grads_g)), lga, lgb, lca, lcb, lia, lib


class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of Cycle GAN discriminator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(D.trainable_params())
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_A, img_B, fake_A, fake_B):
        weights = self.weights
        ld = self.D(img_A, img_B, fake_A, fake_B)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.D, weights)(img_A, img_B, fake_A, fake_B, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)
        return ops.depend(ld, self.optimizer(grads_d))


class Normalize(nn.Cell):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    """
    def forward(self, x):
    norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
    out = x.div(norm + 1e-7)
    return out
    """
    def construct(self, x):
        norm = ops.Pow()(ops.Pow()(x, self.power).sum(1, keepdims=True), 1. / self.power)
        out = ops.Div()(x, norm + 1e-7)
        return out


def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size)
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0])
        kernel_height = math.ceil(kernel_height / output_size[1])
    return kernel_width, kernel_height


class AdaptiveMaxPool2d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.MaxPool(kernel_size, kernel_size)(x)


class PoolingF(nn.Cell):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [AdaptiveMaxPool2d(1)]
        self.model = nn.SequentialCell(*model)
        self.l2norm = Normalize(2)

    def construct(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Cell):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [ops.AdaptiveAvgPool2D(4)]
        self.model = nn.SequentialCell(*model)
        self.l2norm = Normalize(2)

    def construct(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain)
    return net


class StridedConvF(nn.Cell):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.SequentialCell(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def construct(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            # x = F.instance_norm(x)
            x = nn.InstanceNorm2d(x, momentum=0.9)
        return self.l2_norm(x)


class PatchSampleF(nn.Cell):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.SequentialCell(*[nn.Dense()(input_nc, self.nc), nn.ReLU(), nn.Dense(self.nc, self.nc)])
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def construct(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = ops.Randperm()(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class G_Resnet(nn.Cell):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def construct(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon


##################################################################################
# Encoder and Decoders
##################################################################################
class E_adaIN(nn.Cell):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        # style encoder
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def construct(self, image):
        style = self.enc_style(image)
        return style


class StyleEncoder(nn.Cell):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        # self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [ops.AdaptiveAvgPool2D(1)]
        if self.vae:
            # self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            # self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_mean = nn.Dense(dim, style_dim)
        else:
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0, padding=0, has_bias=True)]

        self.model = nn.SequentialCell(*self.model)
        self.output_dim = dim

    def construct(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


class ContentEncoder(nn.Cell):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.SequentialCell(*self.model)
        self.output_dim = dim

    def construct(self, x, nce_layers=[], encode_only=False):
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None


class Decoder_all(nn.Cell):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [Upsample2(scale_factor=2), Conv2dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.SequentialCell(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def construct(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class Decoder(nn.Cell):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample2(scale_factor=2), Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.SequentialCell(*self.model)

    def construct(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)

##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Cell):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.SequentialCell(*self.model)

    def construct(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = ops.Concat()([x, y_expand], 1)
    return x_cat


class ResBlock(nn.Cell):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.SequentialCell(*model)

    def construct(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Cell):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.Pad(padding, mode="REFLECTION")
        elif pad_type == 'zero':
            self.pad = nn.Pad(padding, mode="CONSTANT")
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = ops.SeLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, has_bias=self.use_bias)

    def construct(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Cell):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Dense(input_dim, output_dim, has_bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim, momentum=0.9)
        # ms无对应接口
        # elif norm == 'inst':
        #    self.norm = nn.InstanceNorm1d(norm_dim)

        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = ops.SeLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def construct(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################
class GetMatrix(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def construct(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class LayerNorm(nn.Cell):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = ms.Parameter(ms.Tensor(num_features).uniform_())
            self.beta = ms.Parameter(ops.Zeros(num_features))

    def construct(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class NONLocalBlock2D(nn.Cell):
    def __init__(self):
        super(NONLocalBlock2D, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    def construct(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        """
        batch_size = source.size(0)
        c = source.size(1)
        g_source = source.view(batch_size, c, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = ops.BatchMatMul(weight.to_dense(), g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, c, *source.size()[2:])
        return y


class ResidualBlock(nn.Cell):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.SequentialCell(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, has_bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, has_bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine, momentum=0.9)
        )

    def construct(self, x):
        return x + self.main(x)


class dehaze(nn.Cell):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(dehaze, self).__init__()
        layers = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, has_bias=False),
            nn.InstanceNorm2d(64, affine=True, momentum=0.9),
            nn.ReLU()
        )
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            layers = nn.SequentialCell(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(),
            )

            setattr(self, f'pnet_down_{i+1}', layers)
            curr_dim = curr_dim * 2

        # Bottleneck. All bottlenecks share the same attention module
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        # self.simple_spade = GetMatrix(curr_dim, 1)  # get the makeup matrix

        for i in range(3):
            setattr(self, f'pnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))

        # --------------------------- TNet(MANet) for applying makeup transfer ----------------------------

        self.tnet_in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False, momentum=0.9)
        self.tnet_in_relu = nn.ReLU()

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            setattr(self, f'tnet_down_conv_{i+1}',
                    nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, has_bias=False))
            setattr(self, f'tnet_down_spade_{i+1}', nn.InstanceNorm2d(curr_dim * 2, affine=False))
            setattr(self, f'tnet_down_relu_{i+1}', nn.ReLU())
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(6):
            setattr(self, f'tnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        # Up-Sampling
        for i in range(2):
            setattr(self, f'tnet_up_conv_{i+1}',
                    nn.Conv2dTranspose(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, has_bias=False))
            setattr(self, f'tnet_up_spade_{i+1}', nn.InstanceNorm2d(curr_dim // 2, affine=False, momentum=0.9))
            setattr(self, f'tnet_up_relu_{i+1}', nn.ReLU())
            curr_dim = curr_dim // 2

        layers = nn.SequentialCell(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, has_bias=False),
            nn.Tanh()
        )
        self.tnet_out = layers

    def atten_feature(mask_s, weight, gamma_s, beta_s, atten_module_g, atten_module_b):
        """
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        channel_num = gamma_s.shape[1]
        mask_s_re = nn.ResizeBilinear()(mask_s, size=gamma_s.shape[2:]).repeat(1, channel_num, 1, 1)
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w)
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)
        beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(self, mask_c, mask_s, fea_c, fea_s, diff_c, diff_s):
        """  s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        """
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None   # fea_s when i==3
        # get 3 part fea using mask
        channel_num = fea_s.shape[1]

        mask_c_re = nn.ResizeBilinear()(mask_c, size=64).repeat(1, channel_num, 1, 1)  # (3, c, h, w)
        fea_c = fea_c.repeat(3, 1, 1, 1)                 # (3, c, h, w)
        fea_c = fea_c * mask_c_re                        # (3, c, h, w) 3 stands for 3 parts

        mask_s_re = nn.ResizeBilinear()(mask_s, size=64).repeat(1, channel_num, 1, 1)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re

        theta_input = ops.Concat(axis=1)(fea_c * 0.01, diff_c)
        phi_input = ops.Concat(axis=1)(fea_s * 0.01, diff_s)

        theta_target = theta_input.view(batch_size, -1, HW) # (N, C+136, H*W)
        theta_target = theta_target.permute(0, 2, 1)        # (N, H*W, C+136)

        phi_source = phi_input.view(batch_size, -1, HW)     # (N, C+136, H*W)
        self.track("before mask")

        weight = ops.BatchMatMul()(theta_target, phi_source)        # (3, HW, HW)
        self.track("among bmm")
        v = weight.detach().nonzero().long().permute(1, 0)
        weight_ind = v.clone()
        del v
        weight *= 200                                       # hyper parameters for visual feature
        weight = ops.Softmax(axis=-1)(weight)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        # ret = torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))
        ret = ms.SparseTensor(weight_ind, weight, (3, HW, HW))
        self.track("after bmm")
        return ret

    def construct(self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        c, s, mask_c, mask_s, diff_c, diff_s = [x.squeeze(0) if x.ndim == 5 else x for x in [c, s, mask_c, mask_s, diff_c, diff_s]]
        """attention version
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """

        self.track("start")
        # forward c in tnet(MANet)
        c_tnet = self.tnet_in_conv(c)    #hazy
        s = self.pnet_in(s)              #clear
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f'pnet_down_{i+1}')
                s = cur_pnet_down(s)

            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i+1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i+1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i+1}')
            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)
        self.track("downsampling")

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:               # not in test_mix
                    s, gamma, beta = self.simple_spade(s)
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)
                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta, self.atten_bottleneck_g, self.atten_bottleneck_b)
                    if ret:
                        return [gamma, beta]
                # else:                       # in test mode
                    # gamma, beta = param_A[0]*w + param_B[0]*(1-w), param_A[1]*w + param_B[1]*(1-w)

                c_tnet = c_tnet * (1 + gamma) + beta    # apply makeup transfer using makeup matrices

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)
        self.track("bottleneck")

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i+1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i+1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i+1}')
            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)
        self.track("upsampling")

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet


class PatchDiscriminator(nn.Cell):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def construct(self, input):
        B, C, H, W = input.shape(0), input.shape(1), input.shape(2), input.shape(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().construct(input)


class GroupedChannelNorm(nn.Cell):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def construct(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)
