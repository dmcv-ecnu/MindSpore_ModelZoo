"""SinGAN losses"""
from packaging import version
import mindspore as ms
import numpy as np
from mindspore.nn.probability.distribution import Uniform
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class BCEWithLogits(nn.Cell):
    """
    BCEWithLogits creates a criterion to measure the Binary Cross Entropy between the true labels and
    predicted labels with sigmoid logits.
    Args:
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.
    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `inputs`.
        Otherwise, the output is a scalar.
    """
    def __init__(self, reduction='mean'):
        super(BCEWithLogits, self).__init__()
        if reduction is None:
            reduction = 'none'
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.reduce = False
        if reduction == 'sum':
            self.reduce_mode = ops.ReduceSum()
            self.reduce = True
        elif reduction == 'mean':
            self.reduce_mode = ops.ReduceMean()
            self.reduce = True

    def construct(self, predict, target):
        loss = self.loss(predict, target)
        if self.reduce:
            loss = self.reduce_mode(loss)
        return loss


class GANLoss(nn.Cell):
    """
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    Args:
        mode (str): The type of GAN objective. It currently supports 'vanilla', 'lsgan'. Default: 'lsgan'.
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.
    Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `inputs`.
        Otherwise, the output is a scalar.
    """
    def __init__(self, mode="lsgan", reduction='mean'):
        super(GANLoss, self).__init__()
        self.loss = None
        self.ones = ops.OnesLike()
        if mode == "lsgan":
            self.loss = nn.MSELoss(reduction)
        elif mode == "vanilla":
            self.loss = BCEWithLogits(reduction)
        else:
            raise NotImplementedError(f'GANLoss {mode} not recognized, we support lsgan and vanilla.')

    def construct(self, predict, target):
        target = ops.cast(target, ops.dtype(predict))
        target = self.ones(predict) * target
        loss = self.loss(predict, target)
        return loss


class GeneratorLoss(nn.Cell):
    """
    SinGAN-Dehaze generator loss.
    Args:
        args (class): Option class.
        generator (Cell): Generator of CycleGAN.
        D_A (Cell): The discriminator network of domain A to domain B.
        D_B (Cell): The discriminator network of domain B to domain A.
    Outputs:
        Tuple Tensor, the losses of generator.
    """
    def __init__(self, args, generator, D_A, D_B):
        super(GeneratorLoss, self).__init__()
        self.lambda_A = args.lambda_A
        self.lambda_B = args.lambda_B
        self.lambda_idt = args.lambda_idt
        self.use_identity = args.lambda_idt > 0
        self.dis_loss = GANLoss(args.gan_mode)
        self.rec_loss = nn.L1Loss("mean")
        self.generator = generator
        self.D_A = D_A
        self.D_B = D_B
        self.true = Tensor(True, mstype.bool_)

    def construct(self, img_A, img_B):
        """If use_identity, identity loss will be used."""
        fake_A, fake_B, rec_A, rec_B, identity_A, identity_B = self.generator(img_A, img_B)
        loss_G_A = self.dis_loss(self.D_B(fake_B), self.true)
        loss_G_B = self.dis_loss(self.D_A(fake_A), self.true)
        loss_C_A = self.rec_loss(rec_A, img_A) * self.lambda_A
        loss_C_B = self.rec_loss(rec_B, img_B) * self.lambda_B
        if self.use_identity:
            loss_idt_A = self.rec_loss(identity_A, img_A) * self.lambda_A * self.lambda_idt
            loss_idt_B = self.rec_loss(identity_B, img_B) * self.lambda_B * self.lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        loss_G = loss_G_A + loss_G_B + loss_C_A + loss_C_B + loss_idt_A + loss_idt_B
        return (fake_A, fake_B, loss_G, loss_G_A, loss_G_B, loss_C_A, loss_C_B, loss_idt_A, loss_idt_B)


class DiscriminatorLoss(nn.Cell):
    """
    Cycle GAN discriminator loss.
    Args:
        args (class): option class.
        D_A (Cell): The discriminator network of domain A to domain B.
        D_B (Cell): The discriminator network of domain B to domain A.
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator
    Outputs:
        Tuple Tensor, the loss of discriminator.
    """
    def __init__(self, args, D_A, D_B):
        super(DiscriminatorLoss, self).__init__()
        self.D_A = D_A
        self.D_B = D_B
        self.false = Tensor(False, mstype.bool_)
        self.true = Tensor(True, mstype.bool_)
        self.dis_loss = GANLoss(args.gan_mode)
        self.rec_loss = nn.L1Loss("mean")

    def construct(self, img_A, img_B, fake_A, fake_B):
        D_fake_A = self.D_A(fake_A)
        D_img_A = self.D_A(img_A)
        D_fake_B = self.D_B(fake_B)
        D_img_B = self.D_B(img_B)
        loss_D_A = self.dis_loss(D_fake_A, self.false) + self.dis_loss(D_img_A, self.true)
        loss_D_B = self.dis_loss(D_fake_B, self.false) + self.dis_loss(D_img_B, self.true)
        loss_D = (loss_D_A + loss_D_B) * 0.5
        return loss_D


class PatchNCELoss(nn.Cell):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits(reduction='none')
        self.mask_dtype = ms.uint8 if version.parse(ms.__version__) < version.parse('1.2.0') else ms.bool_
        uniform_range = 0.3
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return  x_noise

    def feature_dropout(self, x):
        attention = ops.ReduceMean(keep_dims=True)(x, axis=1)
        _, max_val = ops.ArgMaxWithValue(axis=1, keep_dims=True)(attention.view(x.size(0), -1))
        threshold = max_val*np.random.uniform(0.7,0.9)
        threshold = threshold.view(x.shape(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def construct(self, feat_q, feat_k):
        # print(feat_q.shape,feat_k.shape)
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = ops.BatchMatMul()(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # print(l_pos.shape)
        # neg logit -- current batch
        # reshape features to batch size
        feat_q = feat_q.view(self.opt.batch_size, -1, dim)
        feat_k = feat_k.view(self.opt.batch_size, -1, dim)
        npatches = feat_q.shape(1)
        # print(feat_q.shape,feat_k.transpose(2, 1).shape)
        l_neg_curbatch = ops.BatchMatMul()(feat_q, feat_k.transpose(2, 1))

        # l_neg_curbatch2 = self.feature_dropout(l_neg_curbatch)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = ops.Eye()(npatches, npatches, self.mask_dtype)[None, :, :]
        l_neg_curbatch = self.feature_based_noise(l_neg_curbatch)
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        # # print(l_neg.shape)
        # k = []
        # for i in range(256):
        #     a=l_pos.t()
        #     b=l_neg[:,i:i+1]
        #     # print(a.shape,b.shape)
        #     smil=torch.mm(a,b)
        #     k.append(smil.cpu().detach().numpy().tolist()[0][0])
        # arr = np.array(k)
        # rank = np.argsort(-arr)
        # # print(,arr)
        # for i in range(25):
        #     index = rank[i]
        #     l_neg_trans=self.feature_based_noise(l_neg[:,index:index+1])
        #     # print('xx',l_neg[:,index:index+1].t(),'yy',l_neg_trans.t())
        #     l_neg[:, index:index + 1] = l_neg_trans

        out = ops.Concat(axis=1)(l_pos, l_neg) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, ops.Zeros()(out.shape(0)))

        return loss


# --- Perceptual loss network  --- #
class LossNetwork(nn.Cell):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def construct(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(nn.MSELoss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)
