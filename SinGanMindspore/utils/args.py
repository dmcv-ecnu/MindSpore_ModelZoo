"""get args."""
import ast
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='SinGAN-Dehaze')
# basic parameters
parser.add_argument('--platform', type=str, default='CPU', help='only support GPU and Ascend')  # 部署时修改
parser.add_argument('--device_id', type=int, default=0, help='device id, default is 0.')
parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
parser.add_argument('--is_save_on_master', type=int, default=1,
                    help='Save ckpt on master or all rank, 1 for master, 0 for all ranks. Default: 1')
parser.add_argument('--rank', type=int, default=0, help='Local rank of distributed. Default: 0')
parser.add_argument('--group_size', type=int, default=1, help='World size of device. Default: 1')
parser.add_argument('--model', type=str, default='ResNet', choices=('DepthResNet', 'ResNet', 'UNet'), \
                    help='generator model')
parser.add_argument('--init_type', type=str, default='normal', choices=('normal', 'xavier'), \
                    help='network initialization, default is normal.')
parser.add_argument('--init_gain', type=float, default=0.02, \
                    help='scaling factor for normal, xavier and orthogonal, default is 0.02.')
parser.add_argument('--image_size', type=int, default=256, help='input image_size, default is 256.')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size, default is 1.')
parser.add_argument('--pool_size', type=int, default=50, \
                     help='the size of image buffer that stores previously generated images')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1, default is 0.5.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default is 0.0002.')
parser.add_argument('--lr_policy', type=str, default='linear', choices=('linear', 'constant'), \
                    help='learning rate policy, default is linear')
parser.add_argument('--max_epoch', type=int, default=200, help='epoch size for training, default is 200.')
parser.add_argument('--n_epochs', type=int, default=100, \
                    help='number of epochs with the initial learning rate, default is 100')

# cut_model
parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

parser.add_argument('--nce_idt', type=str2bool, nargs='?', const=True, default=False,
                    help='use NCE loss for identity mapping: NCE(G(Y), Y))')
parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
parser.add_argument('--G_layers', type=str, default='16', help='compute NCE loss on which layers')
parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                    help='how to downsample the feature map')
parser.add_argument('--netF_nc', type=int, default=256)
parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
parser.add_argument('--flip_equivariance',
                    type=str2bool, nargs='?', const=True, default=False,
                    help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
parser.set_defaults(pool_size=0)  # no image pooling

# model parameters
parser.add_argument('--in_planes', type=int, default=3, help='input channels, default is 3.')
parser.add_argument('--ngf', type=int, default=64, help='generator model filter numbers, default is 64.')
parser.add_argument('--gl_num', type=int, default=9, help='generator model residual block numbers, default is 9.')
parser.add_argument('--ndf', type=int, default=64, help='discriminator model filter numbers, default is 64.')
parser.add_argument('--dl_num', type=int, default=3, \
                    help='discriminator model residual block numbers, default is 3.')
parser.add_argument('--slope', type=float, default=0.2, help='leakyrelu slope, default is 0.2.')
parser.add_argument('--norm_mode', type=str, default='batch', choices=('batch', 'instance'), \
                    help='norm mode, default is batch.')
parser.add_argument('--lambda_A', type=float, default=10.0, \
                    help='weight for cycle loss (A -> B -> A), default is 10.')
parser.add_argument('--lambda_B', type=float, default=10.0, \
                    help='weight for cycle loss (B -> A -> B), default is 10.')
parser.add_argument('--lambda_idt', type=float, default=0.5, \
                    help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the '
                         'weight of the identity mapping loss. For example, if the weight of the identity loss '
                         'should be 10 times smaller than the weight of the reconstruction loss,'
                         'please set lambda_identity = 0.1, default is 0.5.')
parser.add_argument('--gan_mode', type=str, default='lsgan', choices=('lsgan', 'vanilla'), \
                    help='the type of GAN loss, default is lsgan.')
parser.add_argument('--pad_mode', type=str, default='CONSTANT', choices=('CONSTANT', 'REFLECT', 'SYMMETRIC'), \
                    help='the type of Pad, default is CONSTANT.')

# additional parameters
parser.add_argument('--data_url', default='./data/haze2clear', \
                    help='path of images (should have subfolders trainA, trainB, testA, testB, etc).')
parser.add_argument('--data_dir', default='testA', choices=('testA', 'testB'), \
                    help='the translation direction of SinGAN.')
parser.add_argument('--train_url', type=str, default='./outputs/SOTS', \
                    help='models are saved here, default is ./outputs.')
parser.add_argument('--load_ckpt', type=ast.literal_eval, default=False, \
                    help='whether load pretrained ckpt')
parser.add_argument('--G_A_ckpt', type=str, default='./outputs/outdoor/ckpt/G_A_200.ckpt', \
                    help='checkpoint file path of G_A.')
parser.add_argument('--G_B_ckpt', type=str, default='./outputs/outdoor/ckpt/G_B_200.ckpt', \
                    help='checkpoint file path of G_B.')
parser.add_argument('--D_A_ckpt', type=str, default='./outputs/outdoor/ckpt/D_A_200.ckpt', \
                    help='checkpoint file path of D_A.')
parser.add_argument('--D_B_ckpt', type=str, default='./outputs/outdoor/ckpt/D_B_200.ckpt', \
                    help='checkpoint file path of D_B.')
parser.add_argument('--save_checkpoint_epochs', type=int, default=10, \
                    help='Save checkpoint epochs, default is 10.')
parser.add_argument('--print_iter', type=int, default=100, help='log print iter, default is 100.')
parser.add_argument('--need_profiler', type=ast.literal_eval, default=False, \
                    help='whether need profiler, default is False.')
parser.add_argument('--save_graphs', type=ast.literal_eval, default=False, \
                    help='whether save graphs, default is False.')
parser.add_argument('--save_imgs', type=ast.literal_eval, default=True, \
                    help='whether save imgs when epoch end')
parser.add_argument('--use_random', type=ast.literal_eval, default=True, \
                    help='whether use random when training, default is True.')
parser.add_argument('--need_dropout', type=ast.literal_eval, default=False, \
                    help='whether need dropout, default is True.')
parser.add_argument('--max_dataset_size', type=int, default=None, \
                    help='max images pre epoch, default is None.')

# export parameters
parser.add_argument("--export_batch_size", type=int, default=1, \
                    help="batch size")
parser.add_argument("--export_file_name", type=str, default="SinGAN-Dehaze", \
                    help="output file name.")
parser.add_argument("--export_file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], \
                    default="AIR", help="file format")
args = parser.parse_args()


def get_args(phase):
    """Define the common options that are used in both training and test."""
    if args.device_num > 1:

        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=args.save_graphs)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=args.device_num)
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform,
                            save_graphs=args.save_graphs, device_id=args.device_id)
        args.rank = 0
        args.device_num = 1

    if args.platform == "GPU":
        context.set_context(enable_graph_kernel=True)

    if args.platform == "Ascend" or args.platform == "CPU":
        args.pad_mode = "CONSTANT"

    if phase != "train" and (args.G_A_ckpt is None or args.G_B_ckpt is None):
        raise ValueError('Must set G_A_ckpt and G_B_ckpt in predict phase!')

    if args.batch_size == 1:
        args.norm_mode = "instance"

    if args.data_url is None:
        raise ValueError('Must set data_url!')

    if args.max_dataset_size is None:
        args.max_dataset_size = float("inf")

    args.n_epochs = min(args.max_epoch, args.n_epochs)
    args.n_epochs_decay = args.max_epoch - args.n_epochs
    args.phase = phase
    return args
