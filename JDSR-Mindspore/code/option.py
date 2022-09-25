import argparse
import template
from typing import List


class Args:
    """
    此类仅为方便IDE语法高亮，args并非是此类的实例
    """
    def __init__(self):
        StrList = List[str]
        self.debug:bool = True
        self.template:str = '.'
        self.n_threads:int = 4
        self.cpu:bool = True
        self.n_GPUS:int = 1
        self.seed:int = 1
        self.dir_data:str = '/home/hyacinthe/graduation-dissertation/dataset'
        self.data_url:str = '' # ModelArt云平台专用，为保持兼容性未删去原有的dir_data参数
        self.dir_demo:str = '../test'
        self.data_train:str = 'DIV2K'
        self.data_meta:str = 'DIV2KMETA'
        self.data_test:str = 'DIV2k_valid'
        self.benchmark_noise:bool = True
        self.n_train:int = 800
        self.n_val:int = 14
        self.offset_val:int = 800
        self.ext:str = 'sep'
        self.scale:int = 3
        self.patch_size:int = 48
        self.rgb_range:int = 255
        self.n_colors:int = 3
        self.noise:str = '.'
        self.model_mode:str = '.'
        self.feature_distillation_type: StrList = ['SA', 'CA', 'IA', 'FSP', 'AT', 'fitnet']
        self.chop:bool = True
        self.data_type:str = 'lmdb'
        self.model1:str = 'EDSR_LR2HRSL'
        self.model2:str = 'PAN_MSD2'
        self.model_num:int = 2
        self.act:str = 'relu'
        self.pre_train:str = '.'
        self.train_url:str = '.'
        self.extend:str = '.'
        self.n_resblocks:int = 20
        self.n_feats:int = 64
        self.res_scale:float = 1
        self.shift_mean:bool = True
        self.precision:str = 'single'
        self.reset:bool = True
        self.test_every:int = 1000
        self.epochs:int = 1000
        self.batch_size:int = 16
        self.split_batch:int = 16
        self.self_ensemble:bool = True
        self.test_only:bool = True
        self.gan_k:int = 1
        self.lr1:float = 2e-4
        self.lr2:float = 2e-4
        self.meta_lr:float = 2e-4
        self.lr_decay1:int = 200
        self.lr_decay2:int = 200
        self.lr_decay:int = 200
        self.decay_type1:str = 'step'
        self.decay_type2:str = 'step'
        self.decay_type:str = 'step'
        self.gamma:float = 0.5
        self.optimizer:str = 'ADAM'
        self.momentum:float = 0.9
        self.beta1:float = 0.9
        self.beta2:float = 0.999
        self.epsilon:float = 1e-8
        self.weight_decay:float = 0
        self.pi:float = 3.1415926
        self.alpha:float = 0.5
        self.loss0:str = '1*L1+0.001*ML'
        self.loss1:str = '1*L1+1*ISR+1*Tea_supervised+0.001*ML'
        self.skip_threshold:float = 1e6
        self.save:str = 'pan_x4'
        self.load:str = 'pan_x4'
        self.resume:int = 0
        self.print_model:bool = True
        self.save_models:bool = True
        self.print_every:int = 100
        self.save_results:bool = True
        self.reduction:int = 16
        self.testpath:str = '.'
        self.testset:str = 'MyImage'

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/hyacinthe/graduation-dissertation/dataset',
                    help='dataset directory')
parser.add_argument('--data_url', type=str, default='/datatrain/dataset/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_meta', type=str, default='DIV2KMETA',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K_valid',#'DIV2K',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=14,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep',   #'sep_reset',
                    help='dataset file extension')
parser.add_argument('--scale', type=int, default='3',  # 4
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,  # default 192
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',  # G1
                    help='Gaussian noise std.')
parser.add_argument('--model_mode', type=str, default='.')  # T,S,
parser.add_argument('--feature_distilation_type', type=str, default=['SA', 'CA', 'IA', 'FSP', 'AT', 'fitnet'])  # T,S,

parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--data_type',type=str,default='lmdb',help='type of image')
# Model specifications
# parser.add_argument('--model', default='EDSR_LR2HR',
#                     help='model name')
parser.add_argument('--model1', default='EDSR_LR2HRSL',  # EDSR_LR2HRT2 RCAN
                    help='model name')
parser.add_argument('--model2', default='PAN_MSD2',
                    help='model name')
parser.add_argument('--model_num', type=int, default=2)

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--train_url', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000, #1000
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,  # 1000
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
# parser.add_argument('--lr', type=float, default=2e-4,  # 2e-4
#                     help='learning rate')
parser.add_argument('--lr1', type=float, default=2e-4,  # 2e-4 5e-5
                    help='learning rate')
parser.add_argument('--lr2', type=float, default=2e-4,  # 2e-4  5e-5
                    help='learning rate')
parser.add_argument('--meta_lr', type=float, default=2e-4,  # 1e-3
                    help='learning rate')
parser.add_argument('--lr_decay1', type=int, default=200,  # 200
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay2', type=int, default=200,  # 200
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay', type=int, default=200,  # 200
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type1', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--decay_type2', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--pi', type=float, default=3.1415926,
                    help='sin')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='sin')
parser.add_argument('--loss0', type=str, default='1*L1+0.001*ML',
                    help='loss function configuration')
parser.add_argument('--loss1', type=str, default='1*L1+1*ISR+1*Tea_supervised+0.001*ML',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specificationss
parser.add_argument('--save', type=str, default='pan_x4',
                    help='file name to save')
parser.add_argument('--load', type=str, default='pan_x4',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', default=True, action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', default=True, action='store_true',
                    help='save output results')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# options for test
parser.add_argument('--testpath', type=str, default='.',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='MyImage',  # 'Set5',
                    help='dataset name for testing')

args:Args = parser.parse_args()
template.set_template(args)

#args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
