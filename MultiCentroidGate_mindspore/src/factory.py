from ds.dataset_transform import dataset_transform
from ds.dataset_order import dataset_order
from ds.incremental import IncrementalDataset
import importlib

import mindspore
import mindspore.dataset.transforms as C
from mindspore.common import dtype as mstype
from mindspore.dataset import Cifar100Dataset, GeneratorDataset, DistributedSampler


def create_convnet(cfg):
    convnet_type = cfg['convnet']
    if convnet_type == "resnet18_cifar":
        from models.backbones.small.resnet import resnet18
        return resnet18(False), 512
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def create_data(cfg):
    # mindspore.dataset.config.set_enable_autotune(True)
    if cfg['dataset'] == 'CIFAR100':
        train_transform = dataset_transform['CIFAR100']['train']
        test_transform = dataset_transform['CIFAR100']['test']
        trainset = Cifar100Dataset(dataset_dir=cfg['data_folder'], shuffle=True, usage='train')
        testset = Cifar100Dataset(dataset_dir=cfg['data_folder'], shuffle=True, usage='test')
        # 完成数据集增强
        trainset = trainset.map(input_columns='image', operations=train_transform, num_parallel_workers=cfg.num_workers)
        typecast_op = C.TypeCast(mstype.int32)
        trainset = trainset.map(input_columns='fine_label', operations=typecast_op, num_parallel_workers=cfg.num_workers)
        testset = testset.map(input_columns='image', operations=test_transform, num_parallel_workers=cfg.num_workers)
        testset = testset.map(input_columns='fine_label', operations=typecast_op, num_parallel_workers=cfg.num_workers)
        # 构造数据集batch
        trainset = trainset.batch(1, drop_remainder=False)
        testset = testset.batch(1, drop_remainder=False)
        order = dataset_order['CIFAR100'][cfg.class_order_idx]  # 设置每个类别的新id
    print(f"dataset order: {order}")
    return IncrementalDataset(trainset, testset, None, 0, order, cfg.base_classes, cfg.increment)


def create_trainer(cfg, inc_dataset:IncrementalDataset):
    lib = importlib.import_module(f"trainer.{cfg.trainer}")
    return lib.IncModel(cfg, inc_dataset)


def create_network(cfg):
    lib = importlib.import_module(f"models.{cfg['network']}")
    return lib.Model(cfg)


def create_sampler(cfg, dataset, ddp, shuffle):
    # 分布式训练设置的采样器，暂时放弃分布式  Soap
    '''
    if ddp:  # args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        ) 
    else:'''
    if shuffle:
        sampler_train = mindspore.dataset.RandomSampler()
    else:
        sampler_train = mindspore.dataset.SequentialSampler()
    return sampler_train


def create_dataloader(cfg, dataset, ddp, shuffle, drop_last=False):
    mindspore.dataset.config.set_prefetch_size(cfg.batch_size)
    train_sampler = create_sampler(cfg, dataset, ddp, shuffle)
    t_dataset = GeneratorDataset(dataset, 
                                 column_names=["data", "label"], 
                                 sampler=train_sampler, 
                                 num_parallel_workers=cfg.num_workers)
    t_dataset = t_dataset.batch(cfg.batch_size, drop_remainder=drop_last)
    return t_dataset