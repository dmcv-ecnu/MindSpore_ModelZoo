#! /usr/bin/env python

from PIL import Image # 图片类
import os
import glob
import re
from functools import reduce
import random
import numpy as np
import mindspore.dataset as ds

class FolderImagePair:
    def __init__(self, dir_patterns, reader=None):
        self.dir_patterns = dir_patterns
        self.reader = reader
        self.pair_keys, self.image_pairs = self.scan_pair(self.dir_patterns) # 图片对的编号，图片对的路径数组

    @staticmethod
    def scan_pair(dir_patterns):
        images = []
        for _dir in dir_patterns:
            imgs = glob.glob(_dir) # 根据通配符转换成真实存在的linux路径
            _dir = os.path.basename(_dir)
            pat = _dir.replace("*", "(.*)").replace("?", "(.?)") # 把shell通配符转换成正则表达式
            pat = re.compile(pat, re.I | re.M)
            keys = [re.findall(pat, os.path.basename(p))[0] for p in imgs] # 图片路径文件名（不含扩展名）findall仅返回括号内的
            images.append({k: v for k, v in zip(keys, imgs)}) # 字典，key为0001这样的，value是路径
        same_keys = reduce(lambda x, y: set(x) & set(y), images) # 求交集
        same_keys = sorted(same_keys)
        image_pairs = [[d[k] for d in images] for k in same_keys] # 二维数组，k行，每一行是一张图片的各种清晰度
        same_keys = [x if isinstance(x, str) else "_".join(x) for x in same_keys] # 确保key是字符串
        return same_keys, image_pairs

    def get_key(self, idx):
        return self.pair_keys[idx]

    def __getitem__(self, idx): # 魔法函数，实现后可以当数组用
        if self.reader is None:
            images = [Image.open(p) for p in self.image_pairs[idx]] # 打开文件
            images = [img.convert('RGB') for img in images] # 转换成RGB
            images = [np.array(img) for img in images] # 转换成numpy数组
        else:
            images = [self.reader(p) for p in self.image_pairs[idx]]
        pair_key = self.pair_keys[idx]
        return (pair_key, *images) # images被拆成各个元素

    def __len__(self):
        return len(self.pair_keys)


class LrHrImages(FolderImagePair):
    def __init__(self, args, lr_pattern:str, hr_pattern, train=True, reader=None):
        #self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.repeat = 1
        self.train = train
        self.hr_pattern = hr_pattern
        self.lr_pattern = lr_pattern
        self.dir_patterns = []
        if isinstance(self.lr_pattern, str): # 非str数组
            self.is_multi_lr = False
            self.dir_patterns.append(self.lr_pattern)
        elif len(lr_pattern) == 1: # 是str数组但长度为1
            self.is_multi_lr = False
            self.dir_patterns.append(self.lr_pattern[0])
        else: # 是str数组且长度不为1
            self.is_multi_lr = True
            self.dir_patterns.extend(self.lr_pattern)
        self.dir_patterns.append(self.hr_pattern)
        super(LrHrImages, self).__init__(self.dir_patterns, reader=reader)
    
    def __len__(self):
        if self.train:
            return len(self.image_pairs) * self.repeat // 2
        else:
            return len(self.image_pairs)
    
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.image_pairs)
        else:
            return idx

    def __getitem__(self, idx):
        _, *images1 = super(LrHrImages, self).__getitem__(idx) # 第一个元素pair_key赋值给匿名变量丢弃，剩余部分全部赋值给images，images变成数组（列表）
        _, *images2 = super(LrHrImages, self).__getitem__(idx + self.__len__) # __len__只有实际长度的一半，等于是把数据集分成两部分，images2是后半部分
        return tuple(images1 + images2) # 根据datasetgenerator类的需求将其转换成元组

class _BasePatchCutter:
    """
    cut patch from images
    patch_size(int): patch size, input images should be bigger than patch_size.
    lr_scale(int/list): lr scales for input images. Choice from [1,2,3,4, or their combination]
   """
    def __init__(self, patch_size, lr_scale):
        self.patch_size = patch_size
        self.multi_lr_scale = lr_scale
        if isinstance(lr_scale, int):
            self.multi_lr_scale = [lr_scale]
        else:
            self.multi_lr_scale = [*lr_scale]
        self.max_lr_scale_idx = self.multi_lr_scale.index(max(self.multi_lr_scale))
        self.max_lr_scale = self.multi_lr_scale[self.max_lr_scale_idx]

    def get_tx_ty(self, target_height, target_weight, target_patch_size):
        raise NotImplementedError()

    def __call__(self, *images):
        target_img = images[self.max_lr_scale_idx]

        tp = self.patch_size // self.max_lr_scale
        th, tw, _ = target_img.shape

        tx, ty = self.get_tx_ty(th, tw, tp)

        patch_images = []
        for _, (img, lr_scale) in enumerate(zip(images, self.multi_lr_scale)):
            x = tx * self.max_lr_scale // lr_scale
            y = ty * self.max_lr_scale // lr_scale
            p = tp * self.max_lr_scale // lr_scale
            patch_images.append(img[y:(y + p), x:(x + p), :])
        return tuple(patch_images)


class RandomPatchCutter(_BasePatchCutter):

    def __init__(self, patch_size, lr_scale):
        super(RandomPatchCutter, self).__init__(patch_size=patch_size, lr_scale=lr_scale)

    def get_tx_ty(self, target_height, target_weight, target_patch_size):
        target_x = random.randrange(0, target_weight - target_patch_size + 1)
        target_y = random.randrange(0, target_height - target_patch_size + 1)
        return target_x, target_y


class CentrePatchCutter(_BasePatchCutter):

    def __init__(self, patch_size, lr_scale):
        super(CentrePatchCutter, self).__init__(patch_size=patch_size, lr_scale=lr_scale)

    def get_tx_ty(self, target_height, target_weight, target_patch_size):
        target_x = (target_weight - target_patch_size) // 2
        target_y = (target_height - target_patch_size) // 2
        return target_x, target_y


def hflip(img):
    return img[:, ::-1, :]


def vflip(img):
    return img[::-1, :, :]


def trnsp(img):
    return img.transpose(1, 0, 2)


AUG_LIST = [
    [],
    [trnsp],
    [vflip],
    [vflip, trnsp],
    [hflip],
    [hflip, trnsp],
    [hflip, vflip],
    [hflip, vflip, trnsp],
]


AUG_DICT = {
    "0": [],
    "t": [trnsp],
    "v": [vflip],
    "vt": [vflip, trnsp],
    "h": [hflip],
    "ht": [hflip, trnsp],
    "hv": [hflip, vflip],
    "hvt": [hflip, vflip, trnsp],
}


def flip_and_rotate(*images):
    aug = random.choice(AUG_LIST)
    res = []
    for img in images:
        for a in aug:
            img = a(img)
        res.append(img)
    return tuple(res)

def hwc2chw(*images):
    res = [i.transpose(2, 0, 1) for i in images]
    return tuple(res)


def uint8_to_float32(*images):
    res = [(i.astype(np.float32) if i.dtype == np.uint8 else i) for i in images]
    return tuple(res)


def create_dataset_DIV2K(args, dataset_type="train", num_parallel_workers=10, shuffle=True):
    num_parallel_workers = 1
    
    dataset_path:str = args.dir_data # 默认"/home/hyacinthe/graduation-dissertation/dataset/DIV2K"
    lr_scale:int = args.scale # 默认3倍放大
    lr_type:str = "bicubic" # 默认用传统缩放的数据集
    batch_size:int = args.batch_size # batch的大小论文说最小为16，暂时先设置这么多
    patch_size:int = args.patch_size # 图像的大小
    epoch_size:int = args.epochs # 训练的趟数
    
    # get HR_PATH/*.png
    dir_div2k = os.path.join(dataset_path, "DIV2K")
    dir_hr = os.path.join(dir_div2k, f"DIV2K_{dataset_type}_HR") # 高分图片，train或valid
    hr_pattern = os.path.join(dir_hr, "*.png")

    # get LR_PATH/X2/*x2.png, LR_PATH/X3/*x3.png, LR_PATH/X4/*x4.png # 低分图片，包括train和valid
    column_names = []
    lrs_pattern = []
    dir_lr = os.path.join(dir_div2k, f"DIV2K_{dataset_type}_LR_{lr_type}", f"X{lr_scale}")
    lr_pattern = os.path.join(dir_lr, f"*x{lr_scale}.png")
    lrs_pattern.append(lr_pattern)
    column_names.append("lr")
    column_names.append("hr")  # ["lrx2","lrx3","lrx4",..., "hr"]
    # make dataset
    dataset = LrHrImages(lr_pattern=lrs_pattern, hr_pattern=hr_pattern)

    # make mindspore dataset
    if dataset_type == "train":
        generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                num_parallel_workers=num_parallel_workers,
                                                shuffle=shuffle and dataset_type == "train")
    else:
        #sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=False, offset=0)
        generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                num_parallel_workers=num_parallel_workers,
                                                shuffle=shuffle)

    # define map operations
    if dataset_type == "train":
        transform_img = [
            RandomPatchCutter(patch_size, [lr_scale, 1]),
            flip_and_rotate,
            hwc2chw,
            uint8_to_float32,
        ]
    elif patch_size > 0:
        transform_img = [
            CentrePatchCutter(patch_size, [lr_scale, 1]),
            hwc2chw,
            uint8_to_float32,
        ]
    else:
        transform_img = [
            hwc2chw,
            uint8_to_float32,
        ]

    # pre-process hr lr
    generator_dataset = generator_dataset.map(input_columns=column_names,
                                                  output_columns=column_names,
                                                  column_order=column_names,
                                                  operations=transform_img)

    # apply batch operations
    generator_dataset = generator_dataset.batch(batch_size, drop_remainder=False)
    
    # apply repeat operations
    #if dataset_type == "train" and epoch_size is not None and epoch_size != 1:
    #    generator_dataset = generator_dataset.repeat(epoch_size)

    return generator_dataset

def create_dataset_DIV2K_test(args, dataset_type="valid", num_parallel_workers=10, shuffle=True):
    # 生成没有batch没有patch的数据集，仅供获得最终图片使用
    num_parallel_workers = 1
    
    dataset_path:str = args.dir_data # 默认"/home/hyacinthe/graduation-dissertation/dataset/DIV2K"
    lr_scale:int = args.scale # 默认3倍放大
    lr_type:str = "bicubic" # 默认用传统缩放的数据集
    epoch_size:int = args.epochs # 训练的趟数
    n_train = args.n_train
    
    # get HR_PATH/*.png
    dir_hr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_HR") # 高分图片，包括train和valid
    hr_pattern = os.path.join(dir_hr, "*.png")

    # get LR_PATH/X2/*x2.png, LR_PATH/X3/*x3.png, LR_PATH/X4/*x4.png # 低分图片，包括train和valid
    column_names = []
    lrs_pattern = []
    dir_lr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_LR_{lr_type}", f"X{lr_scale}")
    lr_pattern = os.path.join(dir_lr, f"*x{lr_scale}.png")
    lrs_pattern.append(lr_pattern)
    column_names.append(f"lrx{lr_scale}")
    column_names.append("hr")  # ["lrx2","lrx3","lrx4",..., "hr"]

    # make dataset
    dataset = LrHrImages(n_train, lr_pattern=lrs_pattern, hr_pattern=hr_pattern)

    # make mindspore dataset
    generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                            num_parallel_workers=num_parallel_workers,
                                            shuffle=shuffle)

    # define map operations
    transform_img = [
        hwc2chw,
        uint8_to_float32,
    ]

    # pre-process hr lr
    generator_dataset = generator_dataset.map(input_columns=column_names,
                                                  output_columns=column_names,
                                                  column_order=column_names,
                                                  operations=transform_img)

    return generator_dataset


