from __future__ import print_function
import argparse
import time
import datetime
import os
import shutil
import sys
import random
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from mindspore.dataset import transforms, vision, GeneratorDataset
from mindspore import  nn, ops
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.score import SegmentationMetric


from train_edge import parse_args

from config import data_root, model_root

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.flip = args.flip


        # image transform
        input_transform = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize([.485, .456, .406], [.229, .224, .225], False),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, root=data_root, split='val', mode='ms_val',
                                               transform=input_transform)
        self.num_class = val_dataset.num_class
        val_sampler = make_data_sampler(False)
        self.val_loader = GeneratorDataset(source=val_dataset, column_names=["output"],
                                           sampler=val_sampler)
        val_batch_size = 1
        self.val_loader = self.val_loader.batch(val_batch_size)
        # create network
        BatchNorm2d = nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone, root=model_root,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            norm_layer=BatchNorm2d)
        # if args.distributed:
        #     self.model = nn.parallel.DistributedDataParallel(self.model,
        #         device_ids=[args.local_rank], output_device=args.local_rank)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, batch_data in enumerate(self.val_loader):
            print(i)
            batch_data = batch_data[0]
            img_resized_list = batch_data['img_data']
            target = ops.expand_dims(batch_data['seg_label'], 0)
            filename = batch_data['info']
            size = (target.shape[1], target.shape[2])

            segSize = size
            scores = ops.zeros((1, self.num_class, segSize[0], segSize[1]))
            for image in img_resized_list:
                image = ops.expand_dims(image[0], 0)
                a, b = self.model(image)
                logits = a
                logits = ops.interpolate(logits, size=size,
                                         mode='bilinear', align_corners=True)
                scores += ops.softmax(logits, axis=1)
                # scores = scores + outimg / 6
                if self.flip:
                    # print('use flip')
                    image = ops.flip(image, dims=(3,))
                    a, b = self.model(image)
                    logits = a
                    logits = ops.flip(logits, dims=(3,))
                    logits = ops.interpolate(logits, size=size,
                                             mode='bilinear', align_corners=True)
                    scores += ops.softmax(logits, axis=1)
            self.metric.update(scores, target)

        pixAcc, IoU, mIoU = self.metric.get()
        logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.6f}".format(i + 1, pixAcc, mIoU))
        IoU = IoU.asnumpy()
        num = IoU.size
        di = dict(zip(range(num), IoU))
        for k, v in di.items():
            logger.info("{}: {}".format(k, v))


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    logger = setup_logger("semantic_segmentation", args.log_dir, 0,
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
