import os
import argparse
import tqdm
import sys
import random

import cv2

from utils.dataloader import *
from lib.modules.optim import *
from lib import *

import mindspore as ms
from utils.dataloader import *
import mindspore.dataset.vision as vision
from mindspore import ops, nn, value_and_grad

from mindspore.experimental import optim


# python run/Train.py --config configs/UACANet_SwinB-S.yaml --verbose --debug

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

random.seed(42)
np.random.seed(42)
## todo: mindspore的torch.manual_seed(42)
##       和torch.cuda.manual_seed_all(42)

ms.context.set_context(device_target="GPU",device_id=0)
def get_transform_ms():
    resize_op = vision.Resize((384, 384))
    resize_crop_op = vision.RandomResizedCrop(size=(384, 384), scale=(0.75, 1.25))
    lr_flip_op = vision.RandomHorizontalFlip()
    ud_flip_op = vision.RandomVerticalFlip()
    rotate_op = vision.RandomRotation((0, 359))
    # if np.random.random() > 0.5:
    #     factor = float(1 + np.random.random() / 10)
    #     Contrast_op = vision.AdjustContrast(factor)
    #     Brightness_op = vision.AdjustBrightness(factor)
    #     Sharpness_op = vision.AdjustSharpness(factor)
    # else:
    factor = float(1 + np.random.random() / 10)
    Contrast_op = vision.AdjustContrast(factor)
    Brightness_op = vision.AdjustBrightness(factor)
    Sharpness_op = vision.AdjustSharpness(factor)

    normalize_op = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [resize_op, resize_crop_op, lr_flip_op, ud_flip_op, rotate_op
                      , Contrast_op, Brightness_op, Sharpness_op, normalize_op]
    return transform_list


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/tkk/YZZ/configs/UACANet_SwinB-S.yaml')     # Trans
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    # return parser.parse_args(args=[])
    return parser.parse_args()

def train(opt, args):

    epochs = 240
    batch_size = 4
    learning_rate = 1.0e-5

    data_loader = ds.GeneratorDataset(PolypDataset(root=opt.Train.Dataset.root,
                                                   transform_list=opt.Train.Dataset.transform_list),
                                                   column_names=['image', 'gt', 'name', 'shape', 'contours'])

    train_loader = data_loader.map(operations=get_transform_ms(), input_columns=['image'])
    train_loader = train_loader.batch(batch_size=batch_size)

    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                 output_stride=opt.Model.output_stride,
                                 pretrained=opt.Model.pretrained)

    train_dataset = train_loader.create_dict_iterator()
    model.set_train(True)

    backbone_params = list(filter(lambda x: 'backbone' in x.name, model.trainable_params()))
    decoder_params = list(filter(lambda x: 'backbone' not in x.name, model.trainable_params()))
    # group_params = [{'params': backbone_params},
    #                 {'params': decoder_params, 'lr': learning_rate * 10},]
    group_params = [{'params': backbone_params},
                    {'params': decoder_params, 'lr': learning_rate},]
    optimizer = nn.Adam(group_params, learning_rate=learning_rate, weight_decay=0.0, use_lazy=False, use_offload=False)

    def forward(inputs):
        loss = model(inputs)['loss']
        return loss

    def train_step(inputs):
        (loss), grads = grad_fn(inputs)
        optimizer(grads)
        return loss

    grad_fn = value_and_grad(forward, None, optimizer.parameters, has_aux=False)
    step = 0
    for epoch in range(epochs):
        for batch, data in enumerate(train_dataset):
            step = step + 1
            loss = train_step(data)
            print(f"epoch: {epoch}: step: {step} loss:{loss}")


        if (epoch + 1) % 20 == 0:
            print("Saving model...")
            checkpoint_name = "epoch_" + str(epoch) + ".ckpt"
            ms.save_checkpoint(model, checkpoint_name)
            print(f"Successfully saved model {checkpoint_name}")


if __name__ == '__main__':
    args = _args()
    print(args)
    opt = load_config(args.config)
    train(opt, args)