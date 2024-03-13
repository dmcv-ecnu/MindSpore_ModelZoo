# import torch
import os
import argparse
import tqdm
import sys
import cv2

# import torch.nn.functional as F
import numpy as np
import mindspore as ms
from PIL import Image
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.ops as ops
from mindspore import Tensor

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.utils import *
from utils.dataloader import *
from utils.custom_transforms import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/tkk/YZZ/configs/UACANet_SwinB-S.yaml')
    parser.add_argument('--checkpoint_file', type=str, default='/home/tkk/YZZ/run/epoch_239.ckpt')
    parser.add_argument('--source', type=str, default='/home/tkk/YZZ/jcad_code/data/TestDataset/test/images')
    parser.add_argument('--type', type=str,
                        choices=['rgba', 'map'], default='map')
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser.parse_args()


def get_transform_ms():
    resize_op = vision.Resize((384, 384))
    normalize_op = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [resize_op, normalize_op]
    return transform_list

def inference(opt, args):
    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                 pretrained=opt.Model.pretrained)
    # model.load_state_dict(torch.load(os.path.join(
    #     opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)


    param_dict = ms.load_checkpoint(args.checkpoint_file)
    param_not_load, _ = ms.load_param_into_net(model, param_dict)
    print(f"Params not loaded: {param_not_load}")
    # model.cuda()
    # model.eval()
    model.set_train(False)

    # transform = eval(opt.Test.Dataset.type).get_transform(
    #     opt.Test.Dataset.transform_list)
    transform = get_transform_ms()
    # if os.path.isdir(args.source):


    if os.path.isdir(args.source):
        source_dir = args.source
        source_list = os.listdir(args.source)

        save_dir = os.path.join('results', args.source.split(os.sep)[-1])

    elif os.path.isfile(args.source):
        source_dir = os.path.split(args.source)[0]
        source_list = [os.path.split(args.source)[1]]

        save_dir = 'results'
    else:
        return

    os.makedirs(save_dir, exist_ok=True)

    if args.verbose is True:

        sources = tqdm.tqdm(enumerate(source_list), desc='Inference', total=len(
            source_list), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')

    else:
        sources = enumerate(source_list)

    # print(f"Predicted images saved in {os.path.join(save_dir, os.path.splitext(args.source)[0])}")
    for i, source in sources:
        img = Image.open(os.path.join(source_dir, source)).convert('RGB')
        sample = {'image': img}

        # normalize = vision.Normalize(mean=[0, 0, 0], std=[1, 1, 1], is_hwc=False)
        to_tensor = vision.ToTensor()
        resize_op = vision.Resize((224, 224))
        normalize_op = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        sample = transforms.Compose([resize_op, normalize_op])(sample['image'])

        img_s = sample[0]
        img_s = Tensor(img_s)
        sample = {'image': img_s}

        sample['image'] = sample['image'].unsqueeze(0)
        # sample = to_cuda(sample)

        out = model(sample)
        out['pred'] = ops.interpolate(
            out['pred'], img.size[::-1], mode='bilinear', align_corners=True)
        # out['pred'] = out['pred'].data.cpu()
        out['pred'] = ops.sigmoid(out['pred'])
        out['pred'] = out['pred'].numpy().squeeze()
        out['pred'] = (out['pred'] - out['pred'].min()) / \
            (out['pred'].max() - out['pred'].min() + 1e-8)
        out['pred'] = (out['pred'] * 255).astype(np.uint8)

        if args.type == 'map':
            img = out['pred']
        elif args.type == 'rgba':
            img = np.array(img)
            r, g, b = cv2.split(img)
            img = cv2.merge([r, g, b, out['pred']])
        Image.fromarray(img).save(os.path.join(
            "/home/tkk/YZZ/pred_output", os.path.splitext(source)[0] + '.png'))

if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    inference(opt, args)