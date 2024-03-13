import os

import numpy as np
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
from mindspore import tensor

from utils.custom_transforms import *
# from custom_transforms import *

class PolypDataset():
    def __init__(self, root, transform_list):
        root = os.path.join('/home/tkk/YZZ/jcad_code', root)
        image_root, gt_root = os.path.join(root, 'images'), os.path.join(root, 'masks')

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)

        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.gts = sorted(self.gts)

        self.filter_files()

        self.size = len(self.images)
        self.transform = self.get_transform(transform_list)

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        tfs = transforms.Compose(tfs)
        return tfs

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        ## AREA #############################################################
        gt_copy = cv2.imread(self.gts[index])
        # convert the image to grayscale
        gray_image = cv2.cvtColor(gt_copy, cv2.COLOR_BGR2GRAY)
        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
        # find contour in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            cnt = 0
        elif len(contours) == 1:
            cnt = 1
        elif len(contours) >= 2:
            cnt = 2
        cnt = tensor(cnt)
        # print(f"gt type : {type(gt)}")
        gt = gt.resize((384, 384))
        #####################################################################
        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape, 'contours': cnt}

        sample = tuple(sample.values())
        # print(f"image = {sample[0]}")
        # print(f"gt = {sample[1]}")
        # print(f"name = {sample[2]}")
        # print(f"shape = {sample[3]}")
        # print(f"contours = {sample[4]}")

        # return sample
        return image, gt, name, shape, cnt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        cnt = 0
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                cnt = cnt + 1

        self.images, self.gts = images, gts
        print(cnt)
        cnt = 0

    def __len__(self):
        return self.size