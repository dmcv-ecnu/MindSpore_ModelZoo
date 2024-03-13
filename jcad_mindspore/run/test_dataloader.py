
import os
import sys
filepath = os.path.split(__file__)[0]
print(f"filepath = {filepath}")
repopath = os.path.split(filepath)[0]
print(f"repopath = {repopath}")
sys.path.append(repopath)
from utils.dataloader import *
import mindspore.dataset.vision as vision

def get_transform_ms():
    resize_op = vision.Resize((384, 384))
    resize_crop_op = vision.RandomResizedCrop(size=(384, 384), scale=(0.75, 1.25))
    lr_flip_op = vision.RandomHorizontalFlip()
    ud_flip_op = vision.RandomHorizontalFlip()
    rotate_op = vision.RandomRotation((0, 359))
    if np.random.random() > 0.5:
        factor = float(1 + np.random.random() / 10)
        Contrast_op = vision.AdjustContrast(factor)
        Brightness_op = vision.AdjustBrightness(factor)
        Sharpness_op = vision.AdjustSharpness(factor)
    else:
        Contrast_op = vision.AdjustContrast(1.0)
        Brightness_op = vision.AdjustBrightness(1.0)
        Sharpness_op = vision.AdjustSharpness(1.0)

    normalize_op = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [resize_crop_op, lr_flip_op, ud_flip_op, rotate_op
                      , Contrast_op, Brightness_op, Sharpness_op, normalize_op]
    return transform_list

opt = load_config('/home/tkk/YZZ/configs/UACANet_SwinB-S.yaml')
transform_list = opt.Train.Dataset.transform_list
ops = get_transform_ms()

print(f"ops = {ops}")



data_loader = ds.GeneratorDataset(PolypDataset(root=opt.Train.Dataset.root, transform_list=opt.Train.Dataset.transform_list), column_names=['image', 'gt', 'name', 'shape', 'contours'])
# for data in data_loader.create_dict_iterator(num_epochs=1):
#     print(f"shape = {data['shape']}")
aug_dataset = data_loader.map(operations=ops)

#
# for data in aug_dataset.create_dict_iterator(num_epochs=1):
#     print(data['image'])

print(aug_dataset)