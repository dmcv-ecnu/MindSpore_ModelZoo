"""Prepare Cityscapes dataset"""
import os, mindspore
from mindspore import Tensor
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset
# from . import cityscapes_labels
from . import edge_utils

num_classes = 19


class NightEdgeSegmentation(SegmentationDataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, root='../datasets/night', split='train', mode=None, transform=None, **kwargs):
        super(NightEdgeSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        # assert os.path.exists(self.root), "Please setup the dataset using ../datasets/cityscapes.py"
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split, self.mode)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]

        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])

        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def re_class_to_index(self, mask):
        temp = mask.ravel()
        for i in range(0, len(temp)):
            if temp[i] == 255:
                temp[i] = 0

        values = np.unique(temp)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(temp, self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform_2(img, mask)
            _edgemap = mask.numpy()
            _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
            edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
        #             edgemap = Tensor.from_numpy(_edgemap).float()

        elif self.mode == 'val':
            img, mask = self._img_transform(img), self._mask_transform(mask)
            _edgemap = mask.numpy()
            _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
            _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
            edgemap = Tensor.from_numpy(_edgemap).float()

        elif self.mode == 'ms_val':
            output = self._val_ms_transform(img, mask)
            mask = self._mask_transform(mask)
            output['seg_label'] = mask
            output['info'] = os.path.basename(self.images[index])
            return output

        else:
            assert self.mode == 'testval'
            mask = self._test_val_mask(mask)
            img, mask = self._img_transform(img), self._mask_transform(mask)
            _edgemap = mask.numpy()
            _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
            _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
            edgemap = Tensor.from_numpy(_edgemap).float()
        # general resize, normalize and toTensor

        if self.transform is not None:
            img = self.transform(img)
        return img, mask, edgemap, os.path.basename(self.images[index])

    # def re_mask_transform(self, mask):
    #     target = self.re_class_to_index(np.array(mask).astype('int32'))
    #     return torch.LongTensor(np.array(target).astype('int32'))
    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return Tensor(np.array(target).astype('int32'), dtype=mindspore.int64)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train', mode='ms_val'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = os.path.join(os.path.basename(filename)[:-4] + "_labelIds.png")
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'images/' + split)
        mask_folder = os.path.join(folder, 'label/')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'images/train')
        train_mask_folder = os.path.join(folder, 'label/')
        val_img_folder = os.path.join(folder, 'images/val')
        val_mask_folder = os.path.join(folder, 'label/')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = NightSegmentation()
