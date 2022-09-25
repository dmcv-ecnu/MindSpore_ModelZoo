import os
import mindspore #import torch
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cv2
import random
import moxing as mox
import re
from option import args

def make_dataset():
    hazyImages = []
    clearImages = []
    
    #dataset ="C:/Users/83893/PycharmProjects/mindspore/dataset/ITS_v2"
    #dataset = "/hhlv1/hhlKDDN/dataset/ITS_v2"
    #print(os.file.exist(dataset))
    
    dataset = "/cache/dataset/"
    os.makedirs(dataset, exist_ok=True)
    dataroot = args.data_url
    mox.file.copy_parallel(dataroot, dataset)
    
    for i in range(1,1399):
        clearImages.append(dataset+"clear/"+str(i)+".png")
        #hazyImages.append(dataset+"trains/"+str(i)+"_1.png")
        fl = os.listdir(dataset+"hazy")
        for item in fl:
            if re.match(str(i)+"_1_",item):
                hazyImages.append(dataset+"hazy/"+item)
                break
        

    indices = np.arange(len(clearImages))#np.arange(99)#
    np.random.shuffle(indices)
    clearShuffle = []
    hazyShuffle = []

    for i in range(len(indices)):
        index = indices[i]
        clearShuffle.append(clearImages[index])
        hazyShuffle.append(hazyImages[index])

    return clearShuffle, hazyShuffle



def gammaA(image, gamma_value):
    '''
    lum = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114
    avgLum = np.mean(lum)
    gamma_value = 2*(0.5+avgLum)
    '''
    gammaI = (image + 1e-10) ** gamma_value
    #print(gamma_value)
    return gammaI


def random_rot(images):
    randint = random.randint(0, 4)
    if randint == 0:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_90_CLOCKWISE)
    elif randint == 1:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_180)
    elif randint == 2:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        pass
    return images


def random_crop(images, sizeTo=256):
    w = images[0].shape[1]
    h = images[0].shape[0]
    w_offset = random.randint(0, max(0, w - sizeTo - 1))
    h_offset = random.randint(0, max(0, h - sizeTo - 1))

    for i in range(len(images)):
        images[i] = images[i][h_offset:h_offset + sizeTo, w_offset:w_offset + sizeTo, :]
    return images


def random_flip(images):
    if random.random() < 0.5:
        for i in range(len(images)):
            images[i] = cv2.flip(images[i], 1)
    if random.random() < 0.5:
        for i in range(len(images)):
            images[i] = cv2.flip(images[i], 0)
    return images
def image_resize(images, siezeTo=(256,256)):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], siezeTo)
    return images

def normImge(image, num=1.):
    if len(image.shape) > 2:
        for i in range(3):
            img = image[:,:,i]
            max = np.max(img)
            min = np.min(img)
            image[:, :, i] = (img - min)/(max - min + 1e-8)
    else:
        max = np.max(image)
        min = np.min(image)
        image = (image - min) / (max - min + 1e-8) * num
    return image


class dehazeDataloader:#原文这里transform是True
    def __init__(self, args,train=True, transform=False, num_parallel_workers = 8,sample = None):
        #super.__init__(source=generator_multidimensional, column_names=["multi_dimensional_data"],num_parallel_workers = num_parallel_workers,sample = sample )
        clearImages, hazyImages = make_dataset()
        self.images = hazyImages
        self.clearImages = clearImages
        self._transform = transform

    def __getitem__(self, index):
        Ix = Image.open(self.images[index]).convert('RGB')
        Ix = np.array(Ix, dtype=np.float64) / 255.

        Jx = Image.open(self.clearImages[index]).convert('RGB')
        Jx = np.array(Jx, dtype=np.float64) / 255.

        images = [Ix, Jx]

        images = random_crop(images, 256)
        # images = image_resize(images, (256, 256))

        images = random_rot(images)
        images = random_flip(images)

        [Ix, Jx] = images

        if self._transform is not None:
            Ix, Jx = self.transform(Ix, Jx)

        return Ix, Jx

    def __len__(self):
        return len(self.images)
#这里的transform还没解决
    def transform(self, Ix, Jx):
        #plt.imshow(img, cmap=plt.cm.gray), plt.show()
        Ix = Ix.transpose([2, 0, 1])#.float()
        #Ix = mindspore.tensor.from_numpy(Ix).float()

        Jx = Jx.transpose([2, 0, 1])#.float()
        #Jx = mindspore.tensor.from_numpy(Jx).float()
        return Ix, Jx


if __name__ =="__main__":

    trainLoader = dehazeDataloader(train=True, transform=True)

    for index, (Ix, Jx) in enumerate(trainLoader):

        #print(Ix.shape)
        #(3,256,256)
        Ix = Ix.transpose([1, 2, 0])
        Jx = Jx.transpose([1, 2, 0])
        print(Ix.shape)#(256,256,3)
        print(Jx.shape)
        plt.subplot(221), plt.imshow(Ix)
        plt.subplot(222), plt.imshow(Jx)
        plt.show()