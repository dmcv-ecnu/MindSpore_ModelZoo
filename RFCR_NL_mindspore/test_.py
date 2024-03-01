import pickle
import time
import numpy as np
from os.path import join
import mindspore.dataset as ds

from src.utils.tools import ConfigS3DIS as cfg
from src.utils.tools import DataProcessing as DP
from src.utils.helper_ply import read_ply
sub_ply_file = '/media/data1/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Stanford3dDataset_v1.2_Aligned_Version/input_0.040/Area_2_hallway_9.ply' #25486

data = read_ply(sub_ply_file)
gt_label_file = 'runs/rfcr-area5_weak_pretrain_new/gt_1/Area_2_hallway_9.npy'
gt_labels = np.squeeze(np.load(gt_label_file))
print(gt_labels)
print((gt_labels==data['class']).sum())