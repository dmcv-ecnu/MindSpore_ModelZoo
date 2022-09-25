import sys
import numpy as np
import os.path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from utils.cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling
from utils.nearest_neighbors.lib.python import nearest_neighbors as nearest_neighbors


# import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
# import utils.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigPretrain:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 6  # a, b, \mu_{a},\mu_{a}, \sigmoid_{a}, \sigmoid_{b}
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    train_steps = 1000  # Number of steps per epochs

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # 2.0 noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    lr_decays = 0.95  # decay rate of learning rate
    loss_scale = 1.0  # loss scale


class ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 80  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    # lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    lr_decays = 0.95  # decay rate of learning rate
    loss_scale = 1.0  # loss scale

    training_ep0 = {i: 0 for i in range(0, 30)}  #
    training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
    training_ep.update(training_ep0)
    # training_ep = {i: 0 for i in range(max_epoch)}
    # training_ep[2] = 1
    c_epoch = 0
    train_sum_dir = 'train_log'
    saving = True
    saving_path = None
    # pretrain = False
    # checkpoint = '../pretrain/snapshots/snap-11001'

    pretrain = True
    checkpoint = './pretrain/snapshots/snap-11001'
    topk = 500
    loss3_type = -1


class DataProcessing:
    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        elif dataset_name is 'ScanNet':
            num_per_class = np.array(
                [13327131, 11989728, 1909991, 1291399, 3122500, 1197818, 1818311, 1942293, 1332850, 1000934,
                 227936, 244557, 892883, 800339, 246651, 125533, 134002, 110112, 162706, 1575880], dtype=np.int32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU
