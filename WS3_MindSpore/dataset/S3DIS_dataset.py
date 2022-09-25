import glob
import pickle, time, warnings
import numpy as np
from pathlib import Path

import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore import dtype as mstype
import mindspore.ops as ops

from utils.tools import DataProcessing as DP
from utils.helper_ply import read_ply
import random


class S3DISDatasetGenerator:
    def __init__(self, labeled_point_percent=1):

        self.dataset_path = Path("./DataSets/S3DIS")
        # self.dataset_path = Path("./datasets/S3DIS")
        print(self.dataset_path.absolute())
        self.label_to_names = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window', 6: 'door',
                               7: 'table', 8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter',
                               13: 'unlabel'
                               }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([13])  # 13

        self.val_split = 'Area_5'
        self.original_ply_paths = self.dataset_path / 'original_ply'
        self.paths = list(self.original_ply_paths.glob('*.ply'))
        self.size = len(self.paths)
        #
        self.labeled_point = f'{labeled_point_percent}%'
        # self.labeled_point = '1%'
        self.sampling_mode = 0  # 0 random, 1:spt
        self.sub_grid_size = 0.04
        self.weak_label = True
        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.s_indx = {'training': [], 'validation': []}

        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        tree_path = self.dataset_path / f"input_{self.sub_grid_size:.3f}"
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = tree_path / f'{cloud_name}_KDTree.pkl'
            sub_ply_file = tree_path / f'{cloud_name}.ply'

            data = read_ply(str(sub_ply_file))
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T
            all_select_label_indx = []
            if cloud_split == 'training' and self.weak_label:
                ''' ***************** '''
                all_select_label_indx = []
                for i in range(self.num_classes):
                    ind_class = np.where(sub_labels == i)[0]
                    num_classs = len(ind_class)
                    if num_classs > 0:
                        if '%' in self.labeled_point:
                            r = float(self.labeled_point[:-1]) / 100
                            num_selected = max(int(num_classs * r), 1)
                        else:
                            num_selected = int(self.labeled_point)

                        if self.sampling_mode == 1:
                            label_indx = list(range(num_classs))
                            random.shuffle(label_indx)
                            select_labels_indx = label_indx[:num_selected]
                            ind_class_select = ind_class[select_labels_indx]
                            anchor_xyz = sub_xyz[ind_class_select].reshape([1, -1, 3])
                            class_xyz = sub_xyz[ind_class].reshape([1, -1, 3])
                            cluster_idx = DP.knn_search(class_xyz, anchor_xyz, 50).squeeze()  # knn_search （B,N,k）
                            ind_class_noselect = np.delete(label_indx, cluster_idx)
                            ind_class_noselect = ind_class[ind_class_noselect]
                            sub_labels[ind_class_noselect] = 13
                            all_select_label_indx.append(cluster_idx[0])
                        elif self.sampling_mode == 0:
                            label_indx = list(range(num_classs))
                            random.shuffle(label_indx)
                            noselect_labels_indx = label_indx[num_selected:]
                            select_labels_indx = label_indx[:num_selected]
                            ind_class_noselect = ind_class[noselect_labels_indx]
                            ind_class_select = ind_class[select_labels_indx]
                            all_select_label_indx.append(ind_class_select[0])
                            sub_labels[ind_class_noselect] = 13
                ''' ***************** '''

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]
            if cloud_split == 'training' and self.weak_label:
                self.s_indx[cloud_split] += [all_select_label_indx]  # training only]:

            print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = tree_path / f"{cloud_name}_proj.pkl"
                # proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))
        print('Complete data loading')

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        print(f"Length of S3DISDatasetGenerator is {self.size}")
        return self.size


class ActiveLearningSampler(ds.Sampler):

    def __init__(self, dataset, cfg, batch_size=6, split='training'):
        super(ActiveLearningSampler, self).__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.n_samples = self.cfg.train_steps
        else:
            self.n_samples = self.cfg.val_steps

        # Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        len = self.n_samples * self.batch_size
        print(f"Length of ActiveLearningSampler is {len}")
        return len  # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.
        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # Generator loop
            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])

            # Get points from tree structure
            points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            # Check if the number of points in the selected cloud is less than the predefined num_points
            if len(points) < self.cfg.num_points:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
            else:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=self.cfg.num_points)[1][0]

            if self.split == 'training':
                s_indx = self.dataset.s_indx[self.split][cloud_idx]  # training only
                # Shuffle index
                queried_idx = np.concatenate([np.array(s_indx), queried_idx], 0)[:self.cfg.num_points]  # training only

            # Shuffle index
            queried_idx = DP.shuffle_idx(queried_idx)
            # Collect points and colors
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
            queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            # Update the possibility of the selected points
            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[self.split][cloud_idx][queried_idx] += delta
            self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

            # up_sampled with replacement
            if len(points) < self.cfg.num_points:
                # 虽然叫data_aug, 但与印象中的数据增强相差甚远
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                    DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, self.cfg.num_points)

            queried_pc_xyz = queried_pc_xyz.astype(np.float32)
            queried_pc_colors = queried_pc_colors.astype(np.float32)
            queried_pc_labels = queried_pc_labels.astype(np.int32)
            queried_idx = queried_idx.astype(np.int32)
            cloud_idx = np.array([cloud_idx], dtype=np.int32)
            # print(queried_pc_xyz.shape, queried_pc_colors.shape)
            # queried_pc_xyz, queried_pc_colors = data_augment(queried_pc_xyz, queried_pc_colors)

            yield queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cloud_idx


def ms_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batchInfo):
    """
    xyz =>  [B,N,3]
    features =>  [B,N,d]
    labels =>  [B,N,]
    pc_idx =>  [B,N,]
    cloud_idx =>  [B,]
    """
    from .tools import ConfigS3DIS as cfg

    batch_xyz = np.array(batch_xyz, dtype=np.float32)
    batch_features = np.array(batch_features, dtype=np.float32)
    # batch_features = data_augment(batch_xyz, batch_features)
    # batch_xyz = batch_features[:, :3]
    batch_features = np.concatenate((batch_xyz, batch_features), axis=-1)
    input_points = []  # [num_layers, B, N, 3]
    input_neighbors = []  # [num_layers, B, N, 16]
    input_pools = []  # [num_layers, B, N, 16]
    input_up_samples = []  # [num_layers, B, N, 1]

    for i in range(cfg.num_layers):
        neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n).astype(np.int32)
        sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
        up_i = DP.knn_search(sub_points, batch_xyz, 1).astype(np.int32)
        input_points.append(batch_xyz)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_xyz = sub_points
    # b_f:[B, N, 3+d]
    # due to the constraints of the mapping function, only the list elements can be passed back sequentially
    # batch_features.astype(np.float32)
    # batch_features.dtype(np.float32)
    # batch_labels.astype(np.float32)

    return batch_features, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, input_points[0], input_points[1], input_points[2], input_points[3], \
           input_points[4], \
           input_neighbors[0], input_neighbors[1], input_neighbors[2], input_neighbors[3], input_neighbors[4], input_pools[0], input_pools[1], input_pools[2], \
           input_pools[3], input_pools[4], input_up_samples[0], input_up_samples[1], input_up_samples[2], input_up_samples[3], input_up_samples[4]


def data_augment(xyz, feats):
    theta = np.random.uniform(size=(1,), low=0, high=2 * np.pi)
    # Rotation matrices
    c, s = np.cos(theta), np.sin(theta)
    cs0 = np.zeros_like(c)
    cs1 = np.ones_like(c)
    R = np.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
    stacked_rots = np.reshape(R, (3, 3))
    # Apply rotations
    transformed_xyz = np.reshape(np.matmul(xyz, stacked_rots), [-1, 3])
    # Choose random scales for each example
    min_s = 0.8
    max_s = 1.2
    # s = np.random_uniform((1, 3), minval=min_s, maxval=max_s)
    s = np.random.uniform(size=(1, 3), low=min_s, high=max_s)

    symmetries = []
    augment_symmetries = [True, False, False]
    for i in range(3):
        if augment_symmetries[i]:
            symmetries.append(np.round(np.random.uniform(size=(1, 1))) * 2 - 1)
            # symmetries.append(np.round(np.random_uniform((1, 1))) * 2 - 1)
        else:
            symmetries.append(np.ones([1, 1], dtype=np.float32))
    s *= np.concatenate(symmetries, 1)
    # np.concatenate
    # Create N x 3 vector of scales to multiply with stacked_points
    stacked_scales = np.tile(s, [np.shape(transformed_xyz)[0], 1])

    # Apply scales
    transformed_xyz = transformed_xyz * stacked_scales

    noise = np.random.normal(size=np.shape(transformed_xyz), scale=0.001)
    transformed_xyz = transformed_xyz + noise
    rgb = feats[:, :3]
    # print(transformed_xyz.shape, rgb.shape, feats.shape)
    # stacked_features = np.concatenate([transformed_xyz, rgb], axis=-1)
    return transformed_xyz, rgb


def dataloader(cfg, **kwargs):
    dataset = S3DISDatasetGenerator(cfg.labeled_percent)
    val_sampler = ActiveLearningSampler(
        dataset,
        cfg,
        batch_size=cfg.val_batch_size,
        split='validation'
    )
    train_sampler = ActiveLearningSampler(
        dataset,
        cfg,
        batch_size=cfg.batch_size,
        split='training'
    )
    input_columns = ["xyz", "colors", "labels", "q_idx", "c_idx"]
    return ds.GeneratorDataset(train_sampler, input_columns, **kwargs), ds.GeneratorDataset(val_sampler, input_columns, **kwargs), dataset
