import pickle

from dataset.helper_ply import read_ply
import numpy as np
import numpy.random
from pathlib import *
import ext.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors
from config import cfg
from mindspore import Tensor, dtype
from sklearn.neighbors import KDTree


class S3DIS:
    def __init__(self, dataset_path, area, excl,
                 steps, num_points, noise_init, sub_sample_ratios, labeled_point, num_classes,
                 transform=None, target_transform=None):  # ignored num_classes + 1
        """
        dataset_path/[name.ply]
        """
        self.num_classes = num_classes
        self.labeled_points = labeled_point
        self.dataset_path = Path(dataset_path)
        self.area = area
        self.excl = excl

        self.steps = steps
        self.num_points = num_points
        self.noise_init = noise_init
        self.sub_sample_ratios = sub_sample_ratios

        self.transform = transform
        self.target_transform = target_transform

        self.cloud_names = self.list_cloud_names()

        self.tree = []
        self.colors = []
        self.labels = []
        self.xyzs = []
        self.select_label_idx = []

        self.load_sub_sampled_clouds()
        self.gen = self.select_clouds()  # get generator
        self.gen_r = None
        self.val_proportion = self.compute_label_proportion()

    def compute_label_proportion(self):
        """ return (number of original label) / (number of subsampled label)"""
        return np.array([15378795, 13000896, 22950648, 22424, 1383387, 2764012, 2384459, 2948921,
                         1465826, 212002, 8131641, 933756, 7012581])  # TODO: dynamic compute

    def list_cloud_names(self):
        if self.excl:
            ex = 'Area_{}'.format(self.area)
            fs = self.dataset_path.glob('*.ply')
            return [f.stem for f in fs if ex not in f.stem]
        else:
            return [i.stem for i in self.dataset_path.glob('Area_{}*.ply'.format(self.area))]

    def load_sub_sampled_clouds(self):
        for i, cloud_name in enumerate(self.cloud_names):
            sub_ply_file = self.dataset_path / '{:s}.ply'.format(cloud_name)

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).astype(np.float32).T
            sub_labels = data['class']
            sub_xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T

            # with open(self.dataset_path / '{:s}_KDTree.pkl'.format(cloud_name), 'rb') as f:
            #     search_tree = pickle.load(f)
            search_tree = KDTree(sub_xyz)

            if self.excl:
                self.sample_label(sub_labels)

            self.tree.append(search_tree)
            self.colors.append(sub_colors)
            self.labels.append(sub_labels)
            self.xyzs.append(sub_xyz)
            # TODO: quick load. delete it when training
            if i == 3: break

    def sample_label(self, sub_labels):
        all_select_label_indx = []
        for i in range(self.num_classes + 1):
            ind_class = np.where(sub_labels == i)[0]
            num_classs = len(ind_class)
            if num_classs > 0:
                num_selected = max(int(num_classs * self.labeled_points), 1)
                # TODO: .pt limit.
                # else:
                #     num_selected = int(labeled_point)

                label_indx = list(range(num_classs))
                np.random.shuffle(label_indx)

                ind_class_select = ind_class[label_indx[:num_selected]]
                ind_class_noselect = ind_class[label_indx[num_selected:]]
                all_select_label_indx.append(ind_class_select[0])
                sub_labels[ind_class_noselect] = self.num_classes
        self.select_label_idx.append(all_select_label_indx)

    # select points by possibility
    def select_clouds(self):
        possibility = []
        min_possibility = []
        for points in self.xyzs:
            possibility += [np.random.rand(points.shape[0]) / (10 ** 3)]  # every points a possibility
            min_possibility += [float(np.min(possibility[-1]))]  # new added poss, get value.

        def generator():
            for _ in range(self.steps):
                cloud_idx = int(np.argmin(min_possibility))
                center_point_idx = np.argmin(possibility[cloud_idx])

                points = self.xyzs[cloud_idx]
                center_point = points[center_point_idx, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=self.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) > self.num_points:
                    queried_idx = self.tree[cloud_idx].query(pick_point, k=self.num_points)[1][0]
                else:
                    queried_idx = self.tree[cloud_idx].query(pick_point, k=len(points))[1][0]

                if self.excl:
                    s_indx = self.select_label_idx[cloud_idx]  # training only
                    # Shuffle index
                    queried_idx = np.concatenate([np.array(s_indx), queried_idx], 0)[:self.num_points]  # training only

                # Shuffle index
                np.random.shuffle(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.colors[cloud_idx][queried_idx]
                queried_pc_labels = self.labels[cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square(queried_pc_xyz), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                possibility[cloud_idx][queried_idx] += delta
                min_possibility[cloud_idx] = float(np.min(possibility[cloud_idx]))

                # up_sampled with replacement
                if len(points) < self.num_points:
                    rp_idx = self.random_repeat(queried_idx, self.num_points)
                    queried_pc_xyz = queried_pc_xyz[rp_idx]
                    queried_pc_colors = queried_pc_colors[rp_idx]
                    queried_pc_labels = queried_pc_labels[rp_idx]
                    queried_idx = queried_idx[rp_idx]

                if self.target_transform is not None:
                    queried_pc_labels = self.target_transform(queried_pc_labels)
                if self.transform is not None:
                    queried_pc_xyz, queried_pc_colors = self.transform(queried_pc_xyz, queried_pc_colors)

                xyz = np.transpose(queried_pc_xyz, [1, 0])
                color = np.transpose(queried_pc_colors, [1, 0])

                yield xyz, color, queried_pc_labels, queried_idx, cloud_idx

        return generator

    def random_repeat(self, xyz, num_out):
        ni = len(xyz)
        dup = np.random.choice(ni, num_out - ni)
        t = np.concatenate([np.arange(ni), dup])
        return t

    def __iter__(self):
        self.gen_r = self.gen()
        return self.gen_r

    def __next__(self):
        return next(self.gen_r)


def knn(q, s, k):
    return nearest_neighbors.knn_batch(s, q, k, omp=True).astype(np.int32)


def batch_map(xyz, k):
    xyz = xyz.asnumpy().transpose([0, 2, 1])

    batch = xyz.shape[0]
    num_layers = cfg.num_layers
    num_points = cfg.num_points
    sub_sample_ratio = cfg.sub_sampling_ratio
    s = []  # sub_idx
    n = []  # neighbor_idx.
    tx = xyz
    sx = [] # sub xyzs.
    for i in range(num_layers):
        n.append(knn(tx, tx, k)) #neighbor
        num_points //= sub_sample_ratio[i]
        t = np.arange(num_points, dtype=np.int32)[None, :].repeat(batch, 0)
        s.append(t)
        tx = batch_gather(tx, t) #subsample
        sx.append(tx)

    u = []  # up_idx
    for i in range(num_layers):
        if i != num_layers - 1:
            t = knn(sx[-2 - i], sx[-1 - i], 1)
        else:
            t = knn(xyz, sx[0], 1)
        u.append(t)

    def tensorfy(l):
        return [*map(lambda x: Tensor(x, dtype=dtype.int32), l)]
    return tensorfy(s), tensorfy(u), tensorfy(n)



def batch_gather(v, i):
    d = v.shape[2]
    i = i[..., None]
    i = np.repeat(i, d, 2)
    return np.take_along_axis(v, i, 1)