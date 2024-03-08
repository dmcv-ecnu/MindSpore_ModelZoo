# -*-coding:utf-8-*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pickle
import random
import time
import argparse
import os
import glob
import numpy as np
import mindspore.dataset as ds
from os.path import join, exists
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from os import makedirs
from RandLANet_S3DIS_pretrain_mindspore import RandLANet
from tester_S3DIS_mindspore import ModelTester


class S3DISDatasetGenerator:
    def __init__(
        self,
        data_type="ply",
        val_area="Area_5",
    ):
        self.path = "/home/ubuntu/hdd1/yhj/RPSC/Dataset/S3DIS"
        self.all_files = glob.glob(join(self.path, "original_ply", "*.ply"))
        self.data_type = data_type

        self.label_to_names = {
            0: "ceiling",
            1: "floor",
            2: "wall",
            3: "beam",
            4: "column",
            5: "window",
            6: "door",
            7: "table",
            8: "chair",
            9: "sofa",
            10: "bookcase",
            11: "board",
            12: "clutter",
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        cfg.ignored_label_inds = [
            self.label_to_idx[ign_label] for ign_label in self.ignored_labels
        ]

        # Initiate containers
        self.input_trees = {"training": [], "validation": []}
        self.input_colors = {"training": [], "validation": []}
        self.input_labels = {"training": [], "validation": []}
        self.input_names = {"training": [], "validation": []}
        self.s_indx = {"training": [], "validation": []}
        self.val_proj = []
        self.val_labels = []
        self.val_area = val_area

        self.load_data(cfg.sub_grid_size)
        print("Size of training : ", len(self.input_colors["training"]))
        print("Size of validation : ", len(self.input_colors["validation"]))

    def load_data(self, sub_grid_size):
        tree_path = join(self.path, "input_{:.3f}".format(sub_grid_size))

        for _, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split("/")[-1][:-4]
            if self.val_area in cloud_name:
                cloud_split = "validation"
            else:
                cloud_split = "training"

            # Name of the input files
            kd_tree_file = join(tree_path, "{:s}_KDTree.pkl".format(cloud_name))
            sub_ply_file = join(tree_path, "{:s}.ply".format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data["red"], data["green"], data["blue"])).T
            sub_labels = data["class"]

            with open(kd_tree_file, "rb") as f:
                search_tree = pickle.load(f)
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 10
            print(
                "{:s} {:.1f} MB loaded in {:.1f}s".format(
                    kd_tree_file.split("/")[-1], size * 1e-6, time.time() - t0
                )
            )
        print("\nPreparing reprojected indices for testing")

        # Get validation and test reprojected indices
        for _, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split("/")[-1][:-4]

            # Validation projection and labels
            if self.val_area in cloud_name:
                proj_file = join(tree_path, "{:s}_proj.pkl".format(cloud_name))
                with open(proj_file, "rb") as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print("{:s} done in {:.1f}s".format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class ActiveLearningSampler(ds.Sampler):

    def __init__(self, dataset, batch_size=6, split="training"):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == "training":
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        # Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for _, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        # not equal to the actual size of the dataset, but enable nice progress bars
        return self.n_samples * self.batch_size

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for _ in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()

            # Generator loop
            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])

            # Get points from tree structure
            points = np.array(
                self.dataset.input_trees[self.split][cloud_idx].data, copy=False
            )

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            if len(points) < cfg.num_points:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(
                    pick_point, k=len(points)
                )[1][0]
            else:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(
                    pick_point, k=cfg.num_points
                )[1][0]

            queried_idx = DP.shuffle_idx(queried_idx)
            # Collect points and colors
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][
                queried_idx
            ]
            queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][
                queried_idx
            ]

            dists = np.sum(
                np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1
            )
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[self.split][cloud_idx][queried_idx] += delta
            self.min_possibility[self.split][cloud_idx] = float(
                np.min(self.possibility[self.split][cloud_idx])
            )

            if len(points) < cfg.num_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = (
                    DP.data_aug(
                        queried_pc_xyz,
                        queried_pc_colors,
                        queried_pc_labels,
                        queried_idx,
                        cfg.num_points,
                    )
                )

            queried_pc_xyz = queried_pc_xyz.astype(np.float32)
            queried_pc_colors = queried_pc_colors.astype(np.float32)
            queried_pc_labels = queried_pc_labels.astype(np.int32)
            queried_idx = queried_idx.astype(np.int32)
            cloud_idx = np.array([cloud_idx], dtype=np.int32)

            yield queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cloud_idx


def ms_map(
    batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batchInfo
):
    """
    xyz =>  [B,N,3]
    features =>  [B,N,d]
    labels =>  [B,N,]
    pc_idx =>  [B,N,]
    cloud_idx =>  [B,]
    """
    batch_xyz = np.array(batch_xyz)
    batch_features = np.array(batch_features)
    batch_features = np.concatenate((batch_xyz, batch_features), axis=-1)
    input_points = []  # [num_layers, B, N, 3]
    input_neighbors = []  # [num_layers, B, N, 16]
    input_pools = []  # [num_layers, B, N, 16]
    input_up_samples = []  # [num_layers, B, N, 1]

    for i in range(cfg.num_layers):
        neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n).astype(np.int32)
        sub_points = batch_xyz[:, : batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, : batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
        up_i = DP.knn_search(sub_points, batch_xyz, 1).astype(np.int32)
        input_points.append(batch_xyz)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_xyz = sub_points

    # b_f:[B, N, 3+d]
    # due to the constraints of the mapping function, only the list elements can be passed back sequentially
    return (
        batch_features,
        batch_labels,
        batch_pc_idx,
        batch_cloud_idx,
        input_points[0],
        input_points[1],
        input_points[2],
        input_points[3],
        input_points[4],
        input_neighbors[0],
        input_neighbors[1],
        input_neighbors[2],
        input_neighbors[3],
        input_neighbors[4],
        input_pools[0],
        input_pools[1],
        input_pools[2],
        input_pools[3],
        input_pools[4],
        input_up_samples[0],
        input_up_samples[1],
        input_up_samples[2],
        input_up_samples[3],
        input_up_samples[4],
    )


def dataloader(dataset, **kwargs):
    val_sampler = ActiveLearningSampler(
        dataset, batch_size=cfg.val_batch_size, split="validation"
    )
    return ds.GeneratorDataset(
        val_sampler,
        column_names=["xyz", "colors", "labels", "q_idx", "c_idx"],
        **kwargs
    )


def get_log_dir(args):
    import datetime, sys
    from pathlib import Path

    experiment_dir = Path("./experiment_mindspore/")
    experiment_dir = experiment_dir.joinpath("S3DIS")
    if "%" in args.labeled_point:
        n = args.labeled_point[:-1] + "_percent_"
    else:
        n = args.labeled_point + "_points_"

    experiment_dir = experiment_dir.joinpath(n)  # model_name
    if args.log_dir is None:
        logs = np.sort(
            [
                os.path.join(experiment_dir, f)
                for f in os.listdir(experiment_dir)
                if f.startswith("20")
            ]
        )
        experiment_dir = Path(logs[-1])
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    tensorboard_log_dir = experiment_dir.joinpath("tensorboard/")
    return str(experiment_dir), str(checkpoints_dir), str(tensorboard_log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="the number of GPUs to use [default: 0]"
    )
    parser.add_argument(
        "--mode", type=str, default="test", help="options: train, test, vis"
    )
    parser.add_argument(
        "--test_area",
        type=int,
        default=5,
        help="Which area to use for test, option: 1-6 [default: 5]",
    )
    parser.add_argument("--labeled_point", type=str, default="1", help="1, 10 or 100")
    parser.add_argument(
        "--model_name", type=str, default="RandLANet_S3DIS_pretrain.py", help=""
    )
    parser.add_argument("--log_dir", type=str, default="ex", help="")
    parser.add_argument("--knn", type=int, default=16, help="k_nn")
    parser.add_argument("--total_log", type=str, default="1pt", help="")
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    Mode = FLAGS.mode
    
    test_area = FLAGS.test_area
    dataset = S3DISDatasetGenerator()
    test = dataloader(dataset)
    test = test.batch(
        batch_size=cfg.val_batch_size,
        per_batch_map=ms_map,
        input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
        output_columns=[
            "features",
            "labels",
            "input_inds",
            "cloud_inds",
            "p0",
            "p1",
            "p2",
            "p3",
            "p4",
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "pl0",
            "pl1",
            "pl2",
            "pl3",
            "pl4",
            "u0",
            "u1",
            "u2",
            "u3",
            "u4",
        ],
        drop_remainder=True,
    )

    (
        cfg.experiment_dir,
        cfg.checkpoints_dir,
        cfg.tensorboard_log_dir,
    ) = get_log_dir(FLAGS)
    cfg.total_log_dir = FLAGS.total_log
    cfg.log_dir = FLAGS.log_dir

    d_in = 6
    num_classes = 13
    model = RandLANet(d_in, num_classes, bias=True, config=cfg)
    snap_path = cfg.checkpoints_dir
    print(snap_path)
    snap_steps = [int(f[11:-5]) for f in os.listdir(snap_path)]
    chosen_step = np.sort(snap_steps)[-1]
    chosen_snap = os.path.join(snap_path, "best_epoch_{:d}.ckpt".format(chosen_step))

    tester = ModelTester(dataset, cfg=cfg)
    tester.test(model, dataset, test, restore_snap=chosen_snap)
