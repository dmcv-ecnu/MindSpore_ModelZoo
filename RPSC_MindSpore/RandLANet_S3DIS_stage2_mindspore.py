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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn.loss.loss import LossBase
from helper_tool import ConfigS3DIS as cfg
from mindspore.ops import composite as C
from mindspore.ops import operations as op
from main_S3DIS_stage2_mindspore import S3DISDatasetGenerator, dataloader, ms_map
import argparse
import os
import numpy as np
from mindspore.ops import functional as F
from mindspore import dtype as mstype
from mindspore.nn.loss import MSELoss
from mindspore import Tensor
from mindspore import (
    Model,
    Tensor,
    context,
    load_checkpoint,
    load_param_into_net,
    nn,
    ops,
)
import mindspore.numpy as mnp


class SharedMLP(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        pad_mode="valid",
        bn=False,
        activation_fn=None,
        bias=True,
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.Conv2dTranspose if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            has_bias=bias,
            weight_init=TruncatedNormal(sigma=1e-3),
        )
        self.has_bn = bn
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99)
        self.activation_fn = activation_fn

    def construct(self, x):
        r"""
        construct method

        Parameters
        ----------
        x: ms.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        ms.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(x)
        if self.has_bn:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Cell):
    def __init__(self, in_channel=10, out_channel=1, use_pos_encoding=True, bias=True):
        super(LocalSpatialEncoding, self).__init__()

        self.mlp = SharedMLP(
            in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias
        )
        # self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.d = out_channel
        self.use_pos_encoding = use_pos_encoding

    def construct(self, coords, features, neighbor_idx):
        r"""
        construct method
        Parameters
        ----------
        coords: ms.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: ms.Tensor, shape (B, d, N, 1)
            features of the point cloud
        neighbor_idx: ms.Tensor, shape (B, N, K)
        Returns
        -------
        ms.Tensor, shape (B, 2*d, N, K)
        """

        idx = neighbor_idx  # (4,40960,16)

        cat = P.Concat(-3)
        if self.use_pos_encoding:
            # finding neighboring points
            extended_idx = P.Tile()(
                idx.expand_dims(1), (1, 3, 1, 1)
            )  # (4,40960,16) -> (4,1,40960,16) -> (4,3,40960,16)
            xyz_tile = P.Tile()(
                coords.transpose(0, 2, 1).expand_dims(-1), (1, 1, 1, idx.shape[-1])
            )  # (4,3,40960) -> (4,3,40960,16)
            neighbor_xyz = P.GatherD()(
                xyz_tile, 2, extended_idx
            )  # shape (4, 3, 40960, 16)
            relative_xyz = xyz_tile - neighbor_xyz  # relative_xyz

            relative_dist = P.Sqrt()(
                P.ReduceSum(keep_dims=True)(P.Square()(relative_xyz), -3)
            )

            # relative point position encoding
            f_xyz = cat(
                (
                    relative_dist,  # (4,1,40960,16)
                    relative_xyz,  # (4,3,40960,16)
                    xyz_tile,  # (4,3,40960,16)
                    neighbor_xyz,  # (4,3,40960,16)
                )
            )  # (4,10,40960,16)
        else:
            f_xyz = coords
        f_xyz = self.mlp(f_xyz)  # (4,10,40960,16) -> (4,8,40960,16)

        f_tile = P.Tile()(
            features, (1, 1, 1, idx.shape[-1])
        )  # (4, 8, 40960, 1) -> (4,8,40960,16)
        extended_idx_for_feat = P.Tile()(idx.expand_dims(1), (1, f_xyz.shape[1], 1, 1))
        f_neighbours = P.GatherD()(
            f_tile, 2, extended_idx_for_feat
        )  # (4,8,40960,16) -> (4,8,40960,16)

        f_concat = cat(
            [f_xyz, f_neighbours]
        )  # (4,8,40960,16) & (4,8,40960,16) -> (4,16,40960,16)

        if self.use_pos_encoding:
            return f_xyz, f_concat

        return f_concat


class AttentivePooling(nn.Cell):
    def __init__(self, in_channels, out_channels, bias):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.SequentialCell(
            [nn.Dense(in_channels, in_channels, has_bias=False), nn.Softmax(-2)]
        )
        self.mlp = SharedMLP(
            in_channels,
            out_channels,
            bn=True,
            activation_fn=nn.LeakyReLU(0.2),
            bias=bias,
        )

    def construct(self, x):
        r"""
        construct method
        Parameters
        ----------
        x: ms.Tensor, shape (B, d_in, N, K)
        Returns
        -------
        ms.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

        # sum over the neighbors
        features = scores * x
        features = P.ReduceSum(keep_dims=True)(features, -1)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Cell):
    def __init__(self, d_in, d_out, bias):
        super(LocalFeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(
            d_in, d_out // 2, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias
        )
        self.mlp2 = SharedMLP(d_out, 2 * d_out, bn=True, bias=bias)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True, bias=bias)

        self.lse1 = LocalSpatialEncoding(
            in_channel=10, out_channel=d_out // 2, use_pos_encoding=True, bias=bias
        )
        self.lse2 = LocalSpatialEncoding(
            in_channel=d_out // 2,
            out_channel=d_out // 2,
            use_pos_encoding=False,
            bias=bias,
        )

        self.pool1 = AttentivePooling(d_out, d_out // 2, bias=bias)
        self.pool2 = AttentivePooling(d_out, d_out, bias=bias)

        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, coords, features, neighbor_idx):
        r"""
        construct method
        Parameters
        ----------
        coords: ms.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: ms.Tensor, shape (B, d, N, 1)
            features of the point cloud
        neighbor_idx: ms.Tensor, shape (B, N, 16)
            knn neighbor idx
        Returns
        -------
        ms.Tensor, shape (B, 2*d_out, N, 1)
        """
        f_pc = self.mlp1(features)  # (4, 8, 40960, 1)

        f_xyz, f_concat = self.lse1(
            coords, f_pc, neighbor_idx
        )  # (4, 8, 40960, 16) (4, 16, 40960, 16)
        f_pc_agg = self.pool1(f_concat)  # (4, 8, 40960, 1)

        f_concat = self.lse2(
            f_xyz, f_pc_agg, neighbor_idx
        )  # coords: (4, 40960, 3); x: (4, 8, 40960, 1)  neighbor_idx:(4, 40960, 16)
        f_pc_agg = self.pool2(f_concat)

        return self.lrelu(self.mlp2(f_pc_agg) + self.shortcut(features))


class RandLANet(nn.Cell):
    def __init__(self, d_in, num_classes, bias, config):
        super(RandLANet, self).__init__()
        self.cfg = config
        self.score_layer_num = self.cfg.score_layer_num
        self.fc_start = nn.Dense(d_in, 8)
        self.bn_start = nn.SequentialCell(
            [nn.BatchNorm2d(8, eps=1e-6, momentum=0.99), nn.LeakyReLU(0.2)]
        )

        # encoding layers
        self.encoder = nn.CellList(
            [
                LocalFeatureAggregation(8, 16, bias=bias),
                LocalFeatureAggregation(32, 64, bias=bias),
                LocalFeatureAggregation(128, 128, bias=bias),
                LocalFeatureAggregation(256, 256, bias=bias),
                LocalFeatureAggregation(512, 512, bias=bias),
            ]
        )

        self.mlp = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(0.2))
        score_kwargs = dict(bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias)
        self.score_pred_layer = nn.CellList(
            [
                SharedMLP(32, 13, **score_kwargs),
                SharedMLP(32, 13, **score_kwargs),
                SharedMLP(128, 13, **score_kwargs),
                SharedMLP(256, 13, **score_kwargs),
            ]
        )
        # decoding layers
        decoder_kwargs = dict(
            transpose=True, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias
        )
        self.decoder = nn.CellList(
            [
                SharedMLP(1536, 512, **decoder_kwargs),
                SharedMLP(768, 256, **decoder_kwargs),
                SharedMLP(384, 128, **decoder_kwargs),
                SharedMLP(160, 32, **decoder_kwargs),
                SharedMLP(64, 32, **decoder_kwargs),
            ]
        )

        # final semantic prediction
        self.fc_end = nn.SequentialCell(
            [
                SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias),
                SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias),
                nn.Dropout(),
            ]
        )
        self.fc_score = SharedMLP(32, num_classes, bias=bias)
        self.fc_embed = SharedMLP(
            32, 32, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias
        )

    def data_prep(self, logits, embedding, labels, d):
        logits = logits.transpose(0, 2, 1)
        logits = logits.reshape((-1, 13))
        embedding = embedding.transpose(0, 2, 1)
        embedding = embedding.reshape((-1, d))
        labels = labels.reshape((-1))

        # ignored_bool = P.ZerosLike()(labels)
        shape_of_labels = P.Shape()(labels)
        ignored_bool = P.Fill()(mstype.bool_, shape_of_labels, False)

        for ign_label in self.cfg.ignored_label_inds:
            l = P.Equal()(labels, ign_label)
            ignored_bool = P.LogicalOr()(ignored_bool, l)

        not_ignored_bool = P.LogicalNot()(ignored_bool)

        valid_idx = P.NonZero()(not_ignored_bool).squeeze(1)

        # valid_idx = P.Squeeze()(P.where()(P.LogicalNot()(ignored_bool)))
        valid_logits = P.Gather()(logits, valid_idx, 0)
        valid_embedding = P.Gather()(embedding, valid_idx, 0)
        valid_labels_init = P.Gather()(labels, valid_idx, 0)

        reducing_list = P.Range()(
            P.Cast()(0, ms.int32), P.Cast()(13, ms.int32), P.Cast()(1, ms.int32)
        )

        inserted_value = P.Zeros()((1,), ms.int32)
        cat = P.Concat(0)
        for ign_label in self.cfg.ignored_label_inds:
            reducing_list = cat(
                [reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]]
            )

        valid_labels = P.Gather()(reducing_list, valid_labels_init, 0)
        return valid_labels, valid_logits, valid_embedding

    def data_prep_score(self, labels, score):
        labels = labels.reshape((-1, 13))
        score = score.transpose(0, 2, 3, 1)
        score = score.reshape((-1, 13))
        # valid_idx = tf.squeeze(tf.where(tf.logical_not(tf.equal(tf.reduce_sum(score, axis=-1), 0))))
        sum_score = P.ReduceSum(keep_dims=False)(score, -1)
        is_zero = P.Equal()(sum_score, 0)
        non_zero = P.LogicalNot()(is_zero)
        valid_idx = P.NonZero()(non_zero)

        valid_labels = P.Gather()(labels, valid_idx, 0)
        valid_score = P.Gather()(score, valid_idx, 0)

        return valid_labels, valid_score

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx, labels):
        r"""
        construct method
        Parameters
        ----------
        xyz: list of ms.Tensor, shape (num_layer, B, N_layer, 3), each layer xyz
        feature: ms.Tensor, shape (B, N, d), input feature [xyz ; feature]
        neighbor_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer knn neighbor idx
        sub_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer pooling idx
        interp_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 1), each layer interp idx
        Returns
        -------
        ms.Tensor, shape (B, num_classes, N)
            segmentation scores for each point
        """

        feature = self.fc_start(feature).swapaxes(-2, -1).expand_dims(-1)
        feature = self.bn_start(feature)  # shape (B, 8, N, 1)
        sub_sampling_ratio = [4, 4, 4, 4, 2]

        # <<<<<<<<<< ENCODER

        f_stack = []

        for i in range(5):
            # at iteration i, feature.shape = (B, d_layer, N_layer, 1)
            f_encoder_i = self.encoder[i](xyz[i], feature, neighbor_idx[i])
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_stack.append(f_encoder_i)
            f_stack.append(f_sampled_i)
        # # >>>>>>>>>> ENCODER

        feature = self.mlp(f_stack[-1])  # [B, d, N, 1]

        # <<<<<<<<<< DECODER
        f_decoder_list = []
        se_features_list = []
        for j in range(5):
            f_interp_i = self.random_sample(feature, interp_idx[-j - 1])  # [B, d, n, 1]
            # f_interp_i = self.nearest_interpolation(feature, interp_idx[-j - 1])
            cat = P.Concat(1)
            f_decoder_i = self.decoder[j](cat((f_stack[-j - 2], f_interp_i)))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        # >>>>>>>>>> DECODER
        multihot_labels = P.cast(P.one_hot(labels, 13), ms.int32)
        score_pred_list = []
        score_layer_num = self.score_layer_num
        num_points = self.cfg.num_points
        score_loss = 0
        fac = 1
        for i in range(1, score_layer_num + 1):
            score_pred = self.score_pred_layer[i - 1](f_decoder_list[-i])
            valid_labels, valid_score = self.data_prep_score(
                multihot_labels, score_pred
            )

            score_loss += fac * P.ReduceMean()(
                P.SigmoidCrossEntropyWithLogits()(valid_score, valid_labels.float())
            )
            score_pred = P.Sigmoid()(score_pred)
            if i > 1:
                for j in range(7 - i, cfg.num_layers + 1):
                    score_pred = self.random_sample(score_pred, interp_idx[-j])
                    # score_pred = self.nearest_interpolation(score_pred, inputs['interp_idx'][-j])
            score_pred_list.append(P.Squeeze(3)(score_pred))

            multihot_labels = P.ReduceMax()(
                self.gather_neighbour(multihot_labels, neighbor_idx[i - 1]), 2
            )
            multihot_labels = multihot_labels.expand_dims(2)
            multihot_labels = multihot_labels.transpose(0, 3, 1, 2)
            multihot_labels = P.Squeeze(3)(
                self.random_sample(multihot_labels, sub_idx[i - 1])
            )
            multihot_labels = multihot_labels.transpose(0, 2, 1)
            num_points = num_points // sub_sampling_ratio[i - 1]
            fac = fac * 0.9

        scores = f_decoder_list[-1]  #
        for i in range(3):
            scores = self.fc_end[i](scores)
        f_out = self.fc_score(scores)
        f_layer_embed = self.fc_embed(scores)

        return f_out.squeeze(-1), f_layer_embed.squeeze(-1), score_loss, score_pred_list

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, d, N', 1] pooled features matrix
        """

        b, d = feature.shape[:2]
        n_ = pool_idx.shape[1]
        # [B, N', max_num] --> [B, d, N', max_num]
        # pool_idx = P.repeat_elements(pool_idx.expand_dims(1), feature.shape[1], 1)
        pool_idx = P.Tile()(pool_idx.expand_dims(1), (1, feature.shape[1], 1, 1))
        # [B, d, N', max_num] --> [B, d, N'*max_num]
        pool_idx = pool_idx.reshape((b, d, -1))
        pool_features = P.GatherD()(feature.squeeze(-1), -1, pool_idx)
        pool_features = pool_features.reshape((b, d, n_, -1))
        pool_features = P.ReduceMax(keep_dims=True)(pool_features, -1)  # [B, d, N', 1]
        return pool_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        # return: (4, 40960, 16, 13)
        # pc: (4, 40960, 13)
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        idx = neighbor_idx
        extended_idx = P.Tile()(idx.expand_dims(1), (1, 13, 1, 1))  # (4, 13, 40960, 16)

        index_input = extended_idx.reshape(batch_size, d, -1)
        index_input = index_input.transpose(0, 2, 1)

        # (4, 13, 40960, 16)
        # print(xyz_tile.shape, extended_idx.shape)
        features = P.GatherD()(pc, 1, index_input)
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        return features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, 1, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, 1, d] interpolated features matrix
        """
        feature = P.squeeze(feature, axis=2)
        batch_size = P.shape(interp_idx)[0]
        up_num_points = P.shape(interp_idx)[1]
        interp_idx = P.reshape(interp_idx, (batch_size, up_num_points))
        interpolated_features = P.GatherD()(feature, interp_idx)
        interpolated_features = P.expand_dims(interpolated_features, axis=2)
        return interpolated_features


class get_loss(LossBase):
    def __init__(self):
        super(get_loss, self).__init__()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, logits, labels):
        one_hot_label = P.one_hot(labels, 13)
        unweighted_loss = self.ce(logits, one_hot_label.float())
        output_loss = P.ReduceMean()(unweighted_loss)
        return output_loss


class get_loss_ps(LossBase):
    def __init__(self):
        super(get_loss_ps, self).__init__()
        self.beta = 0.999
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_mean = P.ReduceMean()
        self.pow = P.Pow()
        self.divide = P.Div()
        self.subtract = P.Sub()
        self.cast = P.Cast()
        self.one_hot = P.OneHot()
        self.ones_like = P.OnesLike()
        self.gather = P.Gather()
        self.squeeze = P.Squeeze(axis=0)

    def sharpen(self, z):
        a = self.pow(z, 2)
        sum_a = self.reduce_sum(a, axis=1)
        a = self.divide(a, P.reshape(sum_a, (-1, 1)))
        return a

    def construct(self, logits, labels, probs):
        one_hot_labels = P.one_hot(labels, 13)
        one_hot_labels = P.cast(one_hot_labels, mstype.float32)

        num_per_class = self.reduce_sum(one_hot_labels, axis=0)
        num_per_class = self.squeeze(num_per_class)
        beta_tensor = self.beta * self.ones_like(num_per_class)

        beta_tensor = self.cast(beta_tensor, mstype.float32)

        per_class_weight = self.divide(
            1, self.subtract(1.0, self.pow(beta_tensor, num_per_class))
        )

        weight = self.gather(per_class_weight, labels, 0)

        probs2 = self.sharpen(
            probs
        )  # Assuming sharpen is a method defined in your class

        log_probs = nn.LogSoftmax(axis=1)(logits)
        unweighted_losses = -(log_probs * probs2).sum(axis=1)

        weighted_losses = weight * unweighted_losses
        output_loss = self.reduce_mean(weighted_losses)

        return output_loss


class ClusterLoss(LossBase):
    def __init__(self):
        super(ClusterLoss, self).__init__()
        self.fac = 0.5
        self.cast = P.Cast()
        self.one_hot = P.OneHot()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.matmul = P.MatMul()
        self.zeros_like = P.ZerosLike()
        self.mse_loss = MSELoss(reduction="sum")
        self.protype = P.L2Normalize(axis=1)(
            Tensor(mnp.ones((13, 32), dtype=mnp.float32))
        )

    def construct(
        self,
        val_embedding_ps,
        val_embedding_gt,
        valid_labels_ps,
        valid_labels_gt,
        valid_probs,
        protype,
    ):
        # print(protype.shape) 13*32
        protype = self.protype
        valid_labels_gt = self.cast(valid_labels_gt, mstype.int32)
        onehot_valid_labels_gt = P.one_hot(valid_labels_gt, 13)
        onehot_valid_labels_gt = P.cast(onehot_valid_labels_gt, mstype.float32)

        num_class_gt = self.reduce_sum(onehot_valid_labels_gt, axis=0).transpose(1, 0)
        embedding_wise_class_gt = self.matmul(
            onehot_valid_labels_gt.transpose(1, 0).astype(mstype.float32),
            val_embedding_gt,
        ) / (num_class_gt + 1e-4)
        valid_labels_ps = self.cast(valid_labels_ps, mstype.int32)
        onehot_valid_labels_ps = P.one_hot(valid_labels_ps, 13)
        onehot_valid_labels_ps = P.cast(onehot_valid_labels_ps, mstype.float32)

        num_class_ps = self.reduce_sum(onehot_valid_labels_ps, axis=0).transpose(1, 0)

        embedding_wise_class_ps = self.matmul(
            onehot_valid_labels_ps.transpose(1, 0).astype(mstype.float32),
            val_embedding_ps,
        ) / (num_class_ps + 1e-4)

        igter_bool_exist = ~(P.Squeeze(axis=1)(num_class_gt) == 0)

        non_zero_op = P.NonZero()
        indx_exist = non_zero_op(igter_bool_exist.astype(ms.float32))
        indx_exist = P.Squeeze(axis=1)(indx_exist)

        indx_noexist = non_zero_op(~igter_bool_exist.astype(ms.float32))
        indx_noexist = P.Squeeze(axis=1)(indx_noexist)

        embedding_update = embedding_wise_class_gt[indx_exist]
        protype_update = protype[indx_exist]
        protype_old = protype[indx_noexist]

        update_tensor = (1 - self.fac) * protype_update + self.fac * embedding_update

        protype_new = mnp.concatenate([update_tensor, protype_old], axis=0)

        idx = mnp.concatenate([indx_exist, indx_noexist], axis=0)

        sort_op = P.Sort(axis=0, descending=False)
        _, idx_new = sort_op(idx)

        protype_new = protype_new[idx_new]

        embedding_gt_to_ps = protype_new[valid_labels_ps]
        sub_ps_and_gt = val_embedding_ps - embedding_gt_to_ps

        cluster_loss = (
            self.mse_loss(sub_ps_and_gt, self.zeros_like(sub_ps_and_gt))
            / sub_ps_and_gt.shape[0]
        )
        embedding_gt_protype = protype_new[indx_exist]
        embedding_ps_protype = embedding_wise_class_ps[indx_exist]

        aff_gt = self.matmul(embedding_gt_protype, embedding_gt_protype.transpose(1, 0))
        aff_ps = self.matmul(embedding_ps_protype, embedding_ps_protype.transpose(1, 0))

        aff_gt_norm = aff_gt / mnp.clip(aff_gt.sum(axis=1, keepdims=True), 1e-10, 1.0)

        aff_ps_norm = aff_ps / mnp.clip(aff_ps.sum(axis=1, keepdims=True), 1e-10, 1.0)

        aff_gt_norm = mnp.clip(aff_gt_norm, 1e-10, 1.0)
        aff_ps_norm = mnp.clip(aff_ps_norm, 1e-10, 1.0)

        similar_loss = (
            aff_gt_norm * mnp.log(aff_gt_norm / (aff_ps_norm + 1e-10))
        ).sum() / aff_gt_norm.shape[0]

        self.protype = protype_new
        return cluster_loss + 0.1 * similar_loss


class get_matrix_loss(LossBase):

    def __init__(self):
        super(get_matrix_loss, self).__init__()

    def construct(self, embedding, labels):
        one_hot_labels = P.one_hot(labels, 13)
        num = one_hot_labels.shape[0]
        one_hot_labels = P.cast(one_hot_labels, mstype.float32)
        labels_matrix = P.MatMul()(
            one_hot_labels, P.Transpose()(one_hot_labels, (1, 0))
        )

        embedding_norm = P.L2Normalize(axis=1)(embedding)
        embedding_matrix = P.MatMul()(
            embedding_norm, P.Transpose()(embedding_norm, (1, 0))
        )

        sub_matrix = P.Sub()(labels_matrix, embedding_matrix)
        loss = P.L2Loss()(sub_matrix) / P.Cast()(num * num, ms.float32)
        return loss


class RandLAWithLoss(nn.Cell):
    """RadnLA-net with loss"""

    def __init__(self, network):
        super(RandLAWithLoss, self).__init__()
        self.get_loss = get_loss()
        self.get_matrix_loss = get_matrix_loss()
        self.get_loss_ps = get_loss_ps()
        self.cluster_loss = ClusterLoss()
        self.network = network

    def construct(
        self,
        xyz,
        feature,
        neighbor_idx,
        sub_idx,
        interp_idx,
        labels,
        labels_gt,
        input_inds,
        cloud_inds,
        ps_probs,
        current_epoch,
    ):
        logits, embedding, score_loss, _ = self.network(
            xyz, feature, neighbor_idx, sub_idx, interp_idx, labels
        )
        if current_epoch <= 5:
            w_clu = 0.0
            w_sco = 0.0
        else:
            w_clu = 1.0
            w_sco = 1.0
        d = embedding.shape[1]
        ps_probs = ps_probs.transpose(0, 2, 1)
        valid_labels_ps, valid_logits_ps, valid_embedding_ps = self.network.data_prep(
            logits, embedding, labels, d
        )

        _, valid_probs_ps, _ = self.network.data_prep(ps_probs, embedding, labels, d)

        valid_labels_gt, valid_logits_gt, valid_embedding_gt = self.network.data_prep(
            logits, embedding, labels_gt, d
        )

        label_loss_gt = 1.0 * self.get_loss(valid_logits_gt, valid_labels_gt)

        matrix_loss = self.get_matrix_loss(valid_embedding_gt, valid_labels_gt)

        label_loss_ps = self.get_loss_ps(
            valid_logits_ps, valid_labels_ps, valid_probs_ps
        )
        protype = P.L2Normalize(axis=1)(Tensor(mnp.ones((13, 32), dtype=mnp.float32)))

        score_loss = 1.5 * score_loss * w_sco
        embedding_norm_ps = P.L2Normalize(axis=1)(valid_embedding_ps)
        embedding_norm_gt = P.L2Normalize(axis=1)(valid_embedding_gt)
        cluster_loss = self.cluster_loss(
            embedding_norm_ps,
            embedding_norm_gt,
            valid_labels_ps,
            valid_labels_gt,
            valid_probs_ps,
            protype,
        )

        cluster_loss = 0.75 * w_clu * cluster_loss

        loss = label_loss_gt + score_loss + matrix_loss + label_loss_ps + cluster_loss

        return loss


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith(".bias"):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith(".gamma"):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith(".beta"):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{"params": no_decay_params, "weight_decay": 0.0}, {"params": decay_params}]


class TrainingWrapper(nn.Cell):
    """Training wrapper."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network_logits = self.network.network
        self.network.set_grad()
        self.opt_weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(
        self,
        xyz,
        feature,
        neighbor_idx,
        sub_idx,
        interp_idx,
        labels,
        labels_gt,
        input_inds,
        cloud_inds,
        ps_probs,
        current_epoch,
    ):
        """Build a forward graph"""

        # loss calculate
        loss = self.network(
            xyz,
            feature,
            neighbor_idx,
            sub_idx,
            interp_idx,
            labels,
            labels_gt,
            input_inds,
            cloud_inds,
            ps_probs,
            current_epoch,
        )
        # opt update
        opt_weights = self.opt_weights
        sens = op.Fill()(op.DType()(loss), op.Shape()(loss), self.sens)
        grads = self.grad(self.network, opt_weights)(
            xyz,
            feature,
            neighbor_idx,
            sub_idx,
            interp_idx,
            labels,
            labels_gt,
            input_inds,
            cloud_inds,
            ps_probs,
            current_epoch,
            sens,
        )
        res = P.depend(loss, self.optimizer(grads))
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="the number of GPUs to use [default: 0]"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="options: train, test, vis"
    )
    parser.add_argument(
        "--test_area",
        type=int,
        default=5,
        help="Which area to use for test, option: 1-6 [default: 5]",
    )
    parser.add_argument("--labeled_point", type=str, default="1", help="1, 10 or 100")
    parser.add_argument(
        "--model_name", type=str, default="RandLANet_S3DIS_stage2.py", help=""
    )
    parser.add_argument("--log_dir", type=str, default="ex", help="")
    parser.add_argument("--load_dir", type=str, default="1205", help="")
    parser.add_argument("--knn", type=int, default=16, help="k_nn")
    parser.add_argument(
        "--pseudo_label_path",
        type=str,
        default="./experiment_mindspore/S3DIS/1_points_/ex/prediction/pseudo_label",
        help="pseudo label path",
    )
    parser.add_argument("--total_log", type=str, default="ex", help="")
    parser.add_argument(
        "--gt_path", type=str, default="./S3DIS_gt_2", help="gt label path"
    )
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = S3DISDatasetGenerator(FLAGS.pseudo_label_path, FLAGS.gt_path)
    train, test = dataloader(dataset)
    train = train.batch(
        batch_size=cfg.batch_size,
        per_batch_map=ms_map,
        input_columns=[
            "xyz",
            "colors",
            "labels",
            "labels_gt",
            "probs",
            "q_idx",
            "c_idx",
        ],
        output_columns=[
            "features",
            "labels",
            "labels_gt",
            "probs",
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
    d_in = 6
    model = RandLANet(d_in, cfg.num_classes, True, cfg)
    restore_snap = "/home/ubuntu/hdd1/yhj/RPSC/experiment_mindspore/S3DIS/1_points_/ex/checkpoints/best_epoch_80.ckpt"
    param_dict = load_checkpoint(str(restore_snap))
    load_param_into_net(model, param_dict)
    network = RandLAWithLoss(model)
    train_loader = train.create_dict_iterator()
    for i, data in enumerate(train_loader):
        # data prepare
        features = data["features"]
        labels = data["labels"]
        labels_gt = data["labels_gt"]
        ps_probs = data["probs"]
        input_inds = data["input_inds"]
        cloud_inds = data["cloud_inds"]
        xyz = [data["p0"], data["p1"], data["p2"], data["p3"], data["p4"]]
        neigh_idx = [data["n0"], data["n1"], data["n2"], data["n3"], data["n4"]]
        sub_idx = [data["pl0"], data["pl1"], data["pl2"], data["pl3"], data["pl4"]]
        interp_idx = [data["u0"], data["u1"], data["u2"], data["u3"], data["u4"]]

        # predict
        loss = network(
            xyz,
            features,
            neigh_idx,
            sub_idx,
            interp_idx,
            labels,
            labels_gt,
            input_inds,
            cloud_inds,
            ps_probs,
            i,
        )
        print("i:", i, " loss:", loss)
