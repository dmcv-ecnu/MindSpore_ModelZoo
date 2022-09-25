import utils.data_process as dp
from models.common import SharedMLP, Conv2dTranspose
import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops


class RandLaNet(nn.Cell): # RandLaNet and drop last some fc.
    def __init__(self, d_in, layer_dims):
        super(RandLaNet, self).__init__()

        self.fc = SharedMLP(d_in, 8)
        self.encoders = nn.CellList()
        t_in = 8
        for i in layer_dims:
            # layer_dim indicate real d_out, but drb's output is 'd_out' argument * 2.
            self.encoders.append(DilatedResidualBlock(t_in, i))
            t_in = 2 * i

        self.mlp = SharedMLP(t_in, t_in)

        self.decoders = nn.CellList()
        for i in range(len(layer_dims) - 1):
            self.decoders.append(Conv2dTranspose(t_in + 2 * layer_dims[-2 - i], 2 * layer_dims[-2 - i]))
            t_in = 2 * layer_dims[-2 - i]
        self.decoders.append(Conv2dTranspose(2 * t_in, t_in))

        self.fc2 = SharedMLP(t_in, 32)

    def construct(self, xyz, feature, sub_idx, up_idx, neighbor_idx):
        feature = self.fc(feature)

        # ###########################Encoder############################
        en_result = []
        for i in range(len(self.encoders)):
            feature = self.encoders[i](xyz, feature, neighbor_idx[i])
            if i == 0:
                en_result.append(feature)
            # random sub feature.
            pi = ops.expand_dims(sub_idx[i], 2)
            pi = ops.repeat_elements(pi, neighbor_idx[i].shape[2], 2)
            p = ops.gather_d(neighbor_idx[i], 1, pi)
            feature = self.sample(feature, p)
            # random sub xyz.
            xyz = self.sample(xyz, sub_idx[i])
            en_result.append(feature)
        # ###########################Encoder############################

        feature = self.mlp(feature)

        # ###########################Decoder############################
        for i in range(len(self.decoders)):
            feature = self.sample(feature, up_idx[i])
            feature = self.decoders[i](msnp.concatenate([en_result[-2-i], feature], 1))
        # ###########################Decoder############################

        feature = self.fc2(feature)
        return feature

    @staticmethod
    def sample(x, idx):
        """
        :param feature: [B, d, N] input features matrix
        :param interp_idx: [B, N', k] nearest neighbour index [0-N]. dim 3 can be omitted.
        :return: [B, d, N'] interpolated features matrix
        """
        b, d = x.shape[:2]
        n_ = idx.shape[1]
        idx = ops.expand_dims(idx, 1)
        idx = ops.repeat_elements(idx, d, 1)
        idx = idx.reshape((b, d, -1))
        x = ops.gather_d(x, 2, idx)
        x = x.reshape(b, d, n_, -1)
        x = ops.ReduceMax()(x, 3)
        return x
    #
    # @staticmethod
    # def nearest_interpolation(feature, interp_idx):
    #     """
    #     :param feature: [B, d, N] input features matrix
    #     :param interp_idx: [B, N', 1] nearest neighbour index [0-N]
    #     :return: [B, d, N'] interpolated features matrix
    #     """
    #     interp_idx = msnp.squeeze(interp_idx, 2)
    #     d = feature.shape[1]
    #     interp_idx = msnp.expand_dims(interp_idx, 1)
    #     interp_idx = msnp.repeat(interp_idx, d, 1)
    #     pool_features = ops.gather_d(feature, 2, interp_idx)
    #     return pool_features


class DilatedResidualBlock(nn.Cell):
    def __init__(self, d_in, d_out):
        super(DilatedResidualBlock, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.mlp1 = SharedMLP(d_in, d_out // 2)

        self.block = BuildingBlock(d_out // 2, d_out)

        self.mlp2 = SharedMLP(d_out, d_out * 2, activation_fn=None)

        self.mlp3 = SharedMLP(d_in, d_out * 2, activation_fn=None)

        self.lrelu1 = nn.LeakyReLU(alpha=0.2)

    def construct(self, xyz, feature, neighbor_idx):
        feature_clone = feature
        feature = self.mlp1(feature)
        feature = self.block(xyz, feature, neighbor_idx)
        feature = self.mlp2(feature)

        feature_shortcut = self.mlp3(feature_clone)

        output = self.lrelu1(feature + feature_shortcut)
        return output


class BuildingBlock(nn.Cell):
    def __init__(self, d_in, d_out):
        super(BuildingBlock, self).__init__()

        self.mlp1 = SharedMLP(10, d_in, input_1d=False)
        self.att1 = AttentionPooling(2 * d_in, d_out // 2)

        self.norm1 = nn.Norm(1, keep_dims=True)

        self.mlp2 = SharedMLP(d_in, d_out // 2, input_1d=False)
        self.att2 = AttentionPooling(d_out, d_out)

    def construct(self, xyz, feature, neighbor_idx):
        f_xyz = self.relative_pos_encoding(xyz, neighbor_idx)
        f_xyz = self.mlp1(f_xyz)
        f_neighbors = dp.gather_neighbour(feature, neighbor_idx)
        f_concat = msnp.concatenate([f_neighbors, f_xyz], axis=1)
        f_pc_agg = self.att1(f_concat)

        f_xyz = self.mlp2(f_xyz)
        f_neighbors = dp.gather_neighbour(f_pc_agg, neighbor_idx)
        f_concat = msnp.concatenate([f_neighbors, f_xyz], axis=1)
        f_pc_agg = self.att2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neighbor_idx):
        neighbor_xyz = dp.gather_neighbour(xyz, neighbor_idx)  # [b, d, n, k]
        xyz = msnp.expand_dims(xyz, 3)
        xyz = msnp.repeat(xyz, neighbor_idx.shape[-1], 3)
        relative_xyz = xyz - neighbor_xyz

        relative_dis = self.norm1(relative_xyz)
        # pi + pik + (pi - pik) + ||pi - pik||
        relative_feature = msnp.concatenate([relative_dis, relative_xyz, xyz, neighbor_xyz], 1)
        return relative_feature


class AttentionPooling(nn.Cell):
    def __init__(self, d_in, d_out):
        super(AttentionPooling, self).__init__()
        self.att_map = nn.SequentialCell(
            nn.Conv2d(d_in, d_in, (1, 1), (1, 1), has_bias=False),
            nn.Softmax(3)
        )
        self.mlp = SharedMLP(d_in, d_out)

    def construct(self, x):
        att = self.att_map(x)
        x = x * att
        x = ops.ReduceSum()(x, 3)
        x = self.mlp(x)
        return x


