import mindspore.nn as nn
from collections import OrderedDict
from mindspore import ops
import mindspore as ms
import src.util as util


class NodeUpdateNetwork(nn.Cell):
    def __init__(self,
                 in_features,
                 num_features,
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        ratio = [2, 1]
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                has_bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            use_batch_statistics=True)
            layer_list['relu{}'.format(l)] = nn.LeakyReLU(alpha=1e-2)

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout(keep_prob=1. - self.dropout)

        self.network = nn.SequentialCell(layer_list)

    def construct(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.shape[0]
        num_data = node_feat.shape[1]

        # get eye matrix (batch_size x 2 x node_size x node_size)
        tmp = ops.Eye()(num_data, num_data, ms.float32)
        tmp = ops.ExpandDims()(tmp, 0)
        tmp = ops.ExpandDims()(tmp, 0)
        tmp = ops.Tile()(tmp, (num_tasks, 2, 1, 1))
        diag_mask = 1.0 - tmp

        # set diagonal as zero and normalize
        edge_feat = util.l1_norm(edge_feat * diag_mask, -1)

        # compute attention and aggregate
        tmp = ops.Split(axis=1, output_num=2)(edge_feat)
        tmp = ops.Concat(2)(tmp)  # (batch, 1, 2*N, N)
        tmp = ops.Squeeze(axis=1)(tmp)  # (batch, 2*N, N)
        aggr_feat = ops.BatchMatMul()(tmp, node_feat)  # (batch, 2*N, emb)
        node_feat = ops.Concat(axis=-1)([node_feat, ops.Concat(-1)(ops.Split(1, 2)(aggr_feat))])
        node_feat = node_feat.swapaxes(1, 2)
        # non-linear transform
        node_feat = self.network(ops.ExpandDims()(node_feat, -1))
        node_feat = ops.Squeeze(-1)(node_feat.swapaxes(1, 2))
        # print('node_feat_1: ', node_feat.shape)
        return node_feat


class EdgeUpdateNetwork(nn.Cell):
    def __init__(self,
                 in_features,
                 num_features,
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        ratio = [2, 2, 1, 1]
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                has_bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            use_batch_statistics=True)
            layer_list['relu{}'.format(l)] = nn.LeakyReLU(alpha=1e-2)

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout(keep_prob=1. - self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1,
                                           has_bias=True)
        self.sim_network = nn.SequentialCell(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(
                    in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features,
                    out_channels=self.num_features_list[l],
                    kernel_size=1,
                    has_bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                use_batch_statistics=True)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU(alpha=1e-2)

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(keep_prob=1. - self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1,
                                               has_bias=True)
            self.dsim_network = nn.SequentialCell(layer_list)

    def construct(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = ops.ExpandDims()(node_feat, 2)
        x_j = x_i.swapaxes(1, 2)
        x_ij = ops.Abs()(x_i - x_j)
        x_ij = ops.Transpose()(x_ij, (0, 3, 2, 1))

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = ops.Sigmoid()(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = ops.Sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val

        tmp = ops.Eye()(node_feat.shape[1], node_feat.shape[1], ms.float32)
        tmp = ops.ExpandDims()(tmp, 0)
        tmp = ops.ExpandDims()(tmp, 0)
        tmp = ops.Tile()(tmp, (node_feat.shape[0], 2, 1, 1))
        diag_mask = 1.0 - tmp
        edge_feat = edge_feat * diag_mask
        merge_sum = ops.ReduceSum(keep_dims=True)(edge_feat, -1)
        # set diagonal as zero and normalize
        edge_feat = ops.Concat(1)([sim_val, dsim_val]) * edge_feat  # TODO
        edge_feat = util.l1_norm(edge_feat, -1) * merge_sum

        # force_edge_feat = torch.cat(
        #     (torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)),
        #     0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1)

        tmp1 = ops.Eye()(node_feat.shape[1], node_feat.shape[1], ms.float32)
        tmp1 = ops.ExpandDims()(tmp1, 0)
        tmp2 = ops.Zeros()((node_feat.shape[1], node_feat.shape[1]), ms.float32)
        tmp2 = ops.ExpandDims()(tmp2, 0)
        force_edge_feat = ops.Concat(0)((tmp1, tmp2))
        force_edge_feat = ops.ExpandDims()(force_edge_feat, 0)
        force_edge_feat = ops.Tile()(force_edge_feat, (node_feat.shape[0], 1, 1, 1))

        # f0 = ops.ExpandDims()(ops.Eye()(node_feat.shape[1], node_feat.shape[1], ms.float32), -1)
        # f1 = ops.ExpandDims()(ops.Zeros()((node_feat.shape[1], node_feat.shape[1]), ms.float32), -1)
        # force_out = ops.Concat(-1)([f0, f1])
        # force_out = ops.Stack()([force_out] * node_feat.shape[0])
        # force_out = ops.Transpose()(force_out, (0, 3, 1, 2))

        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_sum = ops.ExpandDims()(edge_feat.sum(1), 1)
        edge_sum = ops.Tile()(edge_sum, (1, 2, 1, 1))
        edge_feat = ops.Div()(edge_feat, edge_sum)

        return edge_feat
