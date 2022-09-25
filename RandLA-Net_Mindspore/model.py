"""
Author: Qihang Ma
Date: Sep 2022
"""
import time

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as op
from mindspore import Tensor, context
from mindspore.common.initializer import TruncatedNormal

from pathlib import Path
from dataset import ms_map, dataloader
from utils.tools import DataProcessing as DP
from utils.tools import ConfigS3DIS as cfg



class SharedMLP(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        pad_mode='same',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.Conv2dTranspose if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode='valid',
            has_bias=True,
            weight_init=TruncatedNormal(sigma=1e-3)
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def construct(self, input):
        r"""
            construct method

            Parameters
            ----------
            input: ms.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            ms.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Cell):
    def __init__(self, d):
        super(LocalSpatialEncoding, self).__init__()

        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.d = d

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
        # finding neighboring points
        idx= neighbor_idx
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        # extended_idx = P.repeat_elements(idx.expand_dims(1), 3, 1)
        extended_idx = P.Tile()(idx.expand_dims(1), (1,3,1,1))
        # extended_coords = P.repeat_elements(coords.transpose(0,2,1).expand_dims(-1), idx.shape[-1], -1)
        extended_coords = P.Tile()(coords.transpose(0,2,1).expand_dims(-1), (1,1,1,idx.shape[-1]))
        neighbors = P.GatherD()(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        relative_coords = extended_coords - neighbors
        relative_dist = P.Sqrt()(P.Square()(relative_coords))
        relative_dist = P.ReduceSum(keep_dims=True)(relative_dist, -3)#shape (B, 1, N, K)

        # relative point position encoding
        cat = P.Concat(-3)
        # concat = cat((
        #     extended_coords,
        #     neighbors,
        #     extended_coords - neighbors,
        #     relative_dist
        # ))
        concat = cat((
            relative_dist,
            relative_coords,
            extended_coords,
            neighbors,
        ))
        return cat((
            self.mlp(concat),
            # P.repeat_elements(features, idx.shape[-1], -1)
            P.Tile()(features, (1,1,1,idx.shape[-1]))
        ))



class AttentivePooling(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.SequentialCell([
            nn.Dense(in_channels, in_channels, has_bias=False),
            nn.Softmax(-2)
        ])
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

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
        scores = self.score_fn(x.transpose(0,2,3,1)).transpose(0,3,1,2)

        # sum over the neighbors
        features = scores * x
        features = P.ReduceSum(keep_dims=True)(features, -1) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Cell):
    def __init__(self, d_in, d_out):
        super(LocalFeatureAggregation, self).__init__()


        self.mlp1 = SharedMLP(d_in, d_out//2, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out, bn=True)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2)
        self.lse2 = LocalSpatialEncoding(d_out//2)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

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

        x = self.mlp1(features)

        x = self.lse1(coords, x, neighbor_idx)
        x = self.pool1(x)

        x = self.lse2(coords, x, neighbor_idx)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Cell):
    def __init__(self, d_in, num_classes):
        super(RandLANet, self).__init__()
        
        self.fc_start = nn.Dense(d_in, 8)
        self.bn_start = nn.SequentialCell([
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ])

        # encoding layers
        self.encoder = nn.CellList([
            LocalFeatureAggregation(8, 16),
            LocalFeatureAggregation(32, 64),
            LocalFeatureAggregation(128, 128),
            LocalFeatureAggregation(256, 256),
            LocalFeatureAggregation(512, 512)
        ])

        self.mlp = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(0.2))

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.LeakyReLU(0.2)
        )
        self.decoder = nn.CellList([
            SharedMLP(1536, 512, **decoder_kwargs),
            SharedMLP(768, 256, **decoder_kwargs),
            SharedMLP(384, 128, **decoder_kwargs),
            SharedMLP(160, 32, **decoder_kwargs),
            SharedMLP(64, 32, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.SequentialCell([
            SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        ])

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx):
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
       
        feature = self.fc_start(feature).swapaxes(-2,-1).expand_dims(-1)
        feature = self.bn_start(feature) # shape (B, 8, N, 1)

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

        feature = self.mlp(f_stack[-1]) # [B, d, N, 1]

        # <<<<<<<<<< DECODER
        
        f_decoder_list = []
        for j in range(5):
            f_interp_i = self.random_sample(feature, interp_idx[-j-1]) # [B, d, n, 1]
            cat = P.Concat(1)
            f_decoder_i = self.decoder[j](cat((f_stack[-j-2], f_interp_i)))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        # >>>>>>>>>> DECODER

        scores = self.fc_end(f_decoder_list[-1]) # [B, num_classes, N, 1]

        return scores.squeeze(-1)
    
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
        pool_features = P.ReduceMax(keep_dims=True)(pool_features, -1) # [B, d, N', 1]
        return pool_features


class RandLAWithLoss(nn.Cell):
    """RadnLA-net with loss"""
    def __init__(self, network, weights, num_classes):
        super(RandLAWithLoss, self).__init__()
        self.network = network
        self.weights = weights
        self.num_classes = num_classes
        self.onehot = nn.OneHot(depth = num_classes)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, feature, labels, input_inds, cloud_inds,p0,p1,p2,p3,p4,n0,n1,n2,n3,n4,pl0,pl1,pl2,pl3,pl4,u0,u1,u2,u3,u4):
        xyz = [p0,p1,p2,p3,p4]
        neighbor_idx = [n0,n1,n2,n3,n4]
        sub_idx = [pl0,pl1,pl2,pl3,pl4]
        interp_idx = [u0,u1,u2,u3,u4]
        logits = self.network(xyz, feature, neighbor_idx, sub_idx, interp_idx)
        logit = logits.swapaxes(-2,-1).reshape((-1, self.num_classes)) # [b*n, 13]
        labels = labels.reshape((-1, )) # [b, n] --> [b*n]
        one_hot_labels = self.onehot(labels) # [b*n, 13]
        #self.weights = weights.expand_dims(0) # [13,] --> [1, 13]
        weights = self.weights * one_hot_labels # [b*n, 13]
        weights = P.ReduceSum()(weights, 1) # [b*n]
        unweighted_loss = self.loss_fn(logit, one_hot_labels) # [b*n]
        weighted_loss = unweighted_loss * weights # [b*n]
        CE_loss = weighted_loss.mean() # [1]

        #return CE_loss, logits
        return CE_loss



class TrainingWrapper(nn.Cell):
    """Training wrapper."""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, *args):
        """Build a forward graph"""
        weights = self.weights
        loss, logits = self.network(*args)
        sens = op.Fill()(op.DType()(loss), op.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        return F.depend(loss, self.optimizer(grads)), logits


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]



if __name__ == '__main__':
    import time
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)
    context.set_context(max_call_depth=2048)
    
    print('Computing weights...')

    n_samples = Tensor(cfg.class_weights, ms.float32)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples + 0.02)
    weights.expand_dims(axis=0)

    print('Done')
    
    d_in = 6
    model = RandLANet(d_in, cfg.num_classes)
    #network = RandLAWithLoss(model, weights, cfg.num_classes)

    
    dir = Path('/home/ubuntu/hdd1/mqh/dataset/s3dis/train_0.040')
    print('generating data loader....')
    train_ds, val_ds = dataloader(dir, val_area='Area_5' ,num_parallel_workers=8, shuffle=False)
    train_loader = train_ds.batch(batch_size=4,
                                  per_batch_map=ms_map,
                                  input_columns=["xyz","colors","labels","q_idx","c_idx"],
                                  output_columns=["features","labels","input_inds","cloud_inds",
                                                  "p0","p1","p2","p3","p4",
                                                  "n0","n1","n2","n3","n4",
                                                  "pl0","pl1","pl2","pl3","pl4",
                                                  "u0","u1","u2","u3","u4"],
                                  num_parallel_workers=8,
                                  drop_remainder=True)
    train_loader = train_loader.create_dict_iterator()
    for i, data in enumerate(train_loader):
        # data prepare
        features = data['features']
        labels = data['labels']
        xyz = [data['p0'],data['p1'],data['p2'],data['p3'],data['p4']]
        neigh_idx = [data['n0'],data['n1'],data['n2'],data['n3'],data['n4']]
        sub_idx = [data['pl0'],data['pl1'],data['pl2'],data['pl3'],data['pl4']]
        interp_idx = [data['u0'],data['u1'],data['u2'],data['u3'],data['u4']]

        # predict
        pred = model(xyz, features, neigh_idx, sub_idx, interp_idx)
        print('step:',i, ' pred.shape:', pred.shape)

