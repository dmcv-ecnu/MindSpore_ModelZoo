import mindspore
import mindspore.numpy as msnp

from models.edge_conv import *
from models.randla_net import *
import numpy as np
from mindspore import dtype, Parameter, Tensor


class PSD(nn.Cell):
    def __init__(self, d_in, num_points, num_classes, layer_dim):
        super(PSD, self).__init__()

        self.data_aug_att_fc = nn.Dense(num_points, 1, has_bias=False)
        self.sftmax = nn.Softmax()

        self.num_points = num_points

        self.mirror_opt = 1  # np.random.choice([0, 1, 2])
        self.m1_R = None
        self.m2_jitter = None
        self.init_aug()

        self.backbone = RandLaNet(3 + d_in, layer_dim)
        self.edgeconv = EdgeConv(32)
        self.norm1 = nn.Norm(1, keep_dims=True)
        self.mlp2 = SharedMLP(64, 32) # concat with edge conv
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.mlp3 = SharedMLP(32, num_classes, use_bn=False, activation_fn=None)

    def init_aug(self):
        if self.mirror_opt == 1:
            theta = 2 * 3.14592653 * np.random.rand()
            m = np.array([[np.cos(theta), 0, -np.sin(theta)],
                            [0, 1, 0],
                            [np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)

            self.m1_R = Parameter(Tensor(m, dtype=dtype.float32), requires_grad=False)
        elif self.mirror_opt == 2:
            sigma = 0.01
            clip = 0.05
            t = np.random.rand(3, self.num_points).astype(np.float32)
            jittered_point = np.clip(sigma * t, -clip, clip)[None, :]
            self.m2_jitter = Parameter(Tensor(jittered_point, dtype=dtype.float32), requires_grad=False)

    def construct(self, xyz, feature, sub_idx, up_idx, neighbor_idx):
        feature = msnp.concatenate([xyz, feature], 1)
        if self.training:
            xyz_, feature_ = self.data_augment(feature)
            xyz = msnp.concatenate([xyz, xyz])
            feature = msnp.concatenate([feature, feature_])
            rep_fn = lambda x: msnp.concatenate([x, x])
            sub_idx = [*map(rep_fn, sub_idx)]
            up_idx = [*map(rep_fn, up_idx)]
            neighbor_idx = [*map(rep_fn, neighbor_idx)]

        x = self.backbone(xyz, feature, sub_idx, up_idx, neighbor_idx)
        embd = self.edgeconv(x, neighbor_idx[0])
        x = msnp.concatenate((x, embd), 1)

        x = self.mlp2(x)
        x = self.dropout(x)
        x = self.mlp3(x)

        dis = self.norm1(embd)
        embd /= dis
        return x, embd

    def data_augment(self, feature):
        B, D, N = feature.shape
        xyz = feature[:, :3, :]
        feature = feature[:, 3:, :]

        # mirror_opt = self.mirror_opt
        # if mirror_opt == 0:
        xyz = msnp.stack([xyz[:, 0, :], -xyz[:, 1, :], xyz[:, 2, :]], 1)
        # elif mirror_opt == 1:
        #     xyz = msnp.transpose(xyz, [0, 2, 1])
        #     xyz = xyz.reshape(-1, 3)
        #     xyz = msnp.dot(xyz, self.m1_R)
        #     xyz = xyz.reshape(-1, N, 3)
        #     xyz = msnp.transpose(xyz, [0, 2, 1])
        # elif mirror_opt == 2:
        #     xyz = xyz + self.m2_jitter

        data_aug = msnp.concatenate([xyz, feature], 1)
        att_activation = self.data_aug_att_fc(data_aug)
        att_t = self.sftmax(msnp.transpose(att_activation, [0, 2, 1]))
        att = msnp.transpose(att_t, [0, 2, 1])
        data_aug = data_aug * att
        return xyz, data_aug