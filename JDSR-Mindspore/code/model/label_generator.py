from mindspore import nn

from model.common import *

def make_label_generator(args):
    args = None
    return None


class labelGenerator(nn.Cell):
    def __init__(self, args, auto_prefix=True, flags=None, conv=make_default_conv):
        super().__init__(auto_prefix, flags)

        scale = args.scale
        n_feats = 40
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        kernel_size = 3

        self.T_tdm3 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU()])
        self.L_tdm3 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU()])

        self.T_tdm2 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU()])
        self.L_tdm2 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU()])

        self.T_tdm1 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU()])
        self.L_tdm1 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU()])

        self.tail3 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode="pad", has_bias=True),
            nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=3, pad_mode="pad", has_bias=True),
            PixelShuffle(scale)])

        self.tail2 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode="pad", has_bias=True),
            nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=3, pad_mode="pad", has_bias=True),
            PixelShuffle(scale)])

        self.tail1 = nn.SequentialCell([
            nn.Conv2d(n_feats, n_feats, kernel_size=3, pad_mode="same", has_bias=True),
            nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=3, pad_mode="same", has_bias=True),
            PixelShuffle(scale)])


    def construct(self, feature_map4, feature_map3, feature_map2, feature_map1, feature_map0, x):
        concat = ms.ops.Concat(axis=1)
        T_tdm3 = self.T_tdm3(feature_map4)
        L_tdm3 = self.L_tdm3(feature_map3)
        out_TDM3 = concat((T_tdm3, L_tdm3)) + feature_map0
        label3 = self.tail3(out_TDM3) + x

        T_tdm2 = self.T_tdm2(out_TDM3)
        L_tdm2 = self.L_tdm2(feature_map2)
        out_TDM2 = concat((T_tdm2, L_tdm2)) + feature_map0
        label2 = self.tail2(out_TDM2) + x

        T_tdm1 = self.T_tdm1(out_TDM2)
        L_tdm1 = self.L_tdm1(feature_map1)
        out_TDM1 = concat((T_tdm1, L_tdm1)) + feature_map0
        label1 = self.tail1(out_TDM1) + x

        return [label1, label2, label3]