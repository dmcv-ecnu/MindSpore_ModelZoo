import mindspore.nn as nn
from mindspore.ops import operations as ops
from .layers import *

class SpatialBlock(nn.Cell):
    def __init__(self, channel):
        super(SpatialBlock, self).__init__()
        self.conv1 = Conv(channel, channel, kernel_size=3, relu=True)
        self.conv2 = Conv(channel, channel, kernel_size=3, relu=True)

    def construct(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        tar = x + conv2
        return tar

class DownSample(nn.Cell):
    def __init__(self, out_c, size):
        super(DownSample, self).__init__()
        self.size = size
        self.conv1 = Conv(out_c, out_c, kernel_size=3, stride=2, relu=True)
        self.conv2 = Conv(out_c, out_c, kernel_size=3, stride=2, relu=True)

    def construct(self, x):
        if self.size == 2:
            tar = self.conv1(x)
            return tar
        elif self.size == 4:
            conv1 = self.conv1(x)
            tar = self.conv2(conv1)
            return tar


# class DownSample2(nn.Cell):
#     def __init__(self, out_c, size):
#         super(DownSample2, self).__init__()
#         self.size = size
#         self.conv1 = Conv(out_c, out_c, kernel_size=3, stride=2, relu=True)
#         self.conv2 = Conv(out_c, out_c, kernel_size=3, stride=2, relu=True)
#
#     def construct(self, x):
#         if self.size == 2:
#             tar = self.conv1(x)
#             return tar
#         elif self.size == 4:
#             conv1 = self.conv1(x)
#             tar = self.conv2(conv1)
#             return tar
#

# class up_sample(nn.Cell):
#     def __init__(self, channel, size):
#         super(up_sample, self).__init__()
#         self.size = size
#         self.up1 = nn.Sequential(nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1),
#                                  nn.BatchNorm2d(channel),
#                                  nn.ReLU(True))
#         self.up2 = nn.Sequential(nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=2, padding=1),
#                                  nn.BatchNorm2d(channel),
#                                  nn.ReLU(True))
#
#     def forward(self, x):
#         if self.size == 2:
#             up1 = self.up1(x)
#             return up1
#         elif self.size == 4:
#             up1 = self.up1(x)
#             up2 = self.up2(up1)
#             return up2


class UpSample(nn.Cell):
    def __init__(self, channel, size):
        super(UpSample, self).__init__()
        self.size = size
        self.up1 = nn.SequentialCell([
            nn.Conv2dTranspose(channel, channel, kernel_size=4, stride=2, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        ])
        self.up2 = nn.SequentialCell([
            nn.Conv2dTranspose(channel, channel, kernel_size=4, stride=2, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        ])

    def construct(self, x):
        if self.size == 2:
            up1 = self.up1(x)
            return up1
        elif self.size == 4:
            up1 = self.up1(x)
            up2 = self.up2(up1)
            return up2



# class UpSample2(nn.Cell):
#     def __init__(self, channel, size):
#         super(UpSample2, self).__init__()
#         self.size = size
#         self.up1 = nn.SequentialCell([
#             nn.Conv2dTranspose(channel, channel, kernel_size=4, stride=2, padding=1, pad_mode="pad"),
#             nn.BatchNorm2d(channel),
#             nn.ReLU()
#         ])
#         self.up2 = nn.SequentialCell([
#             nn.Conv2dTranspose(channel, channel, kernel_size=4, stride=2, padding=1, pad_mode="pad"),
#             nn.BatchNorm2d(channel),
#             nn.ReLU()
#         ])
#
#     def construct(self, x):
#         if self.size == 2:
#             up2 = self.up2(x)
#             return up2
#         elif self.size == 4:
#             up1 = self.up1(x)
#             up2 = self.up2(up1)
#             return up2
#

class ColorSpace(nn.Cell):
    def __init__(self, in_c, out_c):
        super(ColorSpace, self).__init__()

        # Assuming conv, down_sample, spatical_block, up_sample, CRM, and SRM are already defined for MindSpore
        self.pre1 = Conv(in_c, out_c, kernel_size=3, relu=True)
        self.pre2 = Conv(in_c, out_c, kernel_size=3, relu=True)
        self.pre3 = Conv(in_c, out_c, kernel_size=3, relu=True)

        self.x2_down = DownSample(out_c, 2)
        self.x4_down = DownSample(out_c, 4)

        self.x1_spatical = SpatialBlock(out_c)
        self.x2_spatical = SpatialBlock(out_c)
        self.x4_spatical = SpatialBlock(out_c)

        self.x2_up21 = UpSample(out_c, 2)
        self.x4_up21 = UpSample(out_c, 4)

        self.x1_down22 = DownSample(out_c, 2)
        self.x4_up22 = UpSample(out_c, 3)

        self.x1_down24 = DownSample(out_c, 4)
        self.x2_down24 = DownSample(out_c, 2)

        self.x2_up = UpSample(out_c, 2)
        self.x4_up = UpSample(out_c, 4)
        self.cat = Conv(out_c * 3, in_c, 3, relu=True)

        self.crm = CRM(in_c)
        self.srm = SRM(in_c)

    def construct(self, x):
        x1_pre = self.pre1(x)
        x2_pre = self.pre2(x)
        x4_pre = self.pre3(x)

        x2_ori = self.x2_down(x2_pre)
        x4_ori = self.x4_down(x4_pre)

        x1_s = self.x1_spatical(x1_pre)
        x2_s = self.x2_spatical(x2_ori)
        x4_s = self.x4_spatical(x4_ori)

        x2_up21 = self.x2_up21(x2_s)
        x4_up21 = self.x4_up21(x4_s)
        x1_res = x1_s - x2_up21 - x4_up21
        x1_tar = x1_pre * x1_res

        x1_down22 = self.x1_down22(x1_s)
        x4_up22 = self.x4_up22(x4_s)
        x2_res = x2_s - x1_down22 - x4_up22
        x2_tar = x2_ori * x2_res

        x1_down24 = self.x1_down24(x1_s)
        x2_down24 = self.x2_down24(x2_s)
        x4_res = x4_s - x1_down24 - x2_down24
        x4_tar = x4_ori * x4_res

        x2_up = self.x2_up(x2_tar)
        x4_up = self.x4_up(x4_tar)
        concat = ops.Concat(1)([x1_tar, x2_up, x4_up])
        cat = self.cat(concat)
        srm = self.srm(cat)
        crm = self.crm(cat)
        fuse = srm * crm

        tar = fuse + x

        return tar

class UP(nn.Cell):
    def __init__(self, in_channel):
        super(UP, self).__init__()
        self.out_channel_2 = int(in_channel / 2)
        self.out_channel_4 = int(in_channel / 4)
        self.conv2 = Conv(in_channel, self.out_channel_2, kernel_size=3, dilation=1, relu=True)
        self.conv4 = Conv(self.out_channel_2, self.out_channel_4, kernel_size=3, dilation=1, relu=True)

    def construct(self, x):
        conv2 = self.conv2(x)
        conv4 = self.conv4(conv2)
        return conv4

class CRM(nn.Cell):
    """
        Channel-wise Refinement Module
    """
    def __init__(self, dim):
        super(CRM, self).__init__()

        r = 16

        self.conv_1 = nn.Conv2d(dim, dim // r, kernel_size=1, stride=1, pad_mode='valid')
        self.conv_2 = nn.Conv2d(dim // r, dim, kernel_size=1, stride=1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.max_pool = ops.ReduceMax(keep_dims=True)

    def construct(self, feature):
        # Global Average Pooling
        gap = self.avg_pool(feature, (2, 3))
        # Global Max Pooling
        gmp = self.max_pool(feature, (2, 3))

        gap = self.conv_1(gap)
        gap = self.relu(gap)
        gap = self.conv_2(gap)

        gmp = self.conv_1(gmp)
        gmp = self.relu(gmp)
        gmp = self.conv_2(gmp)

        x = gap + gmp
        x = self.sigmoid(x)

        x = feature * x

        return x
    
class SRM(nn.Cell):
    """
        Spatial Refinement Module
    """
    def __init__(self, channels):
        super(SRM, self).__init__()

        kernel_size = 7

        self.conv_1_1 = nn.Conv2d(channels, channels,
                                  kernel_size=(kernel_size, 1),
                                  pad_mode='pad',
                                  padding=(kernel_size // 2, kernel_size // 2, 0, 0))
        self.conv_1_2 = nn.Conv2d(channels, channels,
                                  kernel_size=(1, kernel_size),
                                  pad_mode='pad',
                                  padding=(0, 0, kernel_size // 2, kernel_size // 2))

        self.conv_2_1 = nn.Conv2d(channels, channels,
                                  kernel_size=(1, kernel_size),
                                  pad_mode='pad',
                                  padding=(0, 0, kernel_size // 2, kernel_size // 2))
        self.conv_2_2 = nn.Conv2d(channels, channels,
                                  kernel_size=(kernel_size, 1),
                                  pad_mode='pad',
                                  padding=(kernel_size // 2, kernel_size // 2, 0, 0))

        self.relu = nn.ReLU()
        self.sigmoid = ops.Sigmoid()

    def construct(self, features):
        x_1 = self.conv_1_1(features)
        x_1 = self.relu(x_1)
        x_1 = self.conv_1_2(x_1)
        x_1 = self.relu(x_1)

        x_2 = self.conv_2_1(features)
        x_2 = self.relu(x_2)
        x_2 = self.conv_2_2(x_2)
        x_2 = self.relu(x_2)

        x = x_1 + x_2
        x = self.sigmoid(x)
        x = features * x

        return x

# class ColorSpace1(nn.Cell):
#     def __init__(self, in_c, out_c):
#         super(ColorSpace1, self).__init__()
#
#         # Assuming conv, down_sample, spatical_block, up_sample, CRM, and SRM are already defined for MindSpore
#         self.pre1 = Conv(in_c, out_c, kernel_size=3, relu=True)
#         self.pre2 = Conv(in_c, out_c, kernel_size=3, relu=True)
#         self.pre3 = Conv(in_c, out_c, kernel_size=3, relu=True)
#
#         self.x2_down = DownSample(out_c, 2)
#         self.x4_down = DownSample(out_c, 4)
#
#         self.x1_spatical = SpatialBlock(out_c)
#         self.x2_spatical = SpatialBlock(out_c)
#         self.x4_spatical = SpatialBlock(out_c)
#
#         self.x2_up21 = UpSample(out_c, 2)
#         self.x4_up21 = UpSample(out_c, 4)
#
#         self.x1_down22 = DownSample(out_c, 2)
#         self.x4_up22 = UpSample(out_c, 2)
#
#         self.x1_down24 = DownSample(out_c, 4)
#         self.x2_down24 = DownSample(out_c, 2)
#
#         self.x2_up = UpSample(out_c, 2)
#         self.x4_up = UpSample(out_c, 4)
#         self.cat = Conv(out_c * 3, in_c, 3, relu=True)
#
#         self.crm = CRM(in_c)
#         self.srm = SRM(in_c)
#
#     def construct(self, x):
#         x1_pre = self.pre1(x)
#         x2_pre = self.pre2(x)
#         x4_pre = self.pre3(x)
#
#         x2_ori = self.x2_down(x2_pre)
#         x4_ori = self.x4_down(x4_pre)
#
#         x1_s = self.x1_spatical(x1_pre)
#         x2_s = self.x2_spatical(x2_ori)
#         x4_s = self.x4_spatical(x4_ori)
#
#         x2_up21 = self.x2_up21(x2_s)
#         x4_up21 = self.x4_up21(x4_s)
#         x1_res = x1_s - x2_up21 - x4_up21
#         x1_tar = x1_pre * x1_res
#
#         x1_down22 = self.x1_down22(x1_s)
#         x4_up22 = self.x4_up22(x4_s)
#         x2_res = x2_s - x1_down22 - x4_up22
#         x2_tar = x2_ori * x2_res
#
#         x1_down24 = self.x1_down24(x1_s)
#         x2_down24 = self.x2_down24(x2_s)
#         x4_res = x4_s - x1_down24 - x2_down24
#         x4_tar = x4_ori * x4_res
#
#         x2_up = self.x2_up(x2_tar)
#         x4_up = self.x4_up(x4_tar)
#         concat = ops.Concat(1)([x1_tar, x2_up, x4_up])
#         cat = self.cat(concat)
#         srm = self.srm(cat)
#         crm = self.crm(cat)
#         fuse = srm * crm
#
#         tar = fuse + x
#
#         return tar

# class ColorSpace2(nn.Cell):
#     def __init__(self, in_c, out_c):
#         super(ColorSpace2, self).__init__()
#
#         # Assuming conv, down_sample, spatical_block, up_sample, CRM, and SRM are already defined for MindSpore
#         self.pre1 = Conv(in_c, out_c, kernel_size=3, relu=True)
#         self.pre2 = Conv(in_c, out_c, kernel_size=3, relu=True)
#         self.pre3 = Conv(in_c, out_c, kernel_size=3, relu=True)
#
#         self.x2_down = DownSample(out_c, 2)
#         self.x4_down = DownSample(out_c, 4)
#
#         self.x1_spatical = SpatialBlock(out_c)
#         self.x2_spatical = SpatialBlock(out_c)
#         self.x4_spatical = SpatialBlock(out_c)
#
#         self.x2_up21 = UpSample2(out_c, 2)
#         self.x4_up21 = UpSample2(out_c, 4)
#
#         self.x1_down22 = DownSample(out_c, 2)
#         self.x4_up22 = UpSample2(out_c, 2)
#
#         self.x1_down24 = DownSample(out_c, 4)
#         self.x2_down24 = DownSample(out_c, 2)
#
#         self.x2_up = UpSample2(out_c, 2)
#         self.x4_up = UpSample2(out_c, 4)
#         self.cat = Conv(out_c * 3, in_c, 3, relu=True)
#
#         self.crm = CRM(in_c)
#         self.srm = SRM(in_c)
#
#     def construct(self, x):
#         x1_pre = self.pre1(x)
#         x2_pre = self.pre2(x)
#         x4_pre = self.pre3(x)
#
#         x2_ori = self.x2_down(x2_pre)
#         x4_ori = self.x4_down(x4_pre)
#
#         x1_s = self.x1_spatical(x1_pre)
#         x2_s = self.x2_spatical(x2_ori)
#         x4_s = self.x4_spatical(x4_ori)
#
#         x2_up21 = self.x2_up21(x2_s)
#         x4_up21 = self.x4_up21(x4_s)
#         x1_res = x1_s - x2_up21 - x4_up21
#         x1_tar = x1_pre * x1_res
#
#         x1_down22 = self.x1_down22(x1_s)
#         x4_up22 = self.x4_up22(x4_s)
#         x2_res = x2_s - x1_down22 - x4_up22
#         x2_tar = x2_ori * x2_res
#
#         x1_down24 = self.x1_down24(x1_s)
#         x2_down24 = self.x2_down24(x2_s)
#         x4_res = x4_s - x1_down24 - x2_down24
#         x4_tar = x4_ori * x4_res
#
#         x2_up = self.x2_up(x2_tar)
#         x4_up = self.x4_up(x4_tar)
#         concat = ops.Concat(1)([x1_tar, x2_up, x4_up])
#         cat = self.cat(concat)
#         srm = self.srm(cat)
#         crm = self.crm(cat)
#         fuse = srm * crm
#
#         tar = fuse + x
#
#         return tar