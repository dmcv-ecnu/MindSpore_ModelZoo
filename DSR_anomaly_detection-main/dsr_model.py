import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as mnp


class SubspaceRestrictionModule(nn.Cell):
    def __init__(self, embedding_size=64):
        super(SubspaceRestrictionModule, self).__init__()

        base_width = embedding_size
        self.unet = SubspaceRestrictionNetwork(in_channels=base_width, out_channels=base_width, base_width=embedding_size)

    def construct(self, x, quantization):
        x = self.unet(x)
        loss_b, quantized_b, perplexity_b, encodings_b = quantization(x)
        return x, quantized_b, loss_b

class SubspaceRestrictionNetwork(nn.Cell):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.base_width = base_width
        self.encoder = FeatureEncoder(in_channels, self.base_width)
        self.decoder = FeatureDecoder(self.base_width, out_channels=out_channels)

    def construct(self, x):
        b1, b2, b3 = self.encoder(x)
        output = self.decoder(b1, b2, b3)
        return output

class FeatureEncoder(nn.Cell):
    def __init__(self, in_channels, base_width):
        super().__init__()
        self.block1 = nn.SequentialCell(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width, affine=False),
            nn.ReLU())
        self.mp1 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))
        self.block2 = nn.SequentialCell(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width * 2, affine=False),
            nn.ReLU())
        self.mp2 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))
        self.block3 = nn.SequentialCell(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width * 4, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width * 4, affine=False),
            nn.ReLU())

    def construct(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b1, b2, b3

class FeatureDecoder(nn.Cell):
    def __init__(self, base_width, out_channels=1):
        super().__init__()

        self.up2 = nn.SequentialCell(nn.Upsample(scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=True),  # question, scale_factor
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
                                 nn.InstanceNorm2d(base_width * 2, affine=False),
                                 nn.ReLU())

        self.db2 = nn.SequentialCell(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width * 2, affine=False),
            nn.ReLU()
        )

        self.up3 = nn.SequentialCell(nn.Upsample(scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1, pad_mode='pad'),
                                 nn.InstanceNorm2d(base_width, affine=False),
                                 nn.ReLU())
        self.db3 = nn.SequentialCell(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            nn.InstanceNorm2d(base_width, affine=False),
            nn.ReLU()
        )

        self.fin_out = nn.SequentialCell(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1, pad_mode='pad'))

    def construct(self, b1, b2, b3):
        up2 = self.up2(b3)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        out = self.fin_out(db3)
        return out

class Residual(nn.Cell):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, has_bias=False, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,     # question 好怪这里，那还要两个参数来干嘛
                      kernel_size=1, stride=1, has_bias=False, pad_mode='pad')
        )

    def construct(self, x):
        return x + self._block(x)


class ResidualStack(nn.Cell):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.CellList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def construct(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return ops.relu(x)


class ImageReconstructionNetwork(nn.Cell):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ImageReconstructionNetwork, self).__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(in_channels, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(in_channels*2, affine=False),
            nn.ReLU())
        self.mp1 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))
        self.block2 = nn.SequentialCell(
            nn.Conv2d(in_channels*2, in_channels * 2, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(in_channels * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3),
            norm_layer(in_channels * 4, affine=False),
            nn.ReLU())
        self.mp2 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))

        self.pre_vq_conv = nn.Conv2d(in_channels=in_channels*4, out_channels=64, kernel_size=1, stride=1, pad_mode='pad')

        #self.vq = VectorQuantizerEMA(512, 64, 0.25, 0.99)

        self.upblock1 = nn.Conv2dTranspose(in_channels=64,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2, padding=1, pad_mode='pad')    # question

        self.upblock2 = nn.Conv2dTranspose(in_channels=64,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2, padding=1, pad_mode='pad')

        self._conv_1 = nn.Conv2d(in_channels=64,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1, pad_mode='pad')

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.Conv2dTranspose(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1, pad_mode='pad')

        self._conv_trans_2 = nn.Conv2dTranspose(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1, pad_mode='pad')

    def construct(self, inputs):
        x = self.block1(inputs)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)
        x = self.pre_vq_conv(x)

        x = self.upblock1(x)
        x = ops.relu(x)
        x = self.upblock2(x)
        x = ops.relu(x)
        x = self._conv_1(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = ops.relu(x)

        return self._conv_trans_2(x)




class UnetEncoder(nn.Cell):
    def __init__(self, in_channels, base_width):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.SequentialCell(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width, affine=False),
            nn.ReLU())
        self.mp1 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))
        self.block2 = nn.SequentialCell(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 2, affine=False),
            nn.ReLU())
        self.mp2 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))
        self.block3 = nn.SequentialCell(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 4, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 4, affine=False),
            nn.ReLU())
        self.mp3 = nn.SequentialCell(nn.MaxPool2d(2, stride=2, pad_mode='pad'))
        self.block4 = nn.SequentialCell(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 4, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 4, affine=False),
            nn.ReLU())

    def construct(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


class UnetDecoder(nn.Cell):
    def __init__(self, base_width, out_channels=1):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.up1 = nn.SequentialCell(nn.Upsample(scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
                                 norm_layer(base_width * 4, affine=False),
                                 nn.ReLU())
        # cat with base*4
        self.db1 = nn.SequentialCell(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 4, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 4, affine=False),
            nn.ReLU()
        )

        self.up2 = nn.SequentialCell(nn.Upsample(scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
                                 norm_layer(base_width * 2, affine=False),
                                 nn.ReLU())
        # cat with base*2
        self.db2 = nn.SequentialCell(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 2, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width * 2, affine=False),
            nn.ReLU()
        )

        self.up3 = nn.SequentialCell(nn.Upsample(scale_factor=2.0, recompute_scale_factor=True, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1, pad_mode='pad'),
                                 norm_layer(base_width, affine=False),
                                 nn.ReLU())
        # cat with base*1
        self.db3 = nn.SequentialCell(
            nn.Conv2d(base_width * (1 + 1), base_width, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width, affine=False),
            nn.ReLU(),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, pad_mode='pad'),
            norm_layer(base_width, affine=False),
            nn.ReLU()
        )

        self.fin_out = nn.SequentialCell(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1, pad_mode='pad'))

    def construct(self, b1, b2, b3, b4):

        up1 = self.up1(b4)
        cat1 = ops.cat((up1, b3), axis=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = ops.cat((up2, b2), axis=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = ops.cat((up3, b1), axis=1)
        db3 = self.db3(cat3)

        out = self.fin_out(db3)
        return out

class UnetModel(nn.Cell):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def construct(self, x):
        b1, b2, b3, b4 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4)
        return output

class AnomalyDetectionModule(nn.Cell):
    def __init__(self, embedding_size=64):
        super(AnomalyDetectionModule, self).__init__()
        self.unet = UnetModel(in_channels=6, out_channels=2, base_width=64)
        
    def construct(self, image_real, image_anomaly):
        img_x = ops.cat((image_real, image_anomaly), axis=1)
        x = self.unet(img_x)
        return x


class UpsamplingModule(nn.Cell):
    def __init__(self, embedding_size=64):
        super(UpsamplingModule, self).__init__()
        self.unet = UnetModel(in_channels=8, out_channels=2, base_width=64)
        #self.unet = UNetNormalSkip(in_channels=4 * embedding_size + 16, out_channels=embedding_size)
    def construct(self, image_real, image_anomaly, segmentation_map):
        img_x = ops.cat((image_real, image_anomaly, segmentation_map), axis=1)
        x = self.unet(img_x)
        return x
