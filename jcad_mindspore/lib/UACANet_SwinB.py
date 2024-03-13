import mindspore
from mindspore import nn
import mindspore.ops as ops
import numpy as np

from .modules.optim.losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.SwinTransformer import SwinB

class UACANet_SwinB(nn.Cell):
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(UACANet_SwinB, self).__init__()
        self.backbone = SwinB()

        self.context1 = ColorSpace(128, 32)
        self.context2 = ColorSpace(256, 64)
        self.context3 = ColorSpace(512, 128)
        self.context4 = ColorSpace(1024, 256)
        self.context5 = ColorSpace(1024, 256)

        self.up1 = UP(128)
        self.up2 = UP(256)
        self.up3 = UP(512)
        self.up4 = UP(1024)
        self.up5 = UP(1024)

        # self.context1 = ColorSpace2(96, 24)
        # self.context2 = ColorSpace2(192, 48)
        # self.context3 = ColorSpace1(384, 96)
        # self.context4 = ColorSpace(768, 192)
        # self.context5 = ColorSpace(768, 192)
        #
        # self.up1 = UP(96)
        # self.up2 = UP(192)
        # self.up3 = UP(384)
        # self.up4 = UP(768)
        # self.up5 = UP(768)

        self.decoder = SimpleDecoder(256, 256, 128, channel=128)

        self.attention3 = simple_attention(128, 32, size=12)
        self.attention2 = simple_attention(48, 12, size=14)
        self.attention1 = simple_attention(64, 16, size=24)

        self.loss_fn = bce_iou_loss
        self.loss_cls = nn.CrossEntropyLoss()

        self.ret = lambda x, target: ops.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: ops.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channels)

    def construct(self, sample):
        x = sample['image']
        x = ops.permute(x, (0, 3, 1, 2)) # [B, C, H, W]

        if 'gt' in sample.keys():
            y = sample.get('gt', None).float()
            y = ops.unsqueeze(y, dim=1)
            y = y / 255.0
        else:
            y = None

        if 'contours' in sample.keys():
            contours = sample.get('contours', None).int()
        else:
            contours = None

        B, _, H, W = x.shape
        # B, H, W, _ = x.shape


        x1 = self.backbone.forward_features(x)
        # x1 [B, 9216, 96]
        x2 = self.backbone.layers[0](x1)
        x3 = self.backbone.layers[1](x2)
        x4 = self.backbone.layers[2](x3)
        x5 = self.backbone.layers[3](x4)

        # x1 = x1.view((B, H // 4, W // 4, -1))
        # x1 = ops.permute(x1, (0, 3, 1, 2))
        # x1 = ops.permute(x1.view(B, H // 4, W // 4, -1), (0, 3, 1, 2)).contiguous()
        x1 = ops.permute(x1.view(B, H // 4, W // 4, -1), (0, 3, 1, 2))
        x2 = ops.permute(x2.view(B, H // 8, W // 8, -1), (0, 3, 1, 2))
        x3 = ops.permute(x3.view(B, H // 16, W // 16, -1), (0, 3, 1, 2))
        x4 = ops.permute(x4.view(B, H // 32, W // 32, -1), (0, 3, 1, 2))
        x5 = ops.permute(x5.view(B, H // 32, W // 32, -1), (0, 3, 1, 2))
        context5 = self.context5(x5)
        # print(f"context5: {context5.shape}")#(4, 768, 7, 7)
        context4 = self.context4(x4)
        # print(f"context4: {context4.shape}")#(4, 768, 7, 7)
        context3 = self.context3(x3)
        # print(f"context3: {context3.shape}")#(4, 384, 14, 14)
        context2 = self.context2(x2)
        # print(f"context2: {context2.shape}")#(4, 192, 28, 28)
        context1 = self.context1(x1)
        # print(f"context1: {context1.shape}")#(4, 96, 56, 56)

        up5 = self.up5(context5)
        # print(f"up5: {up5.shape}")
        up4 = self.up4(context4)
        # print(f"up4: {up4.shape}")
        up3 = self.up3(context3)
        # print(f"up3: {up3.shape}")
        up2 = self.up2(context2)
        # print(f"up2: {up2.shape}")
        up1 = self.up1(context1)
        # print(f"up1: {up1.shape}")

        _, a5 = self.decoder(up5, up4, up3)
        # print(f"a5: {a5.shape}")
        out5 = self.res(a5, (H, W))
        # print(f"out5: {out5.shape}")

        _, a3, cnt3 = self.attention3(up3, a5)
        # print(f"a3: {a3.shape}")
        out3 = self.res(a3, (H, W))
        # print(f"out3: {out3.shape}")

        _, a2, cnt2 = self.attention2(up2, a3)
        # print(f"a2: {a2.shape}")
        out2 = self.res(a2, (H, W))
        # print(f"out2: {out2.shape}")

        _, a1, cnt1 = self.attention1(up1, a2)
        out1 = self.res(a1, (H, W))
        # print(f"out1: {out1.shape}")

        if y is not None:

            loss1 = self.loss_fn(out1, y)
            loss2 = self.loss_fn(out2, y)
            loss3 = self.loss_fn(out3, y)

            loss5 = self.loss_fn(out5, y)

            # print()
            # print(f"cnt1: {cnt1}")
            # print(f"cnt2: {cnt2}")
            # print(f"cnt3: {cnt3}")
            cnt_loss1 = self.loss_cls(cnt3, contours)
            cnt_loss2 = self.loss_cls(cnt2, contours)
            cnt_loss3 = self.loss_cls(cnt1, contours)

            loss = loss1 + loss2 + loss3 + loss5 + 0.01 * (cnt_loss1 + cnt_loss2 + cnt_loss3)

            debug = [out5, out3, out2]

        else:
            loss = 0
            debug = []

        return {'pred': out1, 'loss': loss, 'debug': debug, 'contours': cnt1}
        # return out1, loss, debug, cnt1