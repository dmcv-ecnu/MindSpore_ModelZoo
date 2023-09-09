from mindspore import nn, ops
from mindcv.models.vgg import vgg16


class FCN32s(nn.Cell):
    """There are some difference from original fcn"""

    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN32s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.head = _FCNHead(512, nclass, norm_layer)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def construct(self, x):
        size = x.shape[2:]
        pool5 = self.pretrained(x)

        outputs = []
        out = self.head(pool5)
        out = ops.interpolate(out, size, mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = ops.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)


class FCN16s(nn.Cell):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN16s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool4 = nn.SequentialCell(*self.pretrained[:24])
        self.pool5 = nn.SequentialCell(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool4 = nn.Conv2d(512, nclass, 1, has_bias=True)#默认为false
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive', ['head', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool4'])

    def construct(self, x):
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)

        outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)

        upscore2 = ops.interpolate(score_fr, score_pool4.shape[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        out = ops.interpolate(fuse_pool4, x.shape[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = ops.interpolate(auxout, x.shape[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)


class FCN8s(nn.Cell):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN8s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool3 = nn.SequentialCell(*self.pretrained[:17])
        self.pool4 = nn.SequentialCell(*self.pretrained[17:24])
        self.pool5 = nn.SequentialCell(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool3 = nn.Conv2d(256, nclass, 1, has_bias=True)#默认为false
        self.score_pool4 = nn.Conv2d(512, nclass, 1, has_bias=True)#默认为false
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive',
                         ['head', 'score_pool3', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool3',
                                                                                         'score_pool4'])

    def construct(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = ops.interpolate(score_fr, score_pool4.shape[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = ops.interpolate(fuse_pool4, score_pool3.shape[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        out = ops.interpolate(fuse_pool3, x.shape[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = ops.interpolate(auxout, x.shape[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)


class _FCNHead(nn.Cell):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.SequentialCell(
            nn.Conv2d(in_channels, inter_channels, 3, pad_mode='pad', padding=1, has_bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(inter_channels, channels, 1, has_bias=True)#默认为false
        )

    def construct(self, x):
        return self.block(x)


if __name__ == '__main__':
    model = FCN16s(21)
    print(model)
