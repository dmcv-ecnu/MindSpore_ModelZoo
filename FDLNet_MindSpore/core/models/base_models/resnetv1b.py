import mindspore
from mindspore import nn,ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer, HeNormal, Constant

__all__ = ['ResNetV1b', 'resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b',
           'resnet101_v1b', 'resnet152_v1b', 'resnet152_v1s', 'resnet101_v1s', 'resnet50_v1s']


class BasicBlockV1b(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               pad_mode='pad', padding=dilation, dilation=dilation, has_bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad_mode='pad', padding=previous_dilation,
                               dilation=previous_dilation, has_bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, has_bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               pad_mode='pad', padding=dilation, dilation=dilation, has_bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, has_bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1b(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if deep_stem:
            self.conv1 = nn.SequentialCell(
                nn.Conv2d(3, 64, 3, 2, pad_mode='pad', padding=1, has_bias=False),
                norm_layer(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, pad_mode='pad', padding=1, has_bias=False),
                norm_layer(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1, pad_mode='pad', padding=1, has_bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, pad_mode='valid')
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512 * block.expansion, num_classes)

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight_init = HeNormal(mode='fan_out', nonlinearity='relu')
        #
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, BottleneckV1b):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlockV1b):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, has_bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        mid = list()
        for i in range(0, x.ndim-2):
            mid.append((0, 0))
        mid.append((1, 0))
        mid.append((1, 0))
        pad = ops.Pad(tuple(mid))#pytorch的pad优先在左上角，tensorflow和mindspore都在右下角，所以这里要手动添加
        x = pad(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        return x


def _create_resnet(path, pretrained=False, **kwargs):
    model = ResNetV1b(**kwargs)
    if pretrained :
        param_dict = load_checkpoint(path)
        load_param_into_net(model, param_dict)
    return model

def resnet18_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    old_dict = dict(block=BasicBlockV1b, layers=[2, 2, 2, 2], **kwargs)
    model_dict = model.parameters_dict()
    old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    return _create_resnet(pretrained, **model_dict)


def resnet34_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    old_dict = dict(block=BasicBlockV1b, layers=[3, 4, 6, 3], **kwargs)
    model_dict = model.parameters_dict()
    old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    return _create_resnet(pretrained, **model_dict)


def resnet50_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], **kwargs)
    old_dict = dict(block=BottleneckV1b, layers=[3, 4, 6, 3], **kwargs)
    model_dict = model.parameters_dict()
    old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    return _create_resnet(pretrained, **model_dict)


def resnet101_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], **kwargs)
    old_dict = dict(block=BottleneckV1b, layers=[3, 4, 23, 3], **kwargs)
    model_dict = model.parameters_dict()
    old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    return _create_resnet(pretrained, **model_dict)


def resnet152_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], **kwargs)
    old_dict = dict(block=BottleneckV1b, layers=[3, 8, 36, 3], **kwargs)
    model_dict = model.parameters_dict()
    old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    model_dict.update(old_dict)
    return _create_resnet(pretrained, **model_dict)


def resnet50_v1s(pretrained=False, **kwargs):
    model_args = dict(block=BottleneckV1b, layers=[3, 4, 6, 3], deep_stem=True, **kwargs)
    return _create_resnet("/tmp/pretrainmodel/ms_weight_res50.ckpt", pretrained, **model_args)
    # model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)
    # if pretrained:
    #     from ..model_store import get_resnet_file
    #     model.load_state_dict(torch.load(get_resnet_file('resnet50', root=root)), strict=False)
    # return model


def resnet101_v1s(pretrained=False, **kwargs):
    model_args = dict(block=BottleneckV1b, layers=[3, 4, 23, 3], deep_stem=True, **kwargs)
    return _create_resnet("/tmp/pretrainmodel/ms_weight_res101.ckpt", pretrained, **model_args)


def resnet152_v1s(pretrained=False, **kwargs):
    model_args = dict(block=BottleneckV1b, layers=[3, 8, 36, 3], deep_stem=True, **kwargs)
    return _create_resnet(pretrained, **model_args)

