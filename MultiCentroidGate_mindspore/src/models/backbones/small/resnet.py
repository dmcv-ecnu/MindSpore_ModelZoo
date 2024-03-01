# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from torch.nn import functional as F
from mindspore import ops
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeNormal

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, remove_last_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

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
        if not self.remove_last_relu:
            out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
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


class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layers,
                 nf=64,
                 zero_init_residual=True,
                 dataset='cifar',
                 start_class=0,
                 remove_last_relu=False):
        super(ResNet, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.inplanes = nf
        self.conv1 = nn.SequentialCell(nn.Conv2d(3, nf, kernel_size=3, stride=1, pad_mode='pad', padding=1),
                                       nn.BatchNorm2d(nf, momentum=0.1), 
                                       nn.ReLU())

        self.layer1 = self._make_layer(block, 1 * nf, layers[0])
        self.layer2 = self._make_layer(block, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, layers[3], stride=2, remove_last_relu=remove_last_relu)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.mean = ops.ReduceMean(keep_dims=True)  # 据说速度比上一行要快  Soap
        
        self.out_dim = 8 * nf * block.expansion

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                                                 cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer("zeros", cell.beta.shape, cell.beta.dtype))

        if zero_init_residual:
            for _, cell in self.cells_and_names():  # 疑似官方文档bug  Soap
                if isinstance(cell, Bottleneck) and cell.bn3.gamma is not None:
                    cell.bn3.gamma.set_data(initializer("zeros", cell.bn3.gamma.shape, cell.bn3.gamma.dtype))
                elif isinstance(cell, BasicBlock) and cell.bn2.gamma is not None:
                    cell.bn2.gamma.set_data(initializer("zeros", cell.bn2.gamma.shape, cell.bn2.gamma.dtype))

    def _make_layer(self, block, planes, blocks, remove_last_relu=False, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if remove_last_relu:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, remove_last_relu=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def reset_bn(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                # 暂时这么写，不知道对错  Soap
                cell.gamma.set_data(initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer("zeros", cell.beta.shape, cell.beta.dtype))
                # m.reset_running_stats()

    def construct(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # 暂时没有预训练模型
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


