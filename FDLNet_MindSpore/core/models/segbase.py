"""Base Model for Semantic Segmentation"""
from mindspore import nn,ops
from .base_models.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
from ..nn import JPU

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Cell):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_construct(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        mid = list()
        for i in range(0, x.ndim - 2):
            mid.append((0, 0))
        mid.append((1, 0))
        mid.append((1, 0))
        pad = ops.Pad(tuple(mid))  # pytorch的pad优先在左上角，tensorflow和mindspore都在右下角，所以这里要手动添加
        x = pad(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.construct(x)[0]

    def demo(self, x):
        pred = self.construct(x)
        if self.aux:
            pred = pred[0]
        return pred
