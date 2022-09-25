from mindspore import context, Tensor, Model
from mindspore import dtype as mstype
import mindspore.dataset.vision.py_transforms as py_transforms

from mindspore.ops import stop_gradient
import mindspore.nn as nn
import mindspore
import mindspore_hub as mshub
from mindspore import nn


class Vgg19(nn.Cell):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="CPU",
                            device_id=0)
        model = mshub.load("mindspore/ascend/1.3/vgg16_v1.3_imagenet2012", pretrained=True)

        #assert isinstance(model.features, object)
        vgg_pretrained_features = model.get_parameters()
        self.slice1 = mindspore.nn.SequentialCell()
        self.slice2 = mindspore.nn.SequentialCell()
        self.slice3 = mindspore.nn.SequentialCell()
        self.slice4 = mindspore.nn.SequentialCell()
        self.slice5 = mindspore.nn.SequentialCell()
        for x in range(2):
            self.slice1.insert_child_to_cell(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.insert_child_to_cell(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.insert_child_to_cell(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.insert_child_to_cell(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.insert_child_to_cell(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def construct(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Cell):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def construct(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            y_vgg[i] = stop_gradient(y_vgg[i])
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

