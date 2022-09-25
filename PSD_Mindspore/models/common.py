import mindspore.nn as nn
import mindspore.numpy as msnp


class SharedMLP(nn.Cell):
    def __init__(self, d_in, d_out, use_bn=True, use_bias=True,
                 activation_fn=nn.LeakyReLU(alpha=0.2), input_1d=True):
        super(SharedMLP, self).__init__()
        self.fc = nn.Conv2d(d_in, d_out, (1, 1), (1, 1), has_bias=use_bias)
        # mindspore's bn1d doesn't support 3d batch normalization
        if use_bn:
            self.bn = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.99) # TODO: momenten?
        self.activation_fn = activation_fn

        self.use_bn = use_bn
        self.input_1d = input_1d

    def construct(self, x):
        if self.input_1d:
            x = msnp.expand_dims(x, 2)
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        if self.input_1d:
            x = msnp.squeeze(x, 2)
        return x


class Conv2dTranspose(nn.Cell):
    def __init__(self, d_in, d_out, use_bn=True, use_bias=True,
                 activation_fn=nn.LeakyReLU(alpha=0.2), input_1d=True):
        super(Conv2dTranspose, self).__init__()
        # mindspore disables bias. pytorch & tf don't.
        # self.fc = nn.Conv2d(d_in, d_out, (1, 1), (1, 1), 'valid')
        # self.bn = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.99)
        self.fc = nn.Conv2dTranspose(d_in, d_out, (1, 1), (1, 1), has_bias=use_bias)
        # mindspore's bn1d doesn't support 3d batch normalization
        if use_bn:
            self.bn = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.99)
        self.activation_fn = activation_fn

        self.use_bn = use_bn
        self.input_1d = input_1d

    def construct(self, x):
        if self.input_1d:
            x = msnp.expand_dims(x, 2)
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        if self.input_1d:
            x = msnp.squeeze(x, 2)
        return x