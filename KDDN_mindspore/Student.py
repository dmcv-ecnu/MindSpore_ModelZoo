import mindspore
import mindspore.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad',padding=1, has_bias=False)

class SELayer(nn.Cell):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        #self.avg_pool = mindspore.ops.AdaptiveAvgPool2D(1)
        self.avg_pool2 = mindspore.ops.ReduceMean(True)
        self.fc = nn.SequentialCell(
            nn.Dense(channel, channel),
            nn.ReLU(),#nn.ReLU(inplace=True),
            nn.Dense(channel, channel),
            nn.Sigmoid()
        )

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool2(x,(2, 3)).view(b, c)
        #y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y*x


class SEBasicBlockSW1(nn.Cell):
    def __init__(self, inplanes, planes, stride=1, with_norm=False):
        super(SEBasicBlockSW1, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes, 1)

        self.model1 = nn.SequentialCell(
            nn.Conv2d(planes, planes, kernel_size=3, pad_mode='pad',padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, pad_mode='pad',padding=1, stride=1),
            nn.Sigmoid()
        )

        self.se = SELayer(planes)
        self.relu = nn.ReLU()

    def construct(self, x):
        residule = x
        out = self.conv1(x)
        out = self.relu(out)
        out0 = self.conv2(out)

        sw = self.model1(out0)

        out1 = sw * out0

        out = self.se(out1)
        out = residule + out

        return out

class Conv2dBlock(nn.Cell):
    def __init__(self, in_dim, out_dim, ks, st, padding=((0,0),(0,0),(0,0),(0,0)),
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.Pad(padding, 'REFLECT')
        elif pad_type == 'replicate':
            self.pad = nn.Pad(padding, 'CONSTANT')
        elif pad_type == 'zero':
            self.pad = nn.Pad(padding, 'CONSTANT')
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, has_bias=self.use_bias,pad_mode='valid')

    def construct(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x).astype("float32"))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Student2(nn.Cell):
    def __init__(self, intc=3,outc =3):
        super(Student2, self).__init__()
        dim = 64
        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'
        self.concat = mindspore.ops.Concat(1)
        self.conv0 = nn.SequentialCell(
            [Conv2dBlock(intc, dim, 3, 1, ((0,0),(0,0),(1,1),(1,1)), norm=norm, activation=activ, pad_type=pad_type)]
            # Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.down1 = nn.SequentialCell(
            [Conv2dBlock(dim, dim, 4, 2, ((0,0),(0,0),(1,1),(1,1)), norm=norm, activation=activ, pad_type=pad_type)]
            # Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.down2 = nn.SequentialCell(
            [Conv2dBlock(dim, dim, 4, 2, ((0,0),(0,0),(1,1),(1,1)), norm=norm, activation=activ, pad_type=pad_type)]
            # Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = nn.SequentialCell(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad',padding=1, stride=1)
        )

        self.res2 = nn.SequentialCell(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad',padding=1, stride=1)
        )

        self.res3 = nn.SequentialCell(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad',padding=1, stride=1)
        )

        self.res4 = nn.SequentialCell(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad',padding=1, stride=1)
        )

        self.res5 = nn.SequentialCell(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad',padding=1, stride=1)
        )

        self.res6 = nn.SequentialCell(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad',padding=1, stride=1)
        )

        self.up1 = nn.SequentialCell(
            # nn.Upsample(scale_factor=2),
            [Conv2dBlock(dim * 2, dim, 3, 1, ((0,0),(0,0),(1,1),(1,1)), norm='none', activation=activ, pad_type=pad_type)]
        )

        self.up2 = nn.SequentialCell(
            # nn.Upsample(scale_factor=2),
            [Conv2dBlock(dim * 2, dim, 3, 1, ((0,0),(0,0),(1,1),(1,1)), norm='none', activation=activ, pad_type=pad_type)]
        )

        self.out = nn.SequentialCell(
            # Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
            [Conv2dBlock(dim, outc, 3, 1, ((0,0),(0,0),(1,1),(1,1)), norm='none', activation='tanh', pad_type=pad_type)]
        )
        self.ResizeBilinear = mindspore.nn.ResizeBilinear()
        self.test = mindspore.ops.Concat(0)

    def construct(self, x):
        x0 = self.conv0(x)
        x_d1 = self.down1(x0)
        x_d2 = self.down2(x_d1)
        x1 = x_d2 + self.res1(x_d2)
        x2 = x1 + self.res2(x1)
        x3 = x2 + self.res3(x2)
        x4 = x3 + self.res4(x3)
        x5 = x4 + self.res5(x4)
        x6 = x5 + self.res6(x5)
        x_u1 = self.concat((self.ResizeBilinear(x6, scale_factor=2), x_d1))
        #x_try = self.test((x_u1, x_d1))
        x_u1 = self.up1(x_u1)
        x_u2 = self.up2(self.concat((self.ResizeBilinear(x_u1, scale_factor=2), x0)))
        out = self.out(x_u2)

        return out, [x1, x2, x3, x4, x5, x6]

