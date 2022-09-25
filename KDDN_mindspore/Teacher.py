import mindspore
import mindspore.nn as nn

class Conv2dBlock(nn.Cell):
    def __init__(self, in_dim, out_dim, ks, st, padding=((0,0),(0,0),(0,0),(0,0)),
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=False, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first

        #initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

            # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.Pad(padding, 'REFLECT')
        elif pad_type == 'replicate':
            self.pad = nn.Pad(padding, 'CONSTANT')
        elif pad_type == 'zero':
            self.pad = nn.Pad(padding, 'CONSTANT')
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
            #self.pad = nn.Pad(padding, 'CONSTANT')

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
            #这里和原版相比少了就地操作的实现,但是好像师兄并没有用到这个就地操作
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.conv = nn.Conv2d(in_dim, out_dim, ks, st,has_bias=use_bias,pad_mode='valid')

    def construct(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x).astype("float32"))
            #x = self.conv(x)#.astype(mindspore.float32))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x).astype("float32"))
            #x = self.conv(x)#.astype(mindspore.float32))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class ResBlock(nn.Cell):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, ((0,0),(0,0),(1,1),(1,1)),
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, ((0,0),(0,0),(1,1),(1,1)),
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.SequentialCell(*model)

    def construct(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class endeFUINT2_1(nn.Cell):
    def __init__(self):
        super(endeFUINT2_1, self).__init__()
        dim = 64
        activ = 'relu'
        pad_type = 'reflect'#改过了
        norm = 'none'

        self.conv0 = Conv2dBlock(3, dim, 7, 1, ((0,0),(0,0),(3,3),(3,3)), norm=norm, activation=activ, pad_type=pad_type)
        self.down = nn.SequentialCell(
            Conv2dBlock(dim, dim, 4, 2, ((0,0),(0,0),(1,1),(1,1)), norm=norm, activation=activ, pad_type=pad_type),
            Conv2dBlock(dim, dim, 4, 2, ((0,0),(0,0),(1,1),(1,1)), norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res2 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res3 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res4 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res5 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res6 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)

        # nn.ResizeBilinear(scale_factor=2),
        self.up1 = nn.ResizeBilinear()
        self.up2 = Conv2dBlock(dim, dim, 5, 1, ((0,0),(0,0),(2,2),(2,2)),norm='none', activation=activ, pad_type=pad_type)
        #nn.ResizeBilinear(scale_factor = 2 ),
        #nn.ResizeBilinear(),
        #这里有个问题，mindspore只支持线性插值采样，和torch的默认值不同
        #Conv2dBlock(dim, dim, 5, 1, ((2,2),(2,2)), norm='none', activation=activ, pad_type=pad_type)
        self.out = Conv2dBlock(dim, 3, 7, 1, ((0,0),(0,0),(3,3),(3,3)), norm='none', activation='tanh', pad_type=pad_type)

    def construct(self, x):
        x0 = self.conv0(x)
        x_d = self.down(x0)
        x1 = self.res1(x_d)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        #x_u = self.up(x6)
        x_u = self.up1(x6,scale_factor=2)
        x_u = self.up2(x_u)
        x_u = self.up1(x_u, scale_factor=2)
        x_u = self.up2(x_u)

        out = self.out(x_u)

        return out, [x1, x2, x3, x4, x5, x6]