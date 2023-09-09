"""
 @Time    : 22/9/2
 @Author  : WangSen
 @Email   : wangsen@shu.edu.cn
 
 @Project : FDLNet
 @File    : fdlnet_deeplab.py
 @Function: FDLNet 
 
"""

from mindspore import nn, ops, load_checkpoint, load_param_into_net
import mindspore
from .segbase import SegBaseModel
from .fcn import _FCNHead
from .frelayer import LFE

from ..nn import _ConvBNReLU
__all__ = ['FDLNet', 'get_fdlnet', 'get_fdlnet_resnet101_citys']

class FDLNet(SegBaseModel):
    def __init__(self, nclass, criterion=None, backbone='resnet50', aux=False, pretrained_base=False, **kwargs):
        super(FDLNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.criterion = criterion
        self.fcm = _FDLHead(2048, 2048, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None)

        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['fcm', 'auxlayer'] if aux else ['fcm'])

    def construct(self, x, gts=None, segSize=None):
        size = x.shape[2:]
        outputs = []
        c1, c2, c3, c4 = self.base_construct(x)
        fcm= self.fcm(c4, c1)
        seg_out_final = ops.interpolate(fcm, size, mode='bilinear', align_corners=True)
        
        outputs.append(seg_out_final)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = ops.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        if self.training:
            return self.criterion(outputs, gts)
        else:
            return tuple(outputs)

class _FDLHead(nn.Cell):
    def __init__(self, in_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FDLHead, self).__init__()
        c1_channels = 256
        self.att = LFE(in_channels, dct_h=8, dct_w=8, frenum=8)
        self.ppm = _DeepLabHead(c1_channels=256, out_channels=512, **kwargs)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.fam = _SFFHead(in_channels=2048, inter_channels=512, **kwargs)

        self.final_seg = nn.SequentialCell(
            _ConvBNReLU(512+48, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(p=0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, nclass, 1, has_bias=True))

    def construct(self, x, c1):
        fre = self.att(x) #B 2048 1 1

        f = self.ppm(x)
        fa = self.fam(f, fre)

        size = c1.shape[2:]
        c1 = self.c1_block(c1)
        fa = ops.interpolate(fa, size, mode='bilinear', align_corners=True)

        seg_out = self.final_seg(ops.cat([fa, c1], axis=1))

        return seg_out


class SFF(nn.Cell):
    """ spatial frequency fusion module"""

    def __init__(self, in_channels, **kwargs):
        super(SFF, self).__init__()
        self.alpha = mindspore.Parameter(ops.zeros(1))
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, fre):
        batch_size, _, height, width = x.shape
        fre = fre.expand_as(x)
        feat_a = x.view(batch_size, -1, height * width) #B C H*W
        feat_f_transpose = fre.view(batch_size, -1, height * width).permute(0, 2, 1) #B H*W C
        attention = ops.bmm(feat_a, feat_f_transpose)  # B C C
        attention_new = ops.max(attention, axis=-1, keepdims=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new) # B C C

        feat_e = ops.bmm(attention, feat_a).view(batch_size, -1, height, width) # B C H*W
        out = self.alpha * feat_e + x
        return out


class _SFFHead(nn.Cell):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_SFFHead, self).__init__()
        self.conv_x1 = nn.SequentialCell(
            nn.Conv2d(inter_channels, inter_channels, 1, pad_mode='pad', has_bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )
        self.conv_f1 = nn.SequentialCell(
            nn.Conv2d(in_channels, inter_channels, 1, pad_mode='pad', has_bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )
        self.freatt = SFF(inter_channels, **kwargs)
        self.conv_p2 = nn.SequentialCell(
            nn.Conv2d(inter_channels, inter_channels, 1, pad_mode='pad', has_bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(),
        )

    def construct(self, x, fre):
        feat_x = self.conv_x1(x)
        feat_f = self.conv_f1(fre)

        feat_p = self.freatt(feat_x, feat_f)
        feat_p = self.conv_p2(feat_p)

        return feat_p

class _DeepLabHead(nn.Cell):
    def __init__(self, c1_channels=256, out_channels=512, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        
        self.block = nn.SequentialCell(
            _ConvBNReLU(256, out_channels, 3, padding=1, norm_layer=norm_layer),
            )

    def construct(self, x):
       
        x = self.aspp(x)
        
       
        return self.block(x)

class _ASPPConv(nn.Cell):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, 3, pad_mode='pad', padding=atrous_rate, dilation=atrous_rate, has_bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )

    def construct(self, x):
        return self.block(x)


class _AsppPooling(nn.Cell):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, pad_mode='pad', has_bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )

    def construct(self, x):
        size = x.shape[2:]
        pool = self.gap(x)
        out = ops.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Cell):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, 1, pad_mode='pad', has_bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.SequentialCell(
            nn.Conv2d(5 * out_channels, out_channels, 1, pad_mode='pad', has_bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def construct(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = ops.cat((feat1, feat2, feat3, feat4, feat5), axis=1)
        x = self.project(x)
        return x

def get_fdlnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='./',
            pretrained_base=True, **kwargs):

    from ..data.dataloader import datasets
    model = FDLNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        checkpoint = load_checkpoint(get_model_file('best_mindspore_fdlnet', root=root))
        load_param_into_net(model, checkpoint)
        # model.load_state_dict(checkpoint['state_dict'])
    return model



def get_fdlnet_resnet101_citys(**kwargs):
    return get_fdlnet('citys', 'resnet101', **kwargs)



if __name__ == '__main__':
    model = get_fdlnet_resnet101_citys()
    img = ops.randn(4, 3, 480, 480)
    output = model(img)
