"""benchmark"""
import os
import glob
from data.srdata import SRData


class Benchmark(SRData):
    """DIV2K"""
    def __init__(self, args, name='Set5', train=False):
        super(Benchmark, self).__init__(args, name=name, train=train)
        self.dir_hr = None
        self.dir_lr = None

    def _scan(self):
        """srdata"""
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))#glob 查找文件路径，将所有HR文件的图像名进行排序
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))#basename - 返回最后一部分
            for si, s in enumerate(self.scale): #enumberate 构成索引序列 0 - a
                if s != 1:
                    scale = s
                    names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}{}' \
                        .format(s, filename, self.ext[1])))#文件地址 X2/1x2.img
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si] = names_hr
        return names_hr, names_lr


    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test[0])
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('.png', '.png')