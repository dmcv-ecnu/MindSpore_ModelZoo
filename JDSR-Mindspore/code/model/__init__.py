"""
参照原版，这个文件还没有实现，待mindspore的特性判断是否需要保留
"""


from mindspore import nn

class Model(nn.Cell):
    def __init__(self) -> None:
        super().__init__()

    def construct(self, *inputs, **kwargs):
        return super().construct(*inputs, **kwargs)
            