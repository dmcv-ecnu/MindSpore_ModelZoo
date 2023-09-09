"""Model store which handles pretrained models """
from .fdlnet_deeplab import *
# from .fdlnet_psp import * 

__all__ = ['get_segmentation_model']


def get_segmentation_model(model, **kwargs):
    models = {
        'fdlnet': get_fdlnet
    }
    return models[model](**kwargs)
