"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .night_edge import NightEdgeSegmentation


datasets = {
    'night': NightEdgeSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
