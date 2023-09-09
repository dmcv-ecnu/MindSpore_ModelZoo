"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import math
import pickle
from mindspore.dataset import RandomSampler, SequentialSampler

__all__ = [ 'make_data_sampler']





def make_data_sampler(shuffle, max_num = None):
    if shuffle:
        sampler = RandomSampler()
        # sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = SequentialSampler()
        # sampler = data.sampler.SequentialSampler(dataset)
    return sampler



if __name__ == '__main__':
    pass
