# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""sampler"""
from mindspore.dataset import RandomSampler, SequentialSampler

__all__ = ['make_data_sampler']


def make_data_sampler(shuffle, max_num=None):
    """sampler"""
    if shuffle:
        sampler = RandomSampler()
    else:
        sampler = SequentialSampler()
    return sampler



if __name__ == '__main__':
    pass
