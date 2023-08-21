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
"""Model store which provides pretrained models."""
from __future__ import print_function

import os

__all__ = ['get_model_file', 'get_resnet_file']


def get_resnet_file(name, root='~/.torch/models'):
    file_name = '{name}'.format(name=name)
    root = '/dataset/NightCity-images'
    root = os.path.expanduser(root)

    file_path = os.path.join(root, file_name + '.ckpt')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file {} is not found. Downloading.'.format(file_path))


def get_model_file(name, root='~/.torch/models'):
    root = 'C:\\Users\\12098\\Desktop\\ecnu_summerwork'
    root = os.path.expanduser(root)
    file_path = os.path.join(root, name + '.ckpt')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')
