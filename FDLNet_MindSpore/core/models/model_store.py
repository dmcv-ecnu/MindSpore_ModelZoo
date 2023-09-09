"""Model store which provides pretrained models."""
from __future__ import print_function

import os
import zipfile


__all__ = ['get_model_file', 'get_resnet_file']

_model_sha1 = {name: checksum for checksum, name in [
    ('25c4b50959ef024fcc050213a06b614899f94b3d', 'resnet50'),
    ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    ('0d43d698c66aceaa2bc0309f55efdd7ff4b143af', 'resnet152'),
]}

encoding_repo_url = 'https://hangzh.s3.amazonaws.com/'
_url_format = '{repo_url}encoding/models/{file_name}.zip'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_resnet_file(name, root='~/.torch/models'):
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = '/tmp/dataset/NightCity-images'
    root = os.path.expanduser(root)

    file_path = os.path.join(root, file_name + '.pth')
    sha1_hash = _model_sha1[name]
    return file_path


def get_model_file(name, root='~/.torch/models'):
    root = '/code'
    root = os.path.expanduser(root)
    file_path = os.path.join(root, name + '.ckpt')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')
