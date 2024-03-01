import argparse
import collections
import sys
from argparse import ArgumentParser
from typing import Dict, List
from easydict import EasyDict

from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml


class EasyConfig(argparse.Namespace):
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)


class SortedArgumentParser():
    """Maybe useless. Python3.6+ keeps the insertation order"""

    def __init__(self, *args, **kwargs):
        self.ap = ArgumentParser(*args, **kwargs)
        self.args_dict = collections.OrderedDict()

    def add_argument(self, *args, **kwargs):
        self.ap.add_argument(*args, **kwargs)
        # Also store dest kwarg
        if 'dest' in kwargs:
            self.args_dict[kwargs['dest'].strip('-')] = None
        else:
            self.args_dict[args[0].strip('-')] = None

    def __getattr__(self, name):
        if name != "add_argument":
            return getattr(self.ap, name)
        else:
            return self.add_argument

    def parse_args(self):
        # Returns a sorted dictionary
        unsorted_dict = self.ap.parse_args().__dict__
        for unsorted_entry in unsorted_dict:
            self.args_dict[unsorted_entry] = unsorted_dict[unsorted_entry]
        return self.args_dict


def add_argument_using_dict(parser: ArgumentParser, d: Dict, prefix=""):
    def preprocess(s: str):
        return "--" + prefix + s.replace("_", "-")

    for k, v in d.items():
        k = preprocess(k)
        if isinstance(v, list):
            parser.add_argument(k, nargs='+', default=v, type=type(v[0]))
        elif isinstance(v, dict):
            add_argument_using_dict(parser, v, f"{k[2:]}.")
        elif isinstance(v, bool):
            parser.add_argument(k, action="store_true", default=v)
            parser.add_argument(f"--no-{k[2:]}", action="store_false", default=v, dest=k[2:].replace("-", "_"))
        elif v is None:
            parser.add_argument(k, default=v)
        else:
            parser.add_argument(k, default=v, type=type(v))


def merge_args(src, dst):
    for k, v in src.items():
        if "." in k:
            attr_list = k.split(".")
            a0 = attr_list[0]
            pt = dst[a0] = EasyDict(dst[a0])
            for a in attr_list[1:-1]:
                pt = pt[a]
            pt[attr_list[-1]] = v
        else:
            dst[k] = v


KeyType = TypeVar('KeyType')


def deep_update(mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def check_valid_args(args: List[str]):
    for a in args:
        if a.startswith("--") and '_' in a:
            raise Exception(f"{a} argument may be wrong.")