import mindspore
import mindspore.nn as nn
from typing import List, Text


def switch_grad(m, requires_grad):
    for p in m:
        p.requires_grad = requires_grad


def remove_component_from_state_dict(state_dict, components: List[str], is_ddp=False): 
    state_dict_copy = state_dict.copy()
    prefix = "module." if is_ddp else "" 
    components = tuple(map(lambda x: f"{prefix}{x}", components)) 
    for k in state_dict.keys():
        if k.startswith(components):
            del state_dict_copy[k]
    return state_dict_copy