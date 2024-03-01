from copy import deepcopy
from typing import *
import numpy as np
# from .metrics import accuracy

class TaskInfoMixin(object):
    def __init__(self):
        super(TaskInfoMixin, self).__init__()
        self._increments = []
        self._idx_task = -1
        self._nb_seen_classes = 0
        self._nb_task_classes = 0
    
    @property
    def _nb_tasks(self):
        return len(self._increments)

    def new_task(self, nb_task_classes):
        self._idx_task += 1
        self._nb_task_classes = nb_task_classes
        self._increments.append(nb_task_classes)
        self._nb_seen_classes += nb_task_classes


def target_to_task(targets, task_size:List[int]):
    targets = deepcopy(targets)
    prev = 0 
    for i, size in enumerate(task_size): 
        targets[(targets >= prev) & (targets < prev + size)] = i
        prev += size
    return targets


def decode_targets(targets: np.ndarray, increments: List[int], overlap: int):
    copy_y = deepcopy(targets) # non decoded.
    task = target_to_task(copy_y, increments)
    ut = np.unique(task)
    for t in ut: copy_y[task == t] -= t * overlap
    return copy_y