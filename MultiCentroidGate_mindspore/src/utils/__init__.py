from .args_utils import *
from .distributed import *  # 分布式先放着  Soap
from .incremental_utils import *
from .results_utils import *
from .metrics import *
from .model_utils import *

import random
import numpy as np
import mindspore


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    mindspore.set_seed(seed + rank)