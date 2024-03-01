import numpy as np
from typing import Iterable
from time import time
import mindspore
from mindspore.ops import stop_gradient


def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))
    w_t = mu
    iter_herding, iter_herding_eff = 0, 0
    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1
        w_t = w_t + mu - D[:, ind_max]
    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000
    return herding_matrix.argsort()[:nb_examplars]


def d2(model, device, sel_loader, n_classes, task_size, memory_per_class: list, cfg):   
    features, targets = [], []
    for _x, _y in sel_loader:
        f = model(_x)
        f = stop_gradient(f)
        features.append(f)
        targets.append(_y)
        if cfg.debug:
            break

    features = mindspore.ops.cat(features, 0).numpy()
    targets = mindspore.ops.cat(targets, 0).numpy()
    idx = [] 
    for class_idx in range(n_classes):
        c_d = np.where(targets == class_idx)[0]
        if class_idx >= n_classes - task_size:
            feat = features[targets == class_idx]
            herding_matrix = icarl_selection(feat, memory_per_class[class_idx])
            idx.append(c_d[herding_matrix])
        else:
            idx.append(c_d[np.arange(memory_per_class[class_idx])])
    return np.concatenate(idx, 0)