from collections import Counter, defaultdict
from args import IncrementalConfig
from ds.incremental import DummyDataset
import factory
import numpy as np
import utils
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

def build_exemplar_broadcast(cfg: IncrementalConfig, model, inc_dataset, memory_per_class: list): 
    sync = []
    if cfg.coreset_strategy == "disable":
        inc_dataset.data_memory = inc_dataset.data_inc[[]]
        inc_dataset.targets_memory = inc_dataset.targets_inc[[]]
        return []
    # if utils.is_main_process():  # 不进行分布式  Soap
    from rehearsal.selection import d2
    dataset = inc_dataset.get_custom_dataset("train", "test") 
    train_loader = factory.create_dataloader(cfg, dataset, False, False)

    if cfg.coreset_strategy == "iCaRL":
        idx = d2(model,
                 cfg.gpu, 
                 train_loader,
                 cfg.nb_seen_classes,
                 cfg.nb_task_classes,
                 memory_per_class,
                 cfg)
    elif cfg.coreset_strategy == "keepall":
        idx = np.arange(len(inc_dataset.data_inc))
    elif cfg.coreset_strategy == "disable":
        idx = []
    sync.append(Tensor.from_numpy(idx))
    # else:  # 不进行分布式  Soap
    #     sync.append(None)  # 不进行分布式  Soap
    # dist.barrier()  # 不进行分布式  Soap
    # dist.broadcast_object_list(sync, 0)  # 不进行分布式  Soap
    idx = sync[0].numpy()
    # dist.barrier()  # 不进行分布式  Soap
    inc_dataset.data_memory = inc_dataset.data_inc[idx]
    inc_dataset.targets_memory = inc_dataset.targets_inc[idx]
    return idx


def collate_result(fn, data_loader, cfg):
    result = defaultdict(list)
    targets = []

    for i, (inputs, lbls) in enumerate(data_loader):
        r = fn(inputs)
        r['logit'] = ops.stop_gradient(r['logit'])
        r['feature'] = ops.stop_gradient(r['feature'])
        r['aux_logit'] = ops.stop_gradient(r['aux_logit'])
        for k, v in r.items():
            if v is not None:
                result[k].append(v) 
        targets.append(lbls.long())
        if cfg.debug:
            break

    for k, v in result.items():
        result[k] = mindspore.ops.cat(v)
    targets = mindspore.ops.cat(targets)
    return result, targets