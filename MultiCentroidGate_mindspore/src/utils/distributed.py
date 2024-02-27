from mindspore.communication import get_rank as ms_get_rank
from mindspore import context

def get_rank():
    return ms_get_rank()


def is_main_process(): 
    return get_rank() == 0


def init_distributed_mode(args):
    args.world_size = 1
    args.rank = 0
    args.gpu = 0

    args.distributed = False
    args.dist_backend = 'nccl'

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(save_graphs=False)
    context.set_context(enable_graph_kernel=True)
    context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    context.set_context(device_id=0)