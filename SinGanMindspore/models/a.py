import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差

		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[	K_ss K_st
							K_ts K_tt ]
    """
    """
    pytorch代码：
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|
    """

    n_samples = int(source.shape[0]) + int(target.shape[0])
    concat_op = ops.Concat()
    total = concat_op((source, target))  # 合并在一起
    # print(total.size)

    expand_dims = ops.ExpandDims()
    shape = (int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    broadcast_to = ops.BroadcastTo(shape)
    total0 = broadcast_to(expand_dims(total, 0))
    total1 = broadcast_to(expand_dims(total, 1))
    op_sum = ops.ReduceSum()
    L2_distance = op_sum(((total0 - total1) ** 2), 2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = op_sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [ops.Exp()(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = ops.ReduceMean()(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
    # 当不同的时候，就需要乘上上面的M矩阵
    return loss


if __name__ == "__main__":
    import numpy as np

    data_1 = ms.Tensor(np.random.normal(0, 10, (2, 5))).astype(np.float64)
    data_2 = ms.Tensor(np.random.normal(10, 10, (2, 5))).astype(np.float64)

    print(data_1, data_2.shape[0])
    print("MMD Loss:", mmd(data_1, data_2))
    #
    # data_1 = ms.Tensor(np.random.normal(0, 10, (100, 50)))
    # data_2 = ms.Tensor(np.random.normal(0, 9, (100, 50)))
    #
    # print("MMD Loss:", mmd(data_1, data_2))
