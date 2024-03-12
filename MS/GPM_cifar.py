import errno
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import value_and_grad


import os
import os.path
from collections import OrderedDict

import numpy as np
import random
import argparse, time
from copy import deepcopy


import os.path as osp


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def space_intersection(space1, space2, ths=2.5e-2):

    A = np.concatenate((space1, -space2), axis=1)
    A[A == np.inf] = 0.
    A[A == -np.inf] = 0.

    while True:
        try:
            u, s, vt = np.linalg.svd(A)
            break
        except:
            print('?')
            A = A[:, :A.shape[1]-1]
            continue

    null_space = np.compress(s <= ths, vt, axis=0).T
    rr = space1.shape[1]
    basis = np.dot(space1, null_space[:rr, :])

    return basis


def svd(space, pre_space=None, ths=0.985, minus=False):
    space[space == np.inf] = 0.
    space[space == -np.inf] = 0.
    if pre_space is not None:
        pre_space[pre_space == np.inf] = 0.
        pre_space[pre_space == -np.inf] = 0.
    if pre_space is None or pre_space.shape[1] == 0:
        while True:
            try:
                U, S, Vh = np.linalg.svd(space, full_matrices=False)
                break
            except:
                print('?')
                space = space[:, :space.shape[1]-1]
                continue
        sval_total = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
        r = np.sum(np.cumsum(sval_ratio) < ths)
        return U[:, 0:r]
    else:
        activation = space
        while True:
            try:
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                break
            except:
                print('?')
                activation = activation[:, :activation.shape[1]-1]
                continue
        sval_total = (S1 ** 2).sum()
        act_hat = activation - np.dot(np.dot(pre_space, pre_space.transpose()), activation)
        while True:
            try:
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                break
            except:
                print('?')
                act_hat = act_hat[:, :act_hat.shape[1]-1]
                continue
        sval_hat = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
        accumulated_sval = (sval_total - sval_hat) / sval_total if not np.equal(sval_total, 0) else (sval_total - sval_hat)
        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < ths:
                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        if minus:
            return U[:, 0:r]
        Ui = np.hstack((pre_space, U[:, 0:r]))
        if Ui.shape[1] > Ui.shape[0]:
            tmp_pre_space = Ui[:, 0:Ui.shape[0]]
        else:
            tmp_pre_space = Ui
        return tmp_pre_space


class SpaceUnion:
    def __init__(self):
        self.space = []
        self.union = []
        self.args = None
        self.I = []
        self.R = []

    def cal_score(self, gradient):
        res_i = []
        res_r = []
        dims_i = []
        dims_r = []
        max_layer = len(self.space[0])
        for i in range(max_layer):
            res_i.append(0)
            res_r.append(0)
            dims_i.append(self.I[i].shape[1])
            dims_r.append(self.R[i].shape[1])
            g = gradient[i]
            s_i = ms.tensor(np.dot(self.I[i], self.I[i].transpose()), dtype=ms.float32)
            s_r = ms.tensor(np.dot(self.R[i], self.R[i].transpose()), dtype=ms.float32)
            g_i = ms.ops.mm(g, s_i)
            g_r = ms.ops.mm(g, s_r)
            for j in range(len(self.space)):
                layer = i
                task_space = self.space[j][layer]
                task_space = ms.tensor(np.dot(task_space, task_space.transpose()), dtype=ms.float32)
                task_gradient_i = ms.ops.mm(g_i, task_space)
                task_gradient_r = ms.ops.mm(g_r, task_space)
                res_i[i] += ms.ops.sum(ms.ops.norm(task_gradient_i, dtype=ms.float32) ** 2).item()
                res_r[i] += ms.ops.sum(ms.ops.norm(task_gradient_r, dtype=ms.float32) ** 2).item()

            res_i[i] = res_i[i] / dims_i[i] if dims_i[i] > 0 else 0
            res_r[i] = res_r[i] / dims_r[i] if dims_r[i] > 0 else 0

        return res_i, res_r

    def update(self, mat_list, threshold, M, M_after):
        tmp = []
        for i in range(len(mat_list)):
            activation = mat_list[i]
            while True:
                try:
                    U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                    break
                except:
                    print('?')
                    activation = activation[:, :activation.shape[1]-1]
                    continue
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            tmp.append(U[:, 0:r])

        if len(self.I) == 0:
            for i in range(len(mat_list)):
                self.I.append([])
                self.R.append([])

        if len(self.space) > 0:
            for i in range(len(self.space[0])):
                activation = mat_list[i]
                while True:
                    try:
                        U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                        break
                    except:
                        print('?')
                        activation = activation[:, :activation.shape[1]-1]
                        continue
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
                r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
                
                r9_th = 0.94
                if i == 0:
                    r9_th = 0.985
                r9 = np.sum(np.cumsum(sval_ratio) < r9_th)

                A = np.concatenate((M[i], -U[:, 0:r]), axis=1)

                while True:
                    try:
                        u, s, vt = np.linalg.svd(A)
                        break
                    except:
                        print('?')
                        A = A[:, :A.shape[1]-1]
                        continue
                
                if A.shape[0] < A.shape[1]:
                    s = np.concatenate((s, np.array([0] * (A.shape[1] - A.shape[0]))), axis=0)

                null_space = np.compress(s <= 2.5e-2, vt, axis=0).T

                rr = M[i].shape[1]
                basis = np.dot(M[i], null_space[:rr, :])

                i_ths = 0.99
                if len(self.I[i]) == 0:
                    self.I[i] = svd(basis, ths=i_ths)
                else:
                    self.I[i] = svd(np.concatenate((basis, self.I[i]), axis=1), ths=i_ths)

                basis = np.concatenate((basis, U[:, :r9]), axis=1)

                if len(self.union) > i:

                    Ui = np.hstack((self.union[i], basis))
                    if Ui.shape[1] > Ui.shape[0]:
                        self.union[i] = Ui[:, 0:Ui.shape[0]]
                    else:
                        self.union[i] = Ui
                    while True:
                        try:
                            U, S, Vh = np.linalg.svd(self.union[i], full_matrices=False)
                            break
                        except:
                            print('?')
                            self.union[i] = self.union[i][:, :self.union[i].shape[1]-1]
                            continue
                    sval_total = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
                    r = np.sum(np.cumsum(sval_ratio) < 0.999)
                    self.union[i] = U[:, 0:r]
                else:
                    while True:
                        try:
                            U, S, Vh = np.linalg.svd(basis, full_matrices=False)
                            break
                        except:
                            print('?')
                            basis = basis[:, :basis.shape[1]-1]
                            continue
                    sval_total = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
                    r = np.sum(np.cumsum(sval_ratio) < 0.999)
                    basis = U[:, 0:r]
                    self.union.append(basis)

        self.space.append(tmp)

        if len(self.I[0]) > 0:
            for i in range(len(self.union)):
                self.R[i] = self.union[i] - np.dot(np.dot(
                    self.I[i], self.I[i].transpose()
                ), self.union[i])
                self.R[i] = svd(self.R[i], ths=0.999)


space_union = SpaceUnion()


class AlexNet(nn.Cell):
    def __init__(self, taskcla):
        super(AlexNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, has_bias=False, pad_mode='valid')
        self.bn1 = nn.BatchNorm2d(64, use_batch_statistics=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, has_bias=False, pad_mode='valid')
        self.bn2 = nn.BatchNorm2d(128, use_batch_statistics=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, has_bias=False, pad_mode='valid')
        self.bn3 = nn.BatchNorm2d(256, use_batch_statistics=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Dense(256 * self.smid * self.smid, 2048, has_bias=False)
        self.bn4 = nn.BatchNorm1d(2048, use_batch_statistics=False)
        self.fc2 = nn.Dense(2048, 2048, has_bias=False)
        self.bn5 = nn.BatchNorm1d(2048, use_batch_statistics=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3 = nn.CellList()
        for t, n in self.taskcla:
            self.fc3.append(nn.Dense(2048, n, has_bias=False))


    def construct(self, x):
        bsz = deepcopy(x.shape[0])
        self.act['conv1'] = x.copy()
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2'] = x.copy()
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3'] = x.copy()
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        self.act['fc1'] = x.copy()
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2'] = x.copy()
        x = self.fc2(x)
        
        x = self.drop2(self.relu(self.bn5(x)))
        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y

    def last_fea(self, x):
        bsz = deepcopy(x.shape[0])
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x)
        self.relu(self.bn5(x))
        return x
        
    def logits(self, x):
        x = self.drop2(x)
        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y


def get_model(model):
    ms.save_checkpoint(model, "best.ckpt")
    return "best.ckpt"


def set_model_(model, state_dict="best.ckpt"):
    param_dict = ms.load_checkpoint(state_dict)
    param_not_load, _ = ms.load_param_into_net(model, param_dict)
    print("Fail to load: ", param_not_load)


def adjust_learning_rate(optimizer, epoch, args):
    if (epoch == 1):
        ms.ops.assign(optimizer.learning_rate, args.lr)        
    else:
        ms.ops.assign(optimizer.learning_rate, optimizer.learning_rate / args.lr_factor)


class Trainner:
    def __init__(self, model, loss, opt, task_id):
        self.model = model
        self.loss = loss
        self.opt = opt
        self.task_id = task_id
        self.value_and_grad = value_and_grad(self.forward, None, self.opt.parameters, has_aux=False)
        
    def forward(self, input, target):
        output = self.model(input)[self.task_id]
        loss = self.loss(output, target)
        return loss
    
    def train_one_step(self, input, target):
        (loss), grad = self.value_and_grad(input, target)
        self.opt(grad)
        return loss, grad


def train(args, model:nn.Cell, x, y, optimizer:nn.Optimizer, criterion, task_id):
    r = np.arange(x.shape[0])
    np.random.shuffle(r)
    r = ms.tensor(r).long()
    # Loop batches
    trainner = Trainner(model, criterion, optimizer, task_id)
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i:i + args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data, y[b]
        trainner.train_one_step(data, target)
        

def train_projected(args, model:nn.Cell, x, y, optimizer, criterion, feature_mat, task_id):
    r = np.arange(x.shape[0])
    np.random.shuffle(r)
    r = ms.tensor(r).long()

    if task_id >= 2:

        unions = []

        gamma1 = 0.00005 * (0.9 ** task_id)
        gamma2 = 0.000001 * (0.9 ** task_id)

        for l in range(len(space_union.I)):
            tmp = space_union.I[l]
            # print('?? tmp: ', tmp.shape)
            unions.append((1. - gamma1) * np.dot(tmp, tmp.transpose()))
            if len(space_union.R) > 0 and space_union.R[l].shape[1] > 0:
                tmp = space_union.R[l]
                unions[l] = unions[l] + (1. - gamma2) * np.dot(tmp, tmp.transpose())
            unions[l] = ms.tensor(unions[l], dtype=ms.float32)

    # Loop batches
    trainner = Trainner(model, criterion, optimizer, task_id)
    for i in range(0, len(r), args.batch_size_train):
        if i + args.batch_size_train <= len(r):
            b = r[i:i + args.batch_size_train]
        else:
            b = r[i:]
        data = x[b]
        data, target = data, y[b]
        loss, grad = trainner.value_and_grad(data, target)

        kk = 0

        new_grad = []
        for k, (params) in enumerate(grad):
            if (k < 15) and (len(params.shape) != 1):
                sz = params.shape[0]  
                if kk >= 1 and task_id >= 2:
                    tmp = ms.ops.mm(params.view(sz, -1), unions[kk]).view(params.shape)
                    params = params - tmp
                else:
                    tmp = ms.ops.mm(params.view(sz, -1), feature_mat[kk]).view(params.shape)
                    params = params - tmp
                kk += 1
            elif (k < 15) and (len(params.shape) == 1) and (task_id != 0):
                params.numpy().fill(0)
            new_grad.append(ms.tensor(params, dtype=ms.float32))
        del grad
        new_grad = tuple(new_grad)
        optimizer(new_grad)


def test(args, model, x, y, criterion, task_id):
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.shape[0])
    np.random.shuffle(r)
    r = ms.tensor(r).long()

    transform = [
        ds.transforms.Compose([
            ds.transforms.vision.RandomHorizontalFlip(prob=0.),
        ]),
        ds.transforms.Compose([
            ds.transforms.vision.RandomHorizontalFlip(prob=1.),
        ])
    ]

    # Loop batches
    for i in range(0, len(r), args.batch_size_test):
        if i + args.batch_size_test <= len(r):
            b = r[i:i + args.batch_size_test]
        else:
            b = r[i:]
        data = x[b]
        data, target = data, y[b]

        output = None
        for j in range(len(transform)):
            tmp_in = ms.tensor(transform[j](data.numpy()))
            tmp_out = model(tmp_in)
            if output is None:
                output = tmp_out
            else:
                output = [x + y for (x, y) in zip(output, tmp_out)]
        output = [x / len(transform) for x in output]
        
        loss = criterion(output[task_id], target)
        pred = output[task_id].argmax(axis=1, keepdims=True)

        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.numpy().item() * len(b)
        total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_representation_matrix(net, x, y=None):
    # Collect activations by forward pass
    r = np.arange(x.shape[0])
    np.random.shuffle(r)
    r = ms.tensor(r).long()
    b = r[0:125]  # Take 125 random samples
    example_data = x[b]
    example_data = example_data
    example_out = net(example_data)

    batch_list = [2 * 12, 100, 100, 125, 125]
    mat_list = []
    act_key = list(net.act.keys())
    for i in range(len(net.map)):
        bsz = batch_list[i]
        k = 0
        if i < 3:
            ksz = net.ksize[i]
            s = compute_conv_output_size(net.map[i], net.ksize[i])
            mat = np.zeros((net.ksize[i] * net.ksize[i] * net.in_channel[i], s * s * bsz))
            act = net.act[act_key[i]].copy().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:, k] = act[kk, :, ii:ksz + ii, jj:ksz + jj].reshape(-1)
                        k += 1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].copy().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    print('-' * 30)
    print('Representation Matrix')
    print('-' * 30)
    for i in range(len(mat_list)):
        print('Layer {} : {}'.format(i + 1, mat_list[i].shape))
    print('-' * 30)
    return mat_list


def update_GPM(model, mat_list, threshold, feature_list=[], ):
    print('Threshold: ', threshold)
    M_before = deepcopy(feature_list)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total if not np.equal(sval_total, 0) else (S ** 2)
            accumulated_sval = (sval_total - sval_hat) / sval_total if not np.equal(sval_total, 0) else (sval_total - sval_hat) 

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print('Skip Updating GPM for layer: {}'.format(i + 1))
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0:Ui.shape[0]]
            else:
                feature_list[i] = Ui

    print('-' * 40)
    print('Gradient Constraints Summary')
    print('-' * 40)
    for i in range(len(feature_list)):
        print('Layer {} : {}/{}'.format(i + 1, feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-' * 40)

    space_union.update(mat_list, threshold, M_before, M_after=feature_list)

    return feature_list


def froze_grad(model:AlexNet, task_id:int):
    for label in range(len(model.fc3)):
        if label==task_id:
            model.fc3[label].weight.requires_grad = True
        else:
            model.fc3[label].weight.requires_grad = False


def main(args):
    tstart = time.time()
    ## Device Setting
    ms.set_context(device_target='GPU', device_id=0, deterministic='ON')
    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    ## Load CIFAR100 DATASET
    if args.inc_task == 10:
        from dataloader import cifar100_ms as cf100
        acc_mat_size = 10
    else:
        from dataloader import cifar100_20 as cf100
        acc_mat_size = 20
    data, taskcla, inputsize = cf100.get(seed=args.seed, pc_valid=args.pc_valid)
    
    
    acc_matrix = np.zeros((acc_mat_size, acc_mat_size))
    criterion = nn.CrossEntropyLoss()

    task_id = 0
    task_list = []

    model = AlexNet(taskcla)
    if args.resume and os.path.exists("best.ckpt"):
        set_model_(model)

    for k, ncla in taskcla:
        # specify threshold hyperparameter
        if args.inc_task == 10:
            threshold = np.array([0.97] * 5) + task_id * np.array([0.003] * 5)
        else:
            threshold = np.array([0.97] * 5) + task_id * np.array([0.0015] * 5)

        print('*' * 100)
        print('Task {:2d} ({:s})'.format(k, data[k]['name']))
        print('*' * 100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)
        
        print('xtrain: ', xtrain.shape)
        print('ytrain: ', min(ytrain), ' ', max(ytrain))

        lr = args.lr
        best_loss = np.inf
        print('-' * 40)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print('-' * 40)

        froze_grad(model, task_id)
        print('Model parameters ---')
        for param in enumerate(model.get_parameters()):
            print(param)
        print('-' * 40)

        if task_id == 0:
            feature_list = []
            best_model = get_model(model)
            optimizer = nn.optim.SGD(model.trainable_params(),
                                     learning_rate=lr)
            for epoch in range(1, args.n_epochs + 1):
                # Train
                clock0 = time.time()
                train(args, model, xtrain, ytrain, optimizer, criterion, k)
                clock1 = time.time()
                tr_loss, tr_acc = test(args, model, xtrain, ytrain, criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, \
                                                                                                 tr_loss, tr_acc,
                                                                                                 1000 * (
                                                                                                         clock1 - clock0)),
                      end='')
                # Validate
                valid_loss, valid_acc = test(args, model, xvalid, yvalid, criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < args.lr_min:
                            print()
                            break
                        patience = args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model, best_model)
            # Test
            print('-' * 40)
            test_loss, test_acc = test(args, model, xtest, ytest, criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss, test_acc))
            # Memory Update
            mat_list = get_representation_matrix(model, xtrain, ytrain)
            feature_list = update_GPM(model, mat_list, threshold, feature_list)

        else: 
            optimizer = nn.optim.SGD(model.trainable_params(), 
                                     learning_rate=lr)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf = ms.tensor(np.dot(feature_list[i], feature_list[i].transpose())).float()
                Uf.requires_grad = False
                print('Layer {} - Projection Matrix shape: {}'.format(i + 1, Uf.shape))
                feature_mat.append(Uf)
            print('-' * 40)
            
            # cal score
            if task_id >= 2:
                print('start to cal score')
                r = np.arange(xtrain.shape[0])
                np.random.shuffle(r)
                r = ms.tensor(r).long()
                x_score = data[8]['train']['x']
                y_score = data[8]['train']['y']

                trainer = Trainner(model, criterion, optimizer, task_id)
                loss, grad = trainer.value_and_grad(x_score, y_score)
                output_score = model(x_score)
                
                gradient = []
                for kc, params in enumerate(grad):
                    if kc < 15 and len(params.shape) != 1:
                        sz = params.shape[0]
                        gradient.append(params.view(sz, -1))
                    grad[kc].numpy().fill(0)
                res_i, res_r = space_union.cal_score(gradient)
                print(res_i, res_r)

            for epoch in range(1, args.n_epochs + 1):
                # Train
                clock0 = time.time()
                train_projected(args, model, xtrain, ytrain, optimizer, criterion, feature_mat, task_id,)
                clock1 = time.time()
                tr_loss, tr_acc = test(args, model, xtrain, ytrain, criterion, task_id)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, \
                                                                                                 tr_loss, tr_acc,
                                                                                                 1000 * (
                                                                                                         clock1 - clock0)),
                      end='')
                # Validate
                valid_loss, valid_acc = test(args, model, xvalid, yvalid, criterion, task_id)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
                # Adapt lr
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = get_model(model)
                    patience = args.lr_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= args.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < args.lr_min:
                            print()
                            break
                        patience = args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
                
            set_model_(model, best_model)
            # Test
            test_loss, test_acc = test(args, model, xtest, ytest, criterion, task_id)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss, test_acc))
            # Memory Update
            mat_list = get_representation_matrix(model, xtrain, ytrain)
            feature_list = update_GPM(model, mat_list, threshold, feature_list)

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id + 1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(args, model, xtest, ytest, criterion, ii)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_id + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
            print()
        # update task id
        task_id += 1
    print('-' * 50)
    # Simulation Results
    print('Task Order : {}'.format(np.array(task_list)))
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time() - tstart) * 1000))
    print('-' * 50)
    
    
def seed_ms(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


if __name__ == "__main__":

    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--pc_valid', default=0.05, type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--inc_task', type=int, default=10)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    
    seed_ms(args.seed)

    space_union.args = args

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    main(args)


