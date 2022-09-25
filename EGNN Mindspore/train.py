import argparse
import os
import mindspore as ms
from mindspore import context, nn, ParameterTuple, ops, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.nn import piecewise_constant_lr
import src.util as util
import mindspore as ms
from src.model.egnn import GraphNetwork, GnnWithLoss
from src.data.dataloader import MiniImageNet
from src.model.embedding import EmbeddingImagenet
import numpy as np
import random
import time


# class TrainOneStepCell(nn.Cell):
#     def __init__(self, network, optimizer, sens=1.0):
#         super(TrainOneStepCell, self).__init__(auto_prefix=True)
#         self.network = network
#         self.network.set_grad()
#         self.optimizer = optimizer
#         self.weights = ParameterTuple(self.network.trainable_params())
#         self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
#         # self.grad = ops.GradOperation(get_by_list=True)
#         self.sens = sens
#
#     def set_sens(self, value):
#         self.sens = value
#
#     def construct(self, *input):
#         weights = self.weights
#         loss = self.network(*input)
#         sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
#         grads = self.grad(self.network, weights)(*input, sens)
#         return ops.functional.depend(loss, self.optimizer(grads)), loss
#         # sum_grad = 0.
#         # for grad in grads:
#         #     sum_grad += ops.ReduceSum()(grad)
#         # print('sum_grad: ', sum_grad)
#         # print('grads_0_shape: ', grads[0].shape)
#         # sum_weight = 0.
#         # for weight in weights:
#         #     sum_weight += ops.ReduceSum()(weight)
#         # print('sum_weight: ', sum_weight)
#         # return self.optimizer(grads), loss


class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 gnn_module,
                 data_loader,
                 arg):
        # set encoder and gnn
        self.enc_module = enc_module
        self.gnn_module = gnn_module
        self.arg = arg

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        # self.module_params = list(self.enc_module.trainable_params()) + list(self.gnn_module.trainable_params())

        # set optimizer
        # multiStepLR = piecewise_constant_lr([15000, 30000, 45000, 60000, 75000, 100000],
        #                                    util.get_lr(arg.lr))

        # self.optimizer = nn.Adam(self.module_params, learning_rate=multiStepLR, weight_decay=arg.weight_decay)

        # set loss
        self.edge_loss = ops.BinaryCrossEntropy(reduction='none')

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def validation(self, network):
        network.set_train(mode=False)
        acc = 0.
        N = self.arg.test_iteration // self.arg.task_batch
        num_supports = self.arg.num_way * self.arg.num_shot
        num_queries = self.arg.num_way * self.arg.num_query
        for iter in range(1, N + 1):
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['val'].get_task_batch(num_tasks=self.arg.task_batch,
                                                                   num_ways=self.arg.num_way,
                                                                   num_shots=self.arg.num_shot,
                                                                   num_queries=self.arg.num_query,
                                                                   seed=iter + self.arg.seed)
            full_label = ops.Concat(1)([support_label, query_label])
            full_edge = util.label2edge(full_label)

            # set init edge
            init_edge = full_edge.copy()  # batch_size x 2 x num_samples x num_samples
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            _ = network(support_data, support_label, query_data, query_label, init_edge, full_edge)
            acc += network.acc

        network.set_train(mode=True)

        return acc / N

    def train(self):

        # set edge mask (to distinguish support and query edges)
        num_supports = self.arg.num_way * self.arg.num_shot
        num_queries = self.arg.num_way * self.arg.num_query
        num_samples = num_supports + num_queries
        support_edge_mask = ops.Zeros()((self.arg.task_batch, num_samples, num_samples), ms.float32)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask

        evaluation_mask = ops.Ones()((self.arg.task_batch, num_samples, num_samples), ms.float32)

        network = GnnWithLoss(self.enc_module, self.gnn_module, self.edge_loss, query_edge_mask, evaluation_mask,
                              self.arg)

        # set optimizer
        module_params = ParameterTuple(network.trainable_params())

        # set optimizer
        multiStepLR = piecewise_constant_lr([15000, 30000, 45000, 60000, 75000, 100000], util.get_lr(arg.lr))

        optimizer = nn.Adam(module_params, learning_rate=multiStepLR, weight_decay=self.arg.weight_decay)
        train_one_step = nn.TrainOneStepCell(network, optimizer)

        best_val = 0.
        avg_loss = {
            'loss': 0.,
            'acc': 0.,
            'num': 0.
        }

        network.set_train(mode=True)

        time_start = time.time()
        time_end = time.time()

        # with open('EGNN/save_model/log/', 'a') as f:
        #     f.write('Train Start')
        save_checkpoint(self.enc_module, 'best_val_enc.ckpt')
        save_checkpoint(self.gnn_module, 'best_val_gnn.ckpt')
        print('Train Start')

        # for each iteration
        for iter in range(self.global_step + 1, self.arg.train_iteration + 1):

            # set current step
            self.global_step = iter

            # load task data list
            [support_data,
             support_label,
             query_data,
             query_label] = self.data_loader['train'].get_task_batch(num_tasks=self.arg.task_batch,
                                                                     num_ways=self.arg.num_way,
                                                                     num_shots=self.arg.num_shot,
                                                                     num_queries=self.arg.num_query,
                                                                     seed=iter + self.arg.seed)
            # print('query_label: ', query_label)
            # with open('EGNN/log/log.txt', 'a') as f:
            #     f.write('support shape: {}'.format(support_data.shape))
            # print('support shape: ', support_data.shape)

            full_label = ops.Concat(1)([support_label, query_label])
            full_edge = util.label2edge(full_label)

            # set init edge
            init_edge = full_edge.copy()  # batch_size x 2 x num_samples x num_samples
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            loss = train_one_step(support_data, support_label, query_data, query_label, init_edge, full_edge)
            acc = network.acc

            avg_loss['loss'] = (avg_loss['loss'] * avg_loss['num'] + loss) / (avg_loss['num'] + 1)
            avg_loss['acc'] = (avg_loss['acc'] * avg_loss['num'] + acc) / (avg_loss['num'] + 1)
            avg_loss['num'] += 1

            if iter > 0 and iter % self.arg.print_every == 0 or iter == self.arg.train_iteration:
                time_end = time.time()
                # with open('./log/log.txt', 'a') as f:
                #     f.write('iter {} loss {:.4f}  train_acc {:.4f}  using {}s\n'.format(iter,
                #                                                                         avg_loss['loss'],
                #                                                                         avg_loss['acc'],
                #                                                                         int(time_end-time_start)))
                print('iter {} loss {}  train_acc {}  using {}s'.format(iter,
                                                                                avg_loss['loss'],
                                                                                avg_loss['acc'],
                                                                                int(time_end-time_start)))
                avg_loss['loss'] = avg_loss['num'] = avg_loss['acc'] = 0.
                time_start = time.time()

            if iter > 0 and iter % self.arg.valid_every == 0 or iter == self.arg.train_iteration:
                time_start = time.time()
                acc = self.validation(network)
                time_end = time.time()
                # acc = acc.asnumpy()
                # with open('./log/log.txt', 'a') as f:
                #     f.write('------iter {} validation acc {:.4f}  using {}s\n'.format(iter,
                #                                                                       acc,
                #                                                                       int(time_end-time_start)))
                print('------iter {} validation acc {}  using {}s'.format(iter,
                                                                              acc,
                                                                              int(time_end-time_start)))
                time_start = time.time()
                if acc >= best_val:
                    best_val = acc
                    save_checkpoint(self.enc_module, 'best_val_enc.ckpt')
                    save_checkpoint(self.gnn_module, 'best_val_gnn.ckpt')
                save_checkpoint(self.enc_module, 'model_iter_{}_enc.ckpt'.format(iter))
                save_checkpoint(self.gnn_module, 'model_iter_{}_gnn.ckpt'.format(iter))
                #     save_checkpoint(self.enc_module, os.path.join(self.arg.save_path, 'best_val_enc.ckpt'))
                #     save_checkpoint(self.gnn_module, os.path.join(self.arg.save_path, 'best_val_gnn.ckpt'))
                # save_checkpoint(self.enc_module, os.path.join(self.arg.save_path,
                #                                               'model_iter_{}_enc.ckpt'.format(iter)))
                # save_checkpoint(self.gnn_module, os.path.join(self.arg.save_path,
                #                                               'model_iter_{}_gnn.ckpt'.format(iter)))

            if self.arg.device_target == 'Ascend' and iter % 5000 == 0:
                md_save_path = self.arg.train_url
                import moxing as mox
                mox.file.make_dirs(md_save_path)
                mox.file.copy_parallel(src_url='./', dst_url=md_save_path)

        if self.arg.device_target == 'Ascend':
            md_save_path = self.arg.train_url
            import moxing as mox
            mox.file.make_dirs(md_save_path)
            mox.file.copy_parallel(src_url='./', dst_url=md_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='./dataset/')
    parser.add_argument('--save_path', default='./save_model/')
    parser.add_argument('--dataset', default='mini-imagenet')
    parser.add_argument('--task_batch', type=int, default=40)
    parser.add_argument('--test_batch', type=int, default=40)
    parser.add_argument('--num_way', type=int, default=5)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=1)
    parser.add_argument('--train_iteration', type=int, default=100000)
    parser.add_argument('--test_iteration', type=int, default=10000)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--valid_every', type=int, default=5000)
    parser.add_argument('--lr_dec', type=int, default=15000)
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_edge_features', type=int, default=96)
    parser.add_argument('--num_node_features', type=int, default=96)
    parser.add_argument('--transductive', type=bool, default=True)

    # add for modelart.
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)

    arg = parser.parse_args()
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=arg.device_target, save_graphs=False)
    context.set_context(mode=context.GRAPH_MODE, device_target=arg.device_target, save_graphs=False)

    np.random.seed(arg.seed)
    ms.set_seed(arg.seed)
    random.seed(arg.seed)

    # enc_module = EmbeddingImagenet(emb_size=arg.embed_size)
    #
    # param_dict = load_checkpoint('./save_model/model_iter_85000_enc.ckpt', strict_load=True)
    # load_param_into_net(enc_module, param_dict, strict_load=True)
    #
    # gnn_module = GraphNetwork(in_features=arg.embed_size,
    #                           node_features=arg.num_node_features,
    #                           edge_features=arg.num_edge_features,
    #                           num_layers=arg.num_layer,
    #                           dropout=arg.dropout)

    enc_module = EmbeddingImagenet(emb_size=arg.embed_size)
    # param_dict = load_checkpoint('./save_model/model_iter_85000_enc.ckpt')
    # tmp_dict = {}
    # for key in param_dict.keys():
    #     tmp_dict[key[11:]] = param_dict[key]
    # param_dict = tmp_dict
    # not_load = load_param_into_net(enc_module, param_dict)
    # print('not load: ', not_load)

    gnn_module = GraphNetwork(in_features=arg.embed_size,
                              node_features=arg.num_node_features,
                              edge_features=arg.num_edge_features,
                              num_layers=arg.num_layer,
                              dropout=arg.dropout)
    # param_dict = load_checkpoint('./save_model/model_iter_85000_gnn.ckpt')
    # tmp_dict = {}
    # for key in param_dict.keys():
    #     s = key[18:]
    #     if s[0] != '0':
    #         s = '0.' + s
    #     tmp_dict[s] = param_dict[key]
    # param_dict = tmp_dict
    # not_load = load_param_into_net(gnn_module, param_dict)
    # print('not load: ', not_load)

    st = time.time()
    save_checkpoint(enc_module, 'best_val_enc.ckpt')
    save_checkpoint(gnn_module, 'best_val_gnn.ckpt')

    if arg.device_target == 'Ascend':
        md_save_path = arg.train_url
        import moxing as mox
        mox.file.make_dirs(md_save_path)
        mox.file.copy_parallel(src_url='./', dst_url=md_save_path)

    ed = time.time()
    print('save ok: {}s'.format(ed-st))

    st = time.time()
    save_checkpoint(enc_module, 'best_val_enc.ckpt')
    save_checkpoint(gnn_module, 'best_val_gnn.ckpt')

    if arg.device_target == 'Ascend':
        md_save_path = arg.train_url
        import moxing as mox
        mox.file.make_dirs(md_save_path)
        mox.file.copy_parallel(src_url='./', dst_url=md_save_path)

    ed = time.time()
    print('save ok: {}s'.format(ed - st))

    if arg.device_target == 'GPU':
        context.set_context(device_id=arg.device_id)
    elif arg.device_target == 'Ascend':
        print('In Ascend')
        import moxing as mox

        mox.file.copy_parallel(src_url=arg.data_url, dst_url=arg.root_path)

    train_loader = MiniImageNet(root=arg.root_path, partition='train')
    valid_loader = MiniImageNet(root=arg.root_path, partition='val')

    data_loader = {'train': train_loader,
                   'val': valid_loader}

    trainer = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           arg=arg)

    trainer.train()




