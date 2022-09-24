from src.model.update import NodeUpdateNetwork, EdgeUpdateNetwork
from mindspore import nn, ops, Parameter
import mindspore as ms
import src.util as util


class GraphNetwork(nn.Cell):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.CellList()

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers - 1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers - 1 else 0.0)

            self.layers.append(nn.CellList([edge2node_net, node2edge_net]))

    # forward
    def construct(self, node_feat, edge_feat):
        # for each layer
        edge_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self.layers[l][0](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self.layers[l][1](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)

        return edge_feat_list


class GnnWithLoss(nn.Cell):
    def __init__(self, enc_module, gnn_module, edge_loss, query_edge_mask, evaluation_mask, arg):
        super(GnnWithLoss, self).__init__()
        self.arg = arg
        self.enc_module = enc_module
        self.gnn_module = gnn_module
        self.edge_loss = edge_loss
        self.query_edge_mask = query_edge_mask
        self.evaluation_mask = evaluation_mask
        self.acc = ms.Tensor(0., dtype=ms.float32)
        # self.acc = Parameter(ms.Tensor(0., dtype=ms.float32), requires_grad=False)
        self.num_way = self.arg.num_way
        print('transductive: ', self.arg.transductive)

    def construct(self, support_data, support_label, query_data, query_label, init_edge, full_edge):
        query_edge_mask = self.query_edge_mask
        evaluation_mask = self.evaluation_mask

        num_supports = support_label.shape[1]
        num_queries = query_label.shape[1]
        num_samples = num_supports + num_queries
        num_way = self.num_way

        full_data = ops.Concat(1)([support_data, query_data])

        tmp_full_data = []
        for data in ops.Split(1, full_data.shape[1])(full_data):
            tmp_full_data.append(self.enc_module(ops.Squeeze(1)(data)))

        # full_data = [self.enc_module(ops.Squeeze(1)(data)) for data in ops.Split(1, full_data.shape[1])(full_data)]
        full_data = tmp_full_data
        full_data = ops.Stack(1)(full_data)  # batch_size x num_samples x featdim

        if self.arg.transductive:
            full_logit_layers = self.gnn_module(node_feat=full_data, edge_feat=init_edge)
            full_logit = full_logit_layers[-1]
        else:
            evaluation_mask[:, num_supports:, num_supports:] = 0
            full_logit = ops.Zeros()((self.arg.test_batch, 2, num_samples, num_samples), ms.float32)
            support_data = full_data[:, :num_supports]
            query_data = full_data[:, num_supports:]
            support_data_tiled = ops.Tile()(ops.ExpandDims()(support_data, 1), (1, num_queries, 1, 1))
            support_data_tiled = support_data_tiled.view(self.arg.test_batch * num_queries, num_supports, -1)
            query_data_reshaped = ops.ExpandDims()(query_data.view(self.arg.test_batch * num_queries, -1), 1)
            input_node_feat = ops.Concat(1)([support_data_tiled, query_data_reshaped])
            input_edge_feat = 0.5 * ops.Ones()((self.arg.test_batch, 2, num_supports + 1, num_supports + 1), ms.float32)
            input_edge_feat[:, :, :num_supports, :num_supports] = init_edge[:, :, :num_supports, :num_supports]
            input_edge_feat = ops.Tile()(input_edge_feat, (num_queries, 1, 1, 1))
            full_logit_layers = self.gnn_module(node_feat=input_node_feat, edge_feat=input_edge_feat)
            logit = full_logit_layers[-1]
            logit = logit.view(self.arg.test_batch, num_queries, 2, num_supports + 1, num_supports + 1)
            full_logit[:, :, :num_supports, :num_supports] = logit[:, :, :, :num_supports, :num_supports].mean(1)
            full_logit[:, :, :num_supports, num_supports:] = logit[:, :, :, :num_supports, -1].\
                swapaxes(1, 2).swapaxes(2, 3)
            full_logit[:, :, num_supports:, :num_supports] = logit[:, :, :, -1, :num_supports].swapaxes(1, 2)
            full_logit_layers = [full_logit]

        # full_edge_loss_layers = [self.edge_loss((1 - full_logit_layer[:, 0]), (1 - full_edge[:, 0]),
        #                                         ops.OnesLike()(full_logit_layer[:, 0]))
        #                          for full_logit_layer in full_logit_layers]

        full_edge_loss_layers = []
        for full_logit_layer in full_logit_layers:
            full_edge_loss_layers.append(self.edge_loss((1. - full_logit_layer[:, 0]), (1. - full_edge[:, 0]),
                                         ops.OnesLike()(full_logit_layer[:, 0])))

        # full_edge_loss_layers = [self.edge_loss(full_logit_layer[:, 0], full_edge[:, 0],
        #                                         ops.OnesLike()(full_logit_layer[:, 0]))
        #                          for full_logit_layer in full_logit_layers]

        pos_query_edge_loss_layers = []
        for full_edge_loss_layer in full_edge_loss_layers:
            pos_query_edge_loss_layers.append(
                ops.ReduceSum()(
                    full_edge_loss_layer * query_edge_mask * full_edge[:, 0] * evaluation_mask) / ops.ReduceSum(
                )(query_edge_mask * full_edge[:, 0] * evaluation_mask)
            )

        # pos_query_edge_loss_layers = [
        #     ops.ReduceSum()(full_edge_loss_layer * query_edge_mask * full_edge[:, 0] * evaluation_mask) / ops.ReduceSum(
        #     )(query_edge_mask * full_edge[:, 0] * evaluation_mask) for full_edge_loss_layer in full_edge_loss_layers]

        neg_query_edge_loss_layers = []
        for full_edge_loss_layer in full_edge_loss_layers:
            neg_query_edge_loss_layers.append(
                ops.ReduceSum()(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) / ops. \
                ReduceSum()(query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask)
            )

        # neg_query_edge_loss_layers = [
        #     ops.ReduceSum()(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) / ops. \
        #     ReduceSum()(query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) for full_edge_loss_layer in
        #     full_edge_loss_layers]

        query_edge_loss_layers = []
        for (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in zip(pos_query_edge_loss_layers,
                                                                          neg_query_edge_loss_layers):
            query_edge_loss_layers.append(
                pos_query_edge_loss_layer + neg_query_edge_loss_layer
            )

        # query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
        #                           (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
        #                           zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]

        # compute accuracy
        query_node_pred = ops.BatchMatMul()(full_logit[:, 0, num_supports:, :num_supports],
                                            util.one_hot_encode(num_way,
                                                                support_label.astype(ms.int64)))
        # print('pred: ', ops.Argmax(-1)(query_node_pred))
        query_node_accr = ops.Equal()(ops.Argmax(-1)(query_node_pred), query_label.astype(ms.int64)).astype(ms.float32)
        query_node_accr = ops.ReduceMean()(query_node_accr)
        self.acc = query_node_accr

        total_loss_layers = query_edge_loss_layers

        total_loss = []
        for l in range(len(total_loss_layers) - 1):
            total_loss += [total_loss_layers[l].view(-1) * 0.5]
        total_loss += [total_loss_layers[-1].view(-1) * 1.0]
        total_loss = ops.ReduceMean()(ops.Concat(0)(total_loss))

        return total_loss
