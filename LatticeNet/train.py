import argparse
import option as op
import mindspore.dataset as ds
from data.div2k import DIV2K
from model.net import LatticeNet
from mindspore import context, Model, load_checkpoint, load_param_into_net
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

if __name__ == "__main__":
    # 设置硬件参数
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"

    # 加载数据库
    train_dataset = DIV2K(op.args, name=op.args.data_train, train=True, benchmark=False)
    train_dataset.set_scale(op.args.task_id)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["LR", "HR"],  shuffle=True)
    train_de_dataset = train_de_dataset.batch(op.args.batch_size, drop_remainder=True)

    # for data in train_de_dataset.create_dict_iterator():
    #     print("LR shape: {}".format(data['LR'].shape), ", HR: {}".format(data['HR'].shape))
    #     break

    print("Init data successfully")

    #加载模型
    net = LatticeNet(op.args)
    print("Init net successfully")

    # 将参数加载到网络中
    param_dict = load_checkpoint("./premodel/LatticeNet_12-1_500.ckpt")
    load_param_into_net(net, param_dict)

    # 优化器 损失函数
    opt = nn.Adam(params=net.trainable_params(), learning_rate=2e-7)
    loss = nn.L1Loss()
    print("Init opt and lossfunction successfully")

    # 保存模型策略
    step_size = train_de_dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=40)
    ckpt_cb = ModelCheckpoint(prefix="LatticeNet", directory='./premodel', config=config_ck)
    cb += [ckpt_cb]

    model = Model(net, loss_fn=loss, optimizer=opt)

    model.train(op.args.epochs, train_de_dataset, dataset_sink_mode=True, callbacks=cb)




