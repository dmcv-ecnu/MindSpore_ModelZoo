import datetime
import os
import time
import argparse
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, nn
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import Adam
from mindspore.profiler.profiling import Profiler
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import RunContext
from mindspore.train.callback import _InternalCallbackParam
from main_S3DIS_pretrain_mindspore import dataloader, ms_map, S3DISDatasetGenerator
from helper_tool import ConfigS3DIS as cfg
from RandLANet_S3DIS_pretrain_mindspore import (
    RandLANet,
    RandLAWithLoss,
    TrainingWrapper,
    get_param_groups,
    get_loss,
    get_matrix_loss,
)
import mindspore.ops as P
import mindspore.numpy as mnp
from mindspore.ops import functional as F
import numpy as np
import shutil


def log_out(out_str, f_out):
    f_out.write(out_str + "\n")
    f_out.flush()
    print(out_str)


def evaluate(network, loader, log_file):
    network = network.network_logits
    network.set_train(False)
    loader = loader.create_dict_iterator()
    gt_classes = [0 for _ in range(cfg.num_classes)]
    positive_classes = [0 for _ in range(cfg.num_classes)]
    true_positive_classes = [0 for _ in range(cfg.num_classes)]
    val_total_correct = 0
    val_total_seen = 0
    step_id = 1

    for i, data in enumerate(loader):
        print(i)

        features = data["features"]
        labels = data["labels"]
        xyz = [data["p0"], data["p1"], data["p2"], data["p3"], data["p4"]]
        neigh_idx = [data["n0"], data["n1"], data["n2"], data["n3"], data["n4"]]
        sub_idx = [data["pl0"], data["pl1"], data["pl2"], data["pl3"], data["pl4"]]
        interp_idx = [data["u0"], data["u1"], data["u2"], data["u3"], data["u4"]]

        logits, _, _, _ = network(xyz, features, neigh_idx, sub_idx, interp_idx, labels)
        logits = logits.transpose(0, 2, 1)  # B *N *13

        logits = logits.reshape(-1, cfg.num_classes)
        labels = labels.reshape(-1)

        stacked_prob = F.softmax(logits, -1)

        pred = np.argmax(stacked_prob.asnumpy(), 1)

        if not cfg.ignored_label_inds:
            pred_valid = pred
            labels_valid = labels
        else:
            invalid_idx = np.where(labels == cfg.ignored_label_inds)[0]
            labels_valid = np.delete(labels.asnumpy(), invalid_idx)
            pred_valid = np.delete(pred, invalid_idx)

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)
        conf_matrix = confusion_matrix(
            labels_valid, pred_valid, np.arange(0, cfg.num_classes, 1)
        )
        gt_classes += np.sum(conf_matrix, axis=1)
        positive_classes += np.sum(conf_matrix, axis=0)
        true_positive_classes += np.diagonal(conf_matrix)
        if step_id % 50 == 0:
            log_out(str(step_id) + " / " + str(cfg.val_steps), log_file)

        step_id += 1

    iou_list = []
    for n in range(0, cfg.num_classes, 1):
        iou = true_positive_classes[n] / float(
            gt_classes[n] + positive_classes[n] - true_positive_classes[n]
        )
        iou_list.append(iou)

    mean_iou = sum(iou_list) / float(cfg.num_classes)
    log_out(
        "eval accuracy: {}".format(val_total_correct / float(val_total_seen)),
        log_file,
    )
    log_out("mean IOU:{}".format(mean_iou), log_file)
    mean_iou *= 100
    log_out("Mean IoU = {:.1f}%".format(mean_iou), log_file)

    s = "{:5.2f} | ".format(mean_iou)
    for IoU in iou_list:
        s += "{:5.2f} ".format(100 * IoU)
    log_out("-" * len(s), log_file)
    log_out(s, log_file)
    log_out("-" * len(s) + "\n", log_file)
    return mean_iou


def train(args):

    context.set_context(
        mode=context.PYNATIVE_MODE,
        device_target="GPU",
        device_id=0,
    )
    if cfg.saving:
        log_file = open(cfg.experiment_dir + "/log_train.txt", "a")
        log_file.write(
            " ".join(
                [
                    "config.%s = %s\n" % (k, v)
                    for k, v in cfg.__dict__.items()
                    if not k.startswith("__")
                ]
            )
        )
    log_out("test Area: 5", log_file)
    bias = True
    d_in = 6
    network = RandLANet(d_in, cfg.num_classes, bias, cfg)

    decay_lr = nn.ExponentialDecayLR(
        cfg.learning_rate, 0.95, decay_steps=cfg.train_steps
    )
    opt = Adam(
        params=get_param_groups(network),
        learning_rate=decay_lr,
    )

    network = RandLAWithLoss(network)
    network = TrainingWrapper(network, opt)

    # data loader
    dataset = S3DISDatasetGenerator(args.labeled_point, args.gt_save_path)
    train_loader, test_loader = dataloader(dataset)

    train_loader = train_loader.batch(
        batch_size=cfg.batch_size,
        per_batch_map=ms_map,
        input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
        output_columns=[
            "features",
            "labels",
            "input_inds",
            "cloud_inds",
            "p0",
            "p1",
            "p2",
            "p3",
            "p4",
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "pl0",
            "pl1",
            "pl2",
            "pl3",
            "pl4",
            "u0",
            "u1",
            "u2",
            "u3",
            "u4",
        ],
        drop_remainder=True,
    )
    test_loader = test_loader.batch(
        batch_size=cfg.val_batch_size,
        per_batch_map=ms_map,
        input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
        output_columns=[
            "features",
            "labels",
            "input_inds",
            "cloud_inds",
            "p0",
            "p1",
            "p2",
            "p3",
            "p4",
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "pl0",
            "pl1",
            "pl2",
            "pl3",
            "pl4",
            "u0",
            "u1",
            "u2",
            "u3",
            "u4",
        ],
        drop_remainder=True,
    )
    first_epoch = 1
    training_step = 1

    miou_list = [0]

    best_epoch = 0
    train_loader = train_loader.create_dict_iterator()

    for epoch in range(first_epoch, cfg.max_epoch + 1):
        log_out(f"=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===", log_file)
        network.set_train()

        for i, data in enumerate(train_loader):
            t0 = time.time()
            features = data["features"]
            labels = data["labels"]
            xyz = [data["p0"], data["p1"], data["p2"], data["p3"], data["p4"]]
            neigh_idx = [data["n0"], data["n1"], data["n2"], data["n3"], data["n4"]]
            sub_idx = [data["pl0"], data["pl1"], data["pl2"], data["pl3"], data["pl4"]]
            interp_idx = [data["u0"], data["u1"], data["u2"], data["u3"], data["u4"]]

            loss = network(xyz, features, neigh_idx, sub_idx, interp_idx, labels, epoch)

            logits, embedding, _, _ = network.network_logits(
                xyz, features, neigh_idx, sub_idx, interp_idx, labels
            )
            d = embedding.shape[1]
            valid_labels, valid_logits, _ = network.network_logits.data_prep(
                logits, embedding, labels, d
            )

            topk = P.TopK(sorted=True)

            _, predicted = topk(valid_logits, 1)

            correct_prediction = P.Equal()(predicted.squeeze(1), valid_labels)

            accuracy = mnp.mean(correct_prediction.astype(mnp.float32))

            t_end = time.time()
            if training_step % 50 == 0:
                message = "Step {:08d} L_out={:5.3f} Acc={:4.2f} " "---{:8.2f} ms/batch"
                log_out(
                    message.format(
                        training_step, float(loss), float(accuracy), 1000 * (t_end - t0)
                    ),
                    log_file,
                )

            training_step += 1

        log_out("evaling", log_file)
        mean_iou = evaluate(network, test_loader, log_file)
        if mean_iou > np.max(miou_list):
            best_epoch = epoch
            save_dir = cfg.checkpoints_dir
            file_name = "best_epoch_" + str(best_epoch) + ".ckpt"
            save_dir = os.path.join(save_dir, file_name)
            ms.save_checkpoint(network.network_logits, save_dir)
        miou_list.append(mean_iou)
        log_out(
            "Best m_IoU is: {:5.3f}, epoch: {}".format(max(miou_list), best_epoch),
            log_file,
        )


def create_log_dir(args):
    """CREATE DIR"""
    import datetime, sys
    from pathlib import Path

    if args.mode == "train":
        timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        experiment_dir = Path("./experiment_mindspore/")
        experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath("S3DIS")
        experiment_dir.mkdir(exist_ok=True)
        if "%" in args.labeled_point:
            n = args.labeled_point[:-1] + "_percent_"
        else:
            n = args.labeled_point + "_points_"

        experiment_dir = experiment_dir.joinpath(n)  # model_name
        experiment_dir.mkdir(exist_ok=True)
        if args.log_dir is None:
            experiment_dir = experiment_dir.joinpath(
                timestr + "_area_" + str(args.test_area)
            )
        else:
            experiment_dir = experiment_dir.joinpath(args.log_dir)
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath("checkpoints/")
        checkpoints_dir.mkdir(exist_ok=True)
        tensorboard_log_dir = experiment_dir.joinpath("tensorboard/")
        tensorboard_log_dir.mkdir(exist_ok=True)
        shutil.copy("helper_tool.py", str(experiment_dir))
        f = sys.argv[0]
        shutil.copy(f, str(experiment_dir))
        try:
            shutil.copy(args.model_name, str(experiment_dir))
        except:
            print("文件复制错误")
            1 / 0
    elif args.mode == "test":
        model_path = args.model_path
        checkpoints_dir = model_path.split("snapshots")[0]
        # log_dir = os.path.join(model_path.split('snapshots')[0], 'logs')
        experiment_dir = model_path.split("checkpoints")[0]
    return str(experiment_dir), str(checkpoints_dir), str(tensorboard_log_dir)


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="the number of GPUs to use [default: 0]"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="options: train, test, vis"
    )
    parser.add_argument(
        "--test_area",
        type=int,
        default=5,
        help="Which area to use for test, option: 1-6 [default: 5]",
    )
    parser.add_argument("--labeled_point", type=str, default="1", help="1, 10 or 100")
    parser.add_argument(
        "--model_name", type=str, default="RandLANet_S3DIS_pretrain.py", help=""
    )
    parser.add_argument("--log_dir", type=str, default="ex", help="")
    parser.add_argument("--knn", type=int, default=16, help="k_nn")
    parser.add_argument("--gt_save_path", type=str, default="./S3DIS_gt_2", help="")
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    (
        cfg.experiment_dir,
        cfg.checkpoints_dir,
        cfg.tensorboard_log_dir,
    ) = create_log_dir(FLAGS)

    train(FLAGS)
