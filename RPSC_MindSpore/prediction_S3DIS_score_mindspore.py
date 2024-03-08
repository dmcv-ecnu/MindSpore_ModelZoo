from os import makedirs
from os.path import exists, join
from helper_tool import DataProcessing as DP
import numpy as np
import os
from mindspore import (
    Model,
    Tensor,
    context,
    load_checkpoint,
    load_param_into_net,
    nn,
    ops,
)
from mindspore.ops import functional as F
import mindspore.ops as P


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + "\n")
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, cfg):
        self.config = cfg
        self.threshold = self.config.threshold
        self.span = self.config.span
        self.score_layer_num = self.config.score_layer_num
        self.gt_path = "./S3DIS_gt_2"

        self.test_log_path = os.path.join(cfg.experiment_dir, "prediction")
        self.last_path = cfg.last_path
        self.test_save_path = os.path.join(self.test_log_path, "val_preds")
        makedirs(self.test_save_path) if not exists(self.test_save_path) else None

        log_file_path = os.path.join(cfg.experiment_dir, "prediction", "log" + ".txt")
        self.Log_file = open(log_file_path, "a")
        print("test log dir:", log_file_path)
        total_log_path = os.path.join(cfg.experiment_dir, "..")
        self.total_log = open(
            total_log_path + "/log-" + cfg.total_log_dir + ".txt", "a"
        )
        self.total_log.write(cfg.log_dir + " prediction!\n")

    def np_normalized(self, z):
        a = z / np.reshape(np.sum(z, axis=1), [-1, 1])
        return a

    def to_one_hot(self, x, y, num_class=13):
        ohx = np.zeros((x.shape[0], int(num_class)))
        ohx[range(len(x)), x] = y
        return ohx

    def test(self, model, dataset, test, weight, restore_snap=None, num_votes=100):
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)
        param_dict = load_checkpoint(str(restore_snap))
        load_param_into_net(model, param_dict)
        log_out("Model restored from " + restore_snap, self.Log_file)
        log_out("Model restored from " + restore_snap, self.total_log)

        test_smooth = 0.95
        val_proportions = np.zeros(self.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum(
                    [np.sum(labels == label_val) for labels in dataset.val_labels]
                )
                i += 1
        test_path = self.test_log_path
        last_path = self.last_path
        makedirs(test_path) if not exists(test_path) else None
        (
            makedirs(join(test_path, "pseudo_label"))
            if not exists(join(test_path, "pseudo_label"))
            else None
        )
        makedirs(join(test_path, "pre")) if not exists(join(test_path, "pre")) else None
        step_id = 1

        test_probs = [
            np.zeros(shape=[l.shape[0], self.config.num_classes], dtype=np.float32)
            for l in dataset.input_labels["validation"]
        ]
        test_score = [
            np.zeros(shape=[l.shape[0], self.config.num_classes], dtype=np.float32)
            for l in dataset.input_labels["validation"]
        ]
        test_embed = [
            np.zeros(shape=[l.shape[0], 32], dtype=np.float32)
            for l in dataset.input_labels["validation"]
        ]

        test_loader = test.create_dict_iterator()
        for i, data in enumerate(test_loader):
            features = data["features"]
            labels = data["labels"]
            input_inds = data["input_inds"]
            cloud_inds = data["cloud_inds"]
            labels = data["labels"]
            xyz = [data["p0"], data["p1"], data["p2"], data["p3"], data["p4"]]
            neigh_idx = [data["n0"], data["n1"], data["n2"], data["n3"], data["n4"]]
            sub_idx = [data["pl0"], data["pl1"], data["pl2"], data["pl3"], data["pl4"]]
            interp_idx = [data["u0"], data["u1"], data["u2"], data["u3"], data["u4"]]

            logits, embedding, _, score_pred_list = model(
                xyz,
                features,
                neigh_idx,
                sub_idx,
                interp_idx,
                labels,
            )
            # print(logits.shape) B * 13 * N
            logits = logits.transpose(0, 2, 1)  # B *N *13
            # print(logits.shape)
            logits = logits.reshape(-1, self.config.num_classes)
            stacked_probs = F.softmax(logits, -1)
            # print(stacked_probs.shape)[B * N,13]
            # print(embedding.shape)[B * d * N]
            embedding = embedding.transpose(0, 2, 1)  # B * N * d
            # print(embedding.shape)
            stacked_embed = P.L2Normalize(axis=-1)(embedding)

            stacked_score_pred_list = score_pred_list
            stacked_labels = labels
            # print(stacked_labels.shape)[B , N]
            stacked_labels = stacked_labels.reshape(-1)

            point_idx = input_inds
            cloud_idx = cloud_inds
            # print(point_idx.shape)B * N
            # print(cloud_idx.shape)B * 1

            correct = np.sum(
                np.argmax(stacked_probs.numpy(), axis=1) == stacked_labels.numpy()
            )
            acc = correct / float(np.prod(np.shape(stacked_labels)))
            log_out("step" + str(step_id) + " acc:" + str(acc), self.Log_file)
            # print(stacked_probs.shape)[B*N ,C]
            stacked_probs = np.reshape(
                stacked_probs.numpy(),
                [
                    self.config.val_batch_size,
                    self.config.num_points,
                    self.config.num_classes,
                ],
            )
            # print(stacked_embed.shape) B * N * 32
            stacked_embed = np.reshape(
                stacked_embed.numpy(),
                [self.config.val_batch_size, self.config.num_points, 32],
            )
            # stacked_score_pred_list:B C N
            # print(stacked_score_pred_list[0])
            l = len(stacked_score_pred_list)
            for i in range(l):
                stacked_score_pred_list[i] = stacked_score_pred_list[i].transpose(
                    0, 2, 1
                )
            for j in range(np.shape(stacked_probs)[0]):
                probs = stacked_probs[j, :, :]
                embed = stacked_embed[j, :, :]
                # print(probs.shape) N * 13
                # print(embed.shape) N * 32
                score_pred = stacked_score_pred_list[-1][j, :, :].numpy()  # N * 13
                for i in range(self.score_layer_num):
                    if i == 0:
                        score_pred = (
                            0.25 * score_pred
                            + 0.75 * stacked_score_pred_list[-2][j, :, :].numpy()
                        )
                    else:
                        score_pred = (
                            0.9 * stacked_score_pred_list[-i - 1][j, :, :].numpy()
                            + 0.1 * score_pred
                        )
                p_idx = point_idx[j, :]  # N
                c_i = cloud_idx[j][0]  # 1
                p_idx = p_idx.asnumpy()

                test_probs[c_i][p_idx] = (
                    test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                )
                # print(test_probs[c_i][p_idx].shape) N * 13
                # print(score_pred.shape) N * 13
                test_score[c_i][p_idx] = (
                    test_smooth * test_score[c_i][p_idx]
                    + (1 - test_smooth) * score_pred
                )
                test_embed[c_i][p_idx] = (
                    test_smooth * test_embed[c_i][p_idx] + (1 - test_smooth) * embed
                )
            step_id += 1

        log_out("\nConfusion on sub clouds", self.Log_file)
        num_val = len(dataset.input_labels["validation"])

        sum_point = 0
        unpseudo_point = 0
        sum_correct = 0
        sum_all = 0
        sum_label = {}
        dt_label = {}
        for i in range(13):
            sum_label[i] = 0
            dt_label[i] = 0

        for i_test in range(num_val):
            pred_save = {"probs": [], "preds": []}
            embedding = test_embed[i_test]
            # print(embedding.shape) 59831 *13
            probs = test_probs[i_test]
            # print(probs.shape) 59831 *13
            probs2 = self.np_normalized(probs)
            # print(probs2.shape)
            pred_save["probs"] = probs2
            preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.int32)
            score_preds = dataset.label_values[
                np.argmax(test_score[i_test], axis=1)
            ].astype(np.int32)

            each_sum = np.zeros_like(np.max(probs2, axis=0))
            for i in range(13):
                sum_label[i] = sum_label[i] + np.sum(preds == i)
                each_sum[i] = np.sum(preds == i)
            probs_one = self.to_one_hot(
                np.argmax(probs2, axis=1).astype(np.int32),
                np.max(probs2, axis=1),
                num_class=13,
            )
            cloud_name = dataset.input_names["validation"][i_test]
            if last_path == "None":
                probs_last = probs2.copy()
            else:
                pre_path = join(last_path, "prediction", "pre", cloud_name + ".npy")
                probs_last = np.squeeze(np.load(pre_path))
            preds_last = dataset.label_values[np.argmax(probs_last, axis=1)].astype(
                np.int32
            )

            ascii_name = join(test_path, "pre", cloud_name)
            np.save(ascii_name, 0.15 * probs2 + 0.85 * probs_last)

            gt_label_file = join(self.gt_path, cloud_name + ".npy")
            gt_labels = np.squeeze(np.load(gt_label_file))
            gt_label = np.reshape(gt_labels[np.argwhere(gt_labels != 13)], [-1])
            label_no_exist = list(np.setdiff1d(dataset.label_values, gt_label))
            for i in label_no_exist:
                preds[np.argwhere(preds == i)] = 13
                score_preds[np.argwhere(score_preds == i)] = 13

            mean_list = []
            for i in range(self.config.num_classes):
                a = probs_one[np.argwhere(preds == dataset.label_values[i])]
                b = a[:, :, i]
                if np.shape(b)[0] == 0 or dataset.label_values[i] in label_no_exist:
                    mean_list.append(0)
                else:
                    mean_list.append(np.mean(b))

            each_sum = each_sum / np.sum(each_sum)
            p_t_low = (0.5 - each_sum / 2) * (0.8 - 0.02 * weight) + (
                0.5 + each_sum / 2
            ) * np.array(mean_list)
            # p_t_low[np.argwhere(p_t_low < 0.5)] = 0.5
            p_t_low[np.argwhere(p_t_low >= 0.0)] = 0.7
            pt_low = np.repeat(
                np.reshape(p_t_low, [1, -1]), repeats=probs2.shape[0], axis=0
            )
            probs3 = probs2 - pt_low
            uncertainty_idx = np.reshape(np.argwhere(np.max(probs3, axis=1) < 0), [-1])
            remove_idx = uncertainty_idx

            if weight > 0:
                last_certainty = np.argwhere(np.max(probs_last, axis=1) > 0.7)
                idx_certainty = last_certainty
                idx_uncon = np.reshape(np.argwhere(preds != score_preds), [-1])
                idx_diff_same = np.reshape(np.argwhere(preds_last == score_preds), [-1])
                idx_uncon = np.intersect1d(idx_uncon, idx_diff_same)
                idx_diff = np.intersect1d(idx_uncon, idx_certainty)
                remove_idx, _ = np.unique(
                    np.concatenate([remove_idx, idx_diff]), return_index=True
                )

                certainty_idx = np.reshape(
                    np.argwhere(np.max(probs3, axis=1) >= 0), [-1]
                )
                cer_uncon_idx = np.intersect1d(idx_uncon, certainty_idx)
                idx_three_same = np.intersect1d(
                    idx_diff_same, np.reshape(np.argwhere(preds == score_preds), [-1])
                )
                half_idx = np.reshape(np.argwhere(np.max(probs2, axis=1) > 0.5), [-1])
                uncer_half_idx = np.intersect1d(uncertainty_idx, half_idx)
                half_three_same = np.intersect1d(uncer_half_idx, idx_three_same)
                candidate_idx = np.union1d(cer_uncon_idx, half_three_same)
                labeled_idx = np.reshape(np.argwhere(gt_labels != 13), [-1])
                preds_labeled = preds[labeled_idx]
                knn_idx = DP.knn_search(
                    np.expand_dims(embedding[labeled_idx, :], axis=0),
                    np.expand_dims(embedding[candidate_idx, :], axis=0),
                    1,
                )
                knn_idx = np.reshape(knn_idx, [-1])
                idx_add = np.reshape(
                    np.argwhere(preds[candidate_idx] == preds_labeled[knn_idx]), [-1]
                )
                remove_idx = np.setdiff1d(remove_idx, candidate_idx[idx_add])

            for id in remove_idx:
                preds[id] = 13

            sum_point += probs2.shape[0]
            unpseudo_point += np.sum(preds == 13)

            labels = dataset.input_labels["validation"][i_test]
            remove_idx = np.reshape(np.argwhere(preds == 13), [-1])
            labels_p = np.delete(labels, remove_idx)
            preds_p = np.delete(preds, remove_idx)

            correct_preds = np.sum(labels_p == preds_p)
            sum_preds = np.prod(np.shape(preds_p))
            acc_preds = correct_preds / float(sum_preds)
            print(sum_preds)
            print(correct_preds)
            log_out(
                "proportion = {:f}, acc_pred = {:.3f}".format(
                    1 - remove_idx.shape[0] / probs2.shape[0], acc_preds
                ),
                self.Log_file,
            )
            sum_correct += correct_preds
            sum_all += sum_preds
            for i in range(13):
                dt_label[i] = dt_label[i] + np.sum(preds == i)
            pred_save["preds"] = preds
            ascii_name = join(test_path, "pseudo_label", cloud_name)
            np.save(ascii_name, pred_save)
            log_out(ascii_name + " has saved", self.Log_file)

        pseudo_point = sum_point - unpseudo_point
        print(pseudo_point)
        print(sum_point)
        log_out(
            "sum: {:d}, pseudo labels: {:d}, proportion = {:f}".format(
                sum_point, pseudo_point, pseudo_point / sum_point
            ),
            self.Log_file,
        )
        log_out(
            "sum correct: {:d}, all: {:d}, correct acc:  = {:.3f}".format(
                sum_correct, sum_all, sum_correct / sum_all
            ),
            self.Log_file,
        )
        log_out(
            "sum: {:d}, pseudo labels: {:d}, proportion = {:f}".format(
                sum_point, pseudo_point, pseudo_point / sum_point
            ),
            self.total_log,
        )
        log_out(
            "sum correct: {:d}, all: {:d}, correct acc:  = {:.3f}".format(
                sum_correct, sum_all, sum_correct / sum_all
            ),
            self.total_log,
        )
