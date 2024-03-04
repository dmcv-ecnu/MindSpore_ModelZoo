from os import makedirs
from os.path import exists, join
from helper_tool import DataProcessing as DP
from sklearn.metrics import confusion_matrix
from main_S3DIS_pretrain_mindspore import ms_map
import numpy as np
import time
import os
from helper_ply import write_ply
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


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + "\n")
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, dataset,cfg):
        self.config=cfg
        self.timeStap = time.strftime('test_%Y-%m-%d_%H-%M-%S', time.gmtime())
        self.test_log_path = os.path.join(cfg.experiment_dir , self.timeStap)
        self.test_save_path = os.path.join(self.test_log_path,'val_preds')
        
        os.makedirs(self.test_save_path)
        
        if cfg.experiment_dir is None:
            log_file_path = 'log_test_' + str(dataset.val_split) + '.txt'
        else:
            fname = str(dataset.val_area) + '_test.txt'
            log_file_path = os.path.join(self.test_log_path, fname)
        self.Log_file = open(log_file_path, 'a')
        total_log_path = os.path.join(cfg.experiment_dir, "..")
        self.total_log = open(total_log_path + '/log-' + cfg.total_log_dir + '.txt', 'a')
        self.total_log.write(cfg.log_dir + ' testing!\n')

        print('test log dir:',log_file_path)

    def test(self, model, dataset, test, restore_snap=None, num_votes=100):
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
        makedirs(test_path) if not exists(test_path) else None
        step_id = 1
        
        test_probs = [
            np.zeros(shape=[l.shape[0], self.config.num_classes], dtype=np.float32)
            for l in dataset.input_labels["validation"]
        ]
        if True:
            test_loader = test.create_dict_iterator()
            for i, data in enumerate(test_loader):
            # break
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
            #print(logits.shape) B * 13 * N
                logits = logits.permute(0, 2, 1)  # B *N *13
            #print(logits.shape)
                logits = logits.reshape(-1, self.config.num_classes)
                stacked_probs = F.softmax(logits, -1)
            #print(stacked_probs.shape)[B * N,13]
                stacked_labels = labels
            #print(stacked_labels.shape)[B , N]
                stacked_labels=stacked_labels.reshape(-1)

                point_idx = input_inds
                cloud_idx = cloud_inds
            
                correct = np.sum(np.argmax(stacked_probs.numpy(), axis=1) == stacked_labels.numpy())
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                log_out("step" + str(step_id) + " acc:" + str(acc), self.Log_file)

                stacked_probs = np.reshape(
                stacked_probs.numpy(),
                [
                    self.config.val_batch_size,
                    self.config.num_points,
                    self.config.num_classes,
                ],
            )
            
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    p_idx=p_idx.asnumpy()
                
                    test_probs[c_i][p_idx] = (
                    test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                )
                step_id += 1
            
        # new_min = np.min(dataset.min_possibility)
            
        log_out('\nConfusion on sub clouds', self.Log_file)
        num_val = len(dataset.input_labels['validation']) 
        confusion_list = []
          
        for i_test in range(num_val):
            probs = test_probs[i_test]
            preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
            labels = dataset.input_labels['validation'][i_test]
            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]
            
        # Regroup confusions
        C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

        # Rescale with the right number of point per class
        C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Compute IoUs
        IoUs = DP.IoU_from_confusions(C)
        m_IoU = np.mean(IoUs)
        
        s = '{:5.2f} | '.format(100 * m_IoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        log_out(s + '\n', self.Log_file)
        
        log_out('\nReproject Vote ', self.Log_file)
        proj_probs_list = []

        for i_val in range(num_val):
            proj_idx = dataset.val_proj[i_val]
            probs = test_probs[i_val][proj_idx, :]
            proj_probs_list += [probs]
            
        log_out('Confusion on full clouds', self.Log_file)
        confusion_list = []
        for i_test in range(num_val):
        # Get the predicted labels      
            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)

            # Confusion
            labels = dataset.val_labels[i_test]
            acc = np.sum(preds == labels) / len(labels)
            log_out(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc), self.Log_file)

            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]
            name = dataset.input_names['validation'][i_test] + '.ply'
            write_ply(join(test_path, 'val_preds', name), [preds, labels], ['pred', 'label'])
            
            
        # Regroup confusions
        C = np.sum(np.stack(confusion_list), axis=0)

        IoUs = DP.IoU_from_confusions(C)
        m_IoU = np.mean(IoUs)
        s = '{:5.2f} | '.format(100 * m_IoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        print('finished \n')

        log_file_path = os.path.join(self.test_log_path, 'test_mIoU_{}.txt'.format(s))
        t = open(log_file_path, 'a')
        t.close()
        log_out('test_mIoU_{}\n\n'.format(s), self.total_log)
