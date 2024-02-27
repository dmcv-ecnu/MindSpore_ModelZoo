import os
import time
from datetime import timedelta
import numpy as np
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
# from torch.cuda import amp

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context, load_checkpoint, load_param_into_net, save_checkpoint, DatasetHelper, Tensor, ops
from mindspore.context import ParallelMode
from mindspore.communication import init, get_group_size, get_rank
from mindspore.dataset.transforms.transforms import Compose
from mindspore.nn import SGD, Adam

from util.eval_metrics import extract_features_clip
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from data.data_manager import process_query_sysu, process_gallery_sysu
from data.dataloader import TestData
from util.eval import tester

def decode(img):
    return Image.fromarray(img)

def get_cluster_loader(args, dataset, batch_size, workers):
    # cluster_loader = data.DataLoader(
    #     dataset,
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=True)
    transform_test = Compose([
        decode,
        vision.Resize((args.img_h, args.img_w)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
    ])
    cluster_set = ds.GeneratorDataset(dataset, ["image", "label", "path", "modal"])
    cluster_set = cluster_set.map(
        operations=transform_test, input_columns=["image"]
    )
    cluster_set = cluster_set.batch(batch_size=batch_size, drop_remainder=True)
    cluster_loader = DatasetHelper(cluster_set, dataset_sink_mode=False)
    return cluster_set, cluster_loader

def do_train_stage1(args,
                    unlabel_dataset,
                    model,
                    optimizer,
                    scheduler
                    ):
    transform_test = Compose([
        decode,
        vision.Resize((args.img_h, args.img_w)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
    ])

    # device = "cuda"
    # scaler = amp.GradScaler()
    xent = SupConLoss(device)

    with torch.no_grad():

        # print('==> Create pseudo labels for unlabeled data')
        print("==> Extract RGB features")
        unlabel_dataset.rgb_cluster = True
        unlabel_dataset.ir_cluster = False
        cluster_set_rgb, cluster_loader_rgb = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
        features_rgb, pseudo_labels_rgb = extract_features_clip(model, cluster_set_rgb, cluster_loader_rgb, modal=1, get_image=True)
        features_rgb = ops.cat([features_rgb[path].unsqueeze(0) for path in unlabel_dataset.train_color_path], 0)
        pseudo_labels_rgb = ops.cat([pseudo_labels_rgb[path].unsqueeze(0) for path in unlabel_dataset.train_color_path], 0)

        print("==> Extract IR features")
        unlabel_dataset.ir_cluster = True
        unlabel_dataset.rgb_cluster = False
        cluster_set_ir, cluster_loader_ir = get_cluster_loader(unlabel_dataset, args.test_batch_size, args.workers)
        features_ir, pseudo_labels_ir = extract_features_clip(model, cluster_set_ir, cluster_loader_ir, modal=2, get_image=True)
        features_ir = ops.cat([features_ir[path].unsqueeze(0) for path in unlabel_dataset.train_thermal_path], 0)
        pseudo_labels_ir = ops.cat([pseudo_labels_ir[path].unsqueeze(0) for path in unlabel_dataset.train_thermal_path], 0)

    del cluster_loader_rgb, cluster_loader_ir

    # adjust pseudo where label is -1
    valid_idx_rgb = np.where(pseudo_labels_rgb.cpu() != -1)[0]
    features_rgb = features_rgb[valid_idx_rgb,:]
    labels_rgb = pseudo_labels_rgb[valid_idx_rgb]

    valid_idx_ir = np.where(pseudo_labels_ir.cpu() != -1)[0]
    features_ir = features_ir[valid_idx_ir, :]
    labels_ir = pseudo_labels_ir[valid_idx_ir]

    nums_rgb = len(labels_rgb)
    nums_ir = len(labels_ir)

    start_time = time.monotonic()
    for epoch in range(1, args.stage1_maxepochs+1):
        print('----------test-----------')
        scheduler.step(epoch)
        model.set_train()

        if nums_rgb > nums_ir:
            iter_list_rgb = ops.randperm(nums_rgb)
            iter_list_ir = ops.cat([ops.randperm(nums_ir),ops.randint(0,nums_ir,(nums_rgb-nums_ir,))],dim=0)
        elif nums_rgb == nums_ir:
            iter_list_rgb = ops.randperm(nums_rgb)
            iter_list_ir = ops.randperm(nums_ir)
        else:
            iter_list_ir = ops.randperm(nums_ir)
            iter_list_rgb = ops.cat([ops.randperm(nums_rgb), ops.randint(0, nums_rgb, (nums_ir - nums_rgb,))], dim=0)

        batch = args.stage1_batch_size
        i_ter = len(iter_list_rgb) // batch

        print('-----len of rgb and ir iter_list------',len(iter_list_rgb),len(iter_list_ir))
        print('---------------------------------------------------------------------')
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')

        loss_meter = AverageMeter()

        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list_rgb = iter_list_rgb[i*batch:(i+1)* batch]
                b_list_ir = iter_list_ir[i*batch:(i+1)* batch]
            else:
                b_list_rgb = iter_list_rgb[i * batch:len(iter_list_rgb)]
                b_list_ir = iter_list_ir[i * batch:len(iter_list_rgb)]

            target_rgb = labels_rgb[b_list_rgb]
            target_ir = labels_ir[b_list_ir]


            image_features_rgb = features_rgb[b_list_rgb]
            image_features_ir = features_ir[b_list_ir]

            text_features_rgb = model(get_text=True, label=target_rgb, modal=1)
            text_features_ir = model(get_text=True, label=target_ir, modal=2)
            loss_i2t_rgb = xent(image_features_rgb, text_features_rgb, target_rgb, target_rgb)
            loss_t2i_rgb = xent(text_features_rgb, image_features_rgb, target_rgb, target_rgb)

            loss_i2t_ir = xent(image_features_ir, text_features_ir, target_ir, target_ir)
            loss_t2i_ir = xent(text_features_ir, image_features_ir, target_ir, target_ir)

            loss = loss_i2t_rgb + loss_t2i_rgb + loss_i2t_ir + loss_t2i_ir

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss_meter.update(loss.item())

            # torch.cuda.synchronize()
            if i % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss_prompt: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (i + 1), i_ter+1,
                                        loss_meter.avg, scheduler._get_lr(epoch)[0]))

            # if epoch % args.stage1_checkpoint == 0:
            #     torch.save(model.state_dict(), os.path.join(args.model_path, args.logs_file + '_stage1_{}.pth'.format(epoch)))

        if epoch == args.stage1_maxepochs:
            print('Test Epoch: {}'.format(epoch))
            test_mode = [1, 2]
            query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
            queryset_generator = TestData(query_img, query_label, img_size=(args.img_w, args.img_h))
            # query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
            queryset = ds.GeneratorDataset(
                queryset_generator, ["img", "label"])
            queryset = queryset.map(
                operations=transform_test, input_columns=["img"])
            query_loader = queryset.batch(batch_size=args.test_batch_size)
            query_loader = DatasetHelper(query_loader, dataset_sink_mode=False)

            for trial in range(10):
                # print('-------test trial {}-------'.format(trial))
                gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
                gallset_generator = TestData(gall_img, gall_label, img_size=(args.img_w, args.img_h))
                # gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
                gallset = ds.GeneratorDataset(gallset_generator, ["img", "label"])
                gallset = gallset.map(
                    operations=transform_test, input_columns=["img"])
                gall_loader = gallset.batch(batch_size=args.test_batch_size)
                gall_loader = DatasetHelper(gall_loader, dataset_sink_mode=False)

                feat_dim = 2048
                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader, feat_dim, query_cam=query_cam, gall_cam=gall_cam)
                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP
                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10

            print(
                "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                    cmc[0], cmc[4],
                    cmc[9], cmc[19],
                    mAP, mINP))


            state = {
                "state_dict": model.state_dict(),
                "cmc": cmc,
                "mAP": mAP,
                "mINP": mINP,
                "epoch": epoch,
            }
            logs_file = str(args.logs_file)
            # torch.save(state, os.path.join(args.model_path, logs_file + "_stage1.pth"))
            save_param_list = get_param_list(model)
            path = os.path.join(args.model_path, args.logs_file + "_stage1.pth")
            save_checkpoint(save_param_list, path)

    end_time = time.monotonic()
    print('Stage1 running time: ', timedelta(seconds=end_time - start_time))









