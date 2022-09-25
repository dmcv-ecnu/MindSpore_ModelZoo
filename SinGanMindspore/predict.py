"""Cycle GAN test."""

import os
from mindspore import Tensor
from models.cyclegan import get_generator_G
from utils.args import get_args
from dataset.cyclegan_dataset import create_dataset
from utils.reporter import Reporter
from utils.tools import save_image, load_ckpt


def predict():
    """Predict function."""
    args = get_args("predict")
    G_A = get_generator_G(args)
    G_B = get_generator_G(args)
    G_A.set_train(True)
    G_B.set_train(True)
    load_ckpt(args, G_A, G_B)
    imgs_out = os.path.join(args.train_url, "fake")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    # if not os.path.exists(os.path.join(imgs_out, "fake")):
    #     os.makedirs(os.path.join(imgs_out, "fake"))
    args.data_dir = 'SOTS'
    ds = create_dataset(args)
    reporter = Reporter(args)
    reporter.start_predict("haze to clear")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_A = Tensor(data["image"])
        path_A = str(data["image_name"][0], encoding="utf-8")
        # print("path_A:", path_A)
        path_B = path_A[0:-4] + ".jpg"
        # print("path_B:", path_B)
        fake_B = G_A(img_A)
        save_image(fake_B, os.path.join(imgs_out, path_B))
        # save_image(img_A, os.path.join(imgs_out, "fake", path_A))
    reporter.info('save fake at %s', imgs_out)
    reporter.end_predict()

    args.data_dir = 'SOTS_gt'
    ds = create_dataset(args)
    reporter = Reporter(args)
    reporter.start_predict("gt_process")
    imgs_out = os.path.join(args.train_url, "gt")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    for data in ds.create_dict_iterator(output_numpy=True):
        img_A = Tensor(data["image"])
        path_A = str(data["image_name"][0], encoding="utf-8")
        # print("path_A:", path_A)
        path_B = path_A[0:-4] + ".jpg"
        # print("path_B:", path_B)
        save_image(img_A, os.path.join(imgs_out, path_B))
        # save_image(img_A, os.path.join(imgs_out, "fake", path_A))
    reporter.info('save gt at %s', imgs_out)
    reporter.end_predict()

    # if not os.path.exists(imgs_out):
    #     os.makedirs(imgs_out)
    # if not os.path.exists(os.path.join(imgs_out, "fake_A")):
    #     os.makedirs(os.path.join(imgs_out, "fake_A"))
    # if not os.path.exists(os.path.join(imgs_out, "fake_B")):
    #     os.makedirs(os.path.join(imgs_out, "fake_B"))
    # args.data_dir = 'testA'
    # ds = create_dataset(args)
    # reporter = Reporter(args)
    # reporter.start_predict("A to B")
    # for data in ds.create_dict_iterator(output_numpy=True):
    #     img_A = Tensor(data["image"])
    #     path_A = str(data["image_name"][0], encoding="utf-8")
    #     path_B = path_A[0:-4] + "_fake_.jpg"
    #     fake_B = G_A(img_A)
    #     save_image(fake_B, os.path.join(imgs_out, "fake", path_B))
    #     save_image(img_A, os.path.join(imgs_out, "fake", path_A))
    # reporter.info('save fake_ at %s', os.path.join(imgs_out, "fake", path_A))
    # reporter.end_predict()
    # BtoA
    # args.data_dir = 'testB'
    # ds = create_dataset(args)
    # reporter.dataset_size = args.dataset_size
    # reporter.start_predict("B to A")
    # for data in ds.create_dict_iterator(output_numpy=True):
    #     img_B = Tensor(data["image"])
    #     path_B = str(data["image_name"][0], encoding="utf-8")
    #     path_A = path_B[0:-4] + "_fake_A.jpg"
    #     fake_A = G_B(img_B)
    #     save_image(fake_A, os.path.join(imgs_out, "fake_A", path_A))
    #     save_image(img_B, os.path.join(imgs_out, "fake_A", path_B))
    # reporter.info('save fake_A at %s', os.path.join(imgs_out, "fake_A", path_B))
    # reporter.end_predict()


if __name__ == "__main__":
    predict()
