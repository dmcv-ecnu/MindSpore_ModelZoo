import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='path to dataset')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--labeled_point', type=float, default=0.01, help='1, 1%')
    parser.add_argument('--knn', type=int, default=16, help='k_nn')

    parser.add_argument('--target_platform', type=int, default=1, help='0:CPU 1:GPU 2:Ascend')
    parser.add_argument('--use_modelart', type=bool, default=False, help='run under modelart')
    parser.add_argument('--data_url', type=str, default='', help='run under modelart')
    parser.add_argument('--train_url', type=str, default='', help='model save dir')
    parser.add_argument('--device_num', type=int, default=0, help='dummy argument for modelart')
    
    # parser.add_argument('--output_dir', type=str, default='', help='model save dir')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    return parser.parse_args()