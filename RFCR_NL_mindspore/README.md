# Omni-supervised Point Cloud Segmentation via Gradual Receptive Field Component Reasoning

# Contents

- [Omni-supervised Point Cloud Segmentation via Gradual Receptive Field Component Reasoning](#omni-supervised-point-cloud-segmentation-via-gradual-receptive-field-component-reasoning)
- [Contents](#contents)
- [RFCR](#rfcr)
  - [Model Architecture](#model-architecture)
  - [Requirements](#requirements)
    - [Install dependencies](#install-dependencies)
  - [Dataset](#dataset)
    - [Preparation](#preparation)
    - [Directory structure of dataset](#directory-structure-of-dataset)
  - [Quick Start](#quick-start)
  - [Script Description](#script-description)
    - [Scripts and Sample Code](#scripts-and-sample-code)
    - [Script Parameter](#script-parameter)
  - [Training](#training)
    - [Training Process](#training-process)
    - [Training Result](#training-result)
  - [Evaluation](#evaluation)
    - [Evaluation Process](#evaluation-process)
    - [Evaluation Result 910](#evaluation-result-910)
  - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Inference Performance](#inference-performance)
    - [S3DIS Area 5](#s3dis-area-5)
  - [Reference](#reference)

# [RFCR](#contents)

Mindspore implementation for ***"Omni-supervised Point Cloud Segmentation via Gradual Receptive Field Component Reasoning"***

Please read the [original paper](https://arxiv.org/pdf/2105.10203v1.pdf) or [original tensorflow implementation](https://github.com/azuki-miho/RFCR/tree/ec2dbdc6c8bd76c531e8decca09ba22f98f87778) for more detailed information.

## [Model Architecture](#contents)

![img](https://gitee.com/r-ight/models/raw/master/research/cv/RFCR/figs/framework.png)

## [Requirements](#contents)

- Hardware
    - For Ascend: Ascend 910.
    - For GPU: cuda==11.1

- Framework
    - Mindspore = 1.7.0

- Third Package
    - Python==3.7.5
    - pandas==1.3.5
    - scikit-learn==0.21.3
    - numpy==1.21.5

### [Install dependencies](#contents)

1. `pip install -r requirements.txt`
2. `cd third_party` & `bash compile_op.sh`

## [Dataset](#contents)

### [Preparation](#contents)

1. Download S3DIS dataset from this [link](https://gitee.com/link?target=https%3A%2F%2Fdocs.google.com%2Fforms%2Fd%2Fe%2F1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw%2Fviewform%3Fc%3D0%26w%3D1) .
2. Uncompress `Stanford3dDataset_v1.2_Aligned_Version.zip` to `dataset/S3DIS`.
3. run `data_prepare_s3dis.py` (in `src/utils/data_prepare_s3dis.py`) to process data. The processed data will be stored in `input_0.040` and `original_ply` folders.

### [Directory structure of dataset](#contents)

```html
dataset
└──S3DIS                                     #  S3DIS dataset
   ├── input_0.040
   │   ├── *.ply
   │   ├── *_proj.pkl
   │   └── *_KDTree.pkl
   ├── original_ply
   │   └── *.ply
   │
   └── Stanford3dDataset_v1.2_Aligned_Version
```

## [Quick Start](#contents)

For GPU:

```shell
bash scripts/train_s3dis_area5_gpu.sh
bash scripts/eval_s3dis_area5_gpu.sh
```

For Ascend:

```shell
bash scripts/train_s3dis_area5_ascend.sh
bash scripts/eval_s3dis_area5_ascend.sh
```

## [Script Description](https://gitee.com/r-ight/models/tree/master/research/cv/Omni-randla#contents)

### [Scripts and Sample Code](https://gitee.com/r-ight/models/tree/master/research/cv/RFCR#contents)

```html
RFCR
├── scripts
│   ├── eval_s3dis_area5_ascend.sh           # Evaluate: S3DIS Area 5 on Ascend
│   ├── eval_s3dis_area5_gpu.sh              # Evaluate: S3DIS Area 5 on GPU
│   ├── train_s3dis_area5_ascend.sh          # Train: S3DIS Area 5 on Ascend
│   └── train_s3dis_area5_gpu.sh             # Train: S3DIS Area 5 on GPU
├── src
|   ├── data                                 # class and functions for Mindspore dataset
│   │   └── dataset.py                       # dataset class for train
│   ├── model                                # network architecture and loss function
│   │   ├── model.py                         # network architecture
│   └── utils
│       ├── data_prepare_s3dis.py            # data processor for s3dis dataset
│       ├── helper_ply.py                    # file utils
│       ├── logger.py                        # logger
│       └── tools.py                         # DataProcessing and Config
├── third_party
|   ├── cpp_wrappers                         # dependency for point cloud subsampling
|   ├── meta                                 # meta information for data processor
|   ├── nearest_neighbors                    # dependency for point cloud nearest_neighbors
|   └── compile_op.sh                        # shell for installing dependencies, including cpp_wrappers and nearest_neighbors
|
├── eval.py
├── README.md
├── requirements.txt
└── train.py
```

### [Script Parameter](#contents)

we use `train_s3dis_area5_gpu.sh` as an example

```shell
python -B train.py \
  --epochs 100 \
  --batch_size 6 \
  --val_area Area_5 \
  --scale \
  --name rfcr-area5-1 \
  --device_id 0 \
  --device_target GPU \
  --lr 0.01
```

The following table describes the arguments. Some default Arguments are defined in `src/utils/tools.py`. You can change freely as you want.

| Config Arguments  | Explanation                                                  |
| ----------------- | ------------------------------------------------------------ |
| `--epoch`         | number of epochs for training                                |
| `--batch_size`    | batch size                                                   |
| `--val_area`      | which area to validate                                       |
| `--scale`         | use auto loss scale or not                                   |
| `--device_target` | chose "Ascend" or "GPU"                                      |
| `--device_id`     | which Ascend AI core/GPU to run(default:0)                   |
| `--outputs_dir`   | where stores log and network weights                         |
| `--name`          | experiment name, which will combine with outputs_dir. The output files for current experiments will be stores in `outputs_dir/name` |
| `--lr`            | learning rate                                                |

## [Training](#contents)

### [Training Process](#contents)

For GPU on S3DIS area 5:

```shell
bash scripts/train_s3dis_area5_gpu.sh
```

For Ascend on S3DIS area 5:

```shell
bash scripts/train_s3dis_area5_ascend.sh
```

### [Training Result](#contents)

Using `bash scripts/train_s3dis_area5_ascend.sh` as an example:

Training results will be stored in `/runs/rfcr-area5-1` , which is determined by `{args.outputs_dir}/{args.name}/ckpt`. For example:

```html
runs
├── rfcr-area5-1
    ├── 2022-1-2_time_11_23_40_rank_0.log
    └── ckpt
         ├── rfcr_1_500.ckpt
         ├── rfcr_2_500.ckpt
         └── ....
```

## [Evaluation](#contents)

### [Evaluation Process](#contents)

For GPU on S3DIS area 5:

```shell
bash scripts/eval_s3dis_area5_gpu.sh
```

For Ascend on S3DIS area 5:

```shell
bash scripts/eval_s3dis_area5_ascend.sh
```

Note: Before you start eval, please guarantee `--model_path` is equal to `{args.outputs_dir}/{args.name}` when training.

### [Evaluation Result 910](#contents)

```html
Area_5_office_7 Acc:0.8879567881846802
Area_5_office_8 Acc:0.9202214707757128
Area_5_office_9 Acc:0.9063314952385957
Area_5_pantry_1 Acc:0.740484280365255
Area_5_storage_1 Acc:0.6220477998785945
Area_5_storage_2 Acc:0.554925740004793
Area_5_storage_3 Acc:0.6989768597112875
Area_5_storage_4 Acc:0.8327093412130874
--------------------------------------------------------------------------------------
62.31 | 92.41 97.59 79.86  0.00 14.14 61.49 40.13 77.26 86.94 67.70 70.90 67.60 54.02
--------------------------------------------------------------------------------------
```

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters          | Ascend 910                                                | GPU (3090)                   |
| ------------------- | --------------------------------------------------------- | ---------------------------- |
| Model Version       | RFCR                                                      | RFCR                         |
| Resource            | Ascend 910; CPU 2.60GHz, 24cores; Memory 96G; OS Euler2.8 | Nvidia GeForce RTX 3090      |
| uploaded Date       | 1/2/2023 (month/day/year)                                 | 1/2/2023 (month/day/year)    |
| MindSpore Version   | 1.7.0                                                     | 1.7.0                        |
| Dataset             | S3DIS                                                     | S3DIS                        |
| Training Parameters | epoch=100, batch_size = 6                                 | epoch=100, batch_size = 20   |
| Optimizer           | Adam                                                      | Adam                         |
| Loss Function       | Softmax Cross Entropy                                     | Softmax Cross Entropy        |
| outputs             | feature vector + probability                              | feature vector + probability |
| Speed               | 1500 ms/step                                              | 1000 ms/step                 |
| Total time          | About 32 h 47 mins                                        | About 20 h 14 mins           |
| Checkpoint          | 57.26 MB (.ckpt file)                                     | 57.26 MB (.ckpt file)        |

### [Inference Performance](#contents)

| Parameters        | Ascend                       | GPU                          |
| ----------------- | ---------------------------- | ---------------------------- |
| Model Version     | RFCR                         | RFCR                         |
| Resource          | Ascend 910; OS Euler2.8      | Nvidia GeForce RTX 3090      |
| Uploaded Date     | 1/2/2023 (month/day/year)    | 1/2/2023 (month/day/year)    |
| MindSpore Version | 1.7.0                        | 1.7.0                        |
| Dataset           | S3DIS                        | S3DIS                        |
| batch_size        | 20                           | 20                           |
| outputs           | feature vector + probability | feature vector + probability |
| Accuracy          | See following tables         | See following tables         |

### [S3DIS Area 5](#contents)

| Metric | Setting | Value(Tensorflow) | Value(Mindspore, Ascend) | Value(Mindspore, GPU) |
| ------ | ------- | ----------------- | ------------------------ | --------------------- |
| mIoU   | 1%      | 64.0%             | 63.1%                    | 63.0%                 |

## [Reference](#contents)

Please kindly cite the original paper references in your publications if it helps your research:

```html
@InProceedings{Gong_2021_CVPR,
    author    = {Gong, Jingyu and Xu, Jiachen and Tan, Xin and Song, Haichuan and Qu, Yanyun and Xie, Yuan and Ma, Lizhuang},
    title     = {Omni-Supervised Point Cloud Segmentation via Gradual Receptive Field Component Reasoning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11673-11682}
}
```
