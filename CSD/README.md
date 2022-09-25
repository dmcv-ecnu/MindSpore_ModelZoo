# Contents

- [CSD Description](#csd-description)
    - [Abstract](#abstract)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
    - [Dependencies](#dependencies)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#train)
    - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Training Performance](#training-performance)
    - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CSD Description](#contents)

Mindspore implementation for ***Towards Compact Single Image Super-Resolution via Contrastive Self-distillation*** in IJCAI 2021. Please read our [paper](https://arxiv.org/abs/2105.11683) for more details.

For original PyTorch implementation please refer to [github](https://github.com/Booooooooooo/CSD).

## Abstract

Convolutional neural networks (CNNs) are highly successful for super-resolution (SR) but often require sophisticated architectures with heavy memory cost and computational overhead, significantly restricts their practical deployments on resource-limited devices. In this paper, we proposed a novel contrastive self-distillation (CSD) framework to simultaneously compress and accelerate various off-the-shelf SR models. In particular, a channel-splitting super-resolution network can first be constructed from a target teacher network as a compact student network. Then, we propose a novel contrastive loss to improve the quality of SR images and PSNR/SSIM via explicit knowledge transfer. Extensive experiments demonstrate that the proposed CSD scheme effectively compresses and accelerates several standard SR models such as EDSR, RCAN and CARN.

![model](https://github.com/Booooooooooo/MindSpore_ModelZoo/blob/main/CSD/images/model.png)

# [Dataset](#contents)

Training set: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), 800 images with 2K resolution.
Test set: [BSD100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), 100 images.
Please download the dataset and place the training set in `./dataset` folder. (like `./dataset/DIV2K`)
Please place the test set in `./dataset/benchmark` folder. (like `./dataset/benchmark/B100`)

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - MindSpore
- For more information, please check the resources below：
    - [MindSpore tutorials](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fen%2Fmaster%2Findex.html)
    - [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fapi%2Fen%2Fmaster%2Findex.html)

## Dependencies

- Python == 3.7.5

- MindSpore: https://www.mindspore.cn/install

- matplotlib

- imageio

- opencv-python

- scipy

- scikit-image

# [Script Description](#contents)

## Script and Sample Code

```markdown
.CSD
├─ README.md                            # descriptions about CSD
├─ images
  ├─ model.png                          # CSD model image
  ├─ table.png                          # Results of CSD
  ├─ tradeoff.png                       # Trade-off of CSD
  └─ visual.png                         # Visual Results
├─ script
  ├─ run_standalone_train_ascend.sh     # launch csd training with ascend platform
  ├─ run_distribute_train_ascend.sh     # launch csd training with ascend platform, 8pcs
  ├─ run_standalone_train_gpu.sh        # launch csd training with gpu platform
  ├─ run_eval_ascend.sh                 # launch evaluating with ascend platform
  └─ run_eval_gpu.sh                    # launch evaluating with gpu platform
├─ src
  ├─ data
    ├─ __init__.py                      # init file
    ├─ common.py                        # common functions for process dataset
    ├─ div2k.py                         # div2k dataset define
    └─ srdata.py                        # dataset define
  ├─ utils
    ├─ __init__.py                      # init file
    └─ var_init.py                      # init functions for VGG
  ├─ args.py                            # parse args
  ├─ common.py                          # common network modules define
  ├─ config.py                          # configurations for VGG
  ├─ contras_loss.py                    # contrastive loss define
  ├─ edsr_model.py                      # baseline edsr model define
  ├─ edsr_slim.py                       # slimmable edsr model define
  ├─ metric.py                          # evaluation utils
  └─ vgg_model.py                       # VGG model define
├─ eval.py                              # evaluation script
├─ export.py                            # export mindir script
└─ csd_train.py                         # csd train script
```

## Script Parameters

Major parameters in scripts as follows:

```python
"rgb_range": 255        # range of image pixels, default is 255.
"patch_size": 48        # patch size for training, default is 48.
"batch_size": 16        # batch size for training, default is 16.
"epochs": 1000          # epochs for training, default is 1000.
"dir_data":             # path of train set.
"data_test": "B100"     # name of test set.
"filename":             # model name.
"ckpt_path":            # path of the checkpoint to load.
"ckpt_save_path":       # models are saved here.
"modelArts_mode": False # whether use ModelArts.
"data_url":             # data path on OBS.
"test_only": False      # whether do evaluation.

"lr": 1e-5              # learning rate, default is 1e-5.
"loss_scale": 1024.0    # loss scale for LossScaleManager, default is 1024.0

"scale": 4              # scale factor of super resolution, default is 4.
"stu_width_mult": 0.25  # width scale factor of the student model, default is 0.25.
"contra_lambda": 0.1    # weight for contrastive loss, default is 0.1.
"neg_num": 10           # number of negative samples, default is 10.
```

## Training Process

### Train CSD

VGG pre-trained on ImageNet is used in our contrastive loss. Please download the pre-trained model from [https://download.mindspore.cn/model_zoo/r1.3/](https://download.mindspore.cn/model_zoo/r1.3/) and place it in `./` .
Please place your teacher model in `./ckpt`.

```bash
sh run_standalone_train_ascend.sh [TRAIN_DATA_PATH] [FILENAME] [TEACHER_MODEL]
sh run_standalone_train_gpu.sh [TRAIN_DATA_PATH] [FILENAME] [TEACHER_MODEL]
```

For example:

```bash
sh run_standalone_train_ascend.sh ./dataset csd_id_1 ckpt/edsr_slimbaseline_id1.ckpt
```

## Evaluation

```bash
sh run_eval_ascend.sh [DATA_PATH] [CKPT]
sh run_eval_gpu.sh [DATA_PATH] [CKPT]
```

For example:

```bash
sh run_eval_ascend.sh ./dataset/benchmark ckpt/edsr_csd_id1.ckpt
```

# [Model Description](#contents)

## Training Performance

| Parameters                 | GPU                                                |
| -------------------------- | -------------------------------------------------- |
| Model Version              | CSD                                                |
| Resource                   | GPU/TITAN RTX                                      |
| MindSpore Version          | 1.5.0-rc1                                          |
| Dataset                    | DIV2K                                              |
| Training Parameters        | epoch=1000, batch_size=16, lr=0.0001, neg_num=10   |
| Optimizer                  | Adam                                               |
| Loss Function              | L1 Loss + 200 * Contrastive Loss                   |
| outputs                    | high resolution images                             |
| Speed                      | 1pc(GPU): 5590.8 ms/step                           |
| Total time                 | 1pc(GPU): 77.65h;                                  |
| Checkpoint for Fine tuning | 6.91M (.ckpt file)                                 |

### Evaluation Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Model Version       | CSD                         |
| Resource            | GPU/TITAN RTX               |
| MindSpore Version   | 1.5.0-rc1                   |
| Dataset             | BSD100                      |
| batch_size          | 1                           |
| outputs             | high resolution images      |
| PSNR of T           | 27.5711                     |
| PSNR of S(0.25)     | 27.1373                     |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
