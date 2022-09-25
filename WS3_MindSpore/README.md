# AAAI 2021 Weakly Supervised Semantic Segmentation for Large-Scale Point Cloud (Mindspore)

[[中文版]](README_cn.md)

- [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16455)
- [Tensorflow implementation (official)](https://github.com/Yachao-Zhang/WS3)

# Env
- python==3.7.0
- CUDA==11.1
- mindspore-gpu==1.8.1

# Quick Start

## Training from scratch
### Step 1. Train our model in s3dis dataset
```shell
python train_s3dis.py 
```
The training information, including log_file and ckpts, will be saved in `{output_path}`
### Step 2. Test our model in s3dis dataset
```shell
python test_s3dis.py --model_path {output_path}
```
Note: `{output_path}` is the output dir generated in step 1

For example:
```shell
python test_s3dis.py --model_path runs/s3dis_model/TSteps500_MaxEpoch100_BatchS6_lr0.01_lrd0.95_ls1.0_Topk500_NumTrainEp030_LP_1_RS_888_PyNateiveM_2022-09-25_15-20
```


## Using our pre-trained model
```shell
python test_s3dis.py --model_path weights/randla-59_500.ckpt 
```
