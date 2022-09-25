# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds (CVPR 2020)

This is the unofficial implementation of **RandLA-Net** (CVPR2020, Oral presentation) using [Mindspore](https://www.mindspore.cn/), a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds. 

For official implementation, please refer to:
**[RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net)** <br />
[Qingyong Hu](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang*](https://yang7879.github.io/), [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), [Stefano Rosa](https://www.cs.ox.ac.uk/people/stefano.rosa/), [Yulan Guo](http://yulanguo.me/), [Zhihua Wang](https://www.cs.ox.ac.uk/people/zhihua.wang/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). <br />
**[[Paper](https://arxiv.org/abs/1911.11236)] [[Video](https://youtu.be/Ar3eY_lwzMk)] [[Blog](https://zhuanlan.zhihu.com/p/105433460)] [[Project page](http://randla-net.cs.ox.ac.uk/)]** <br />


	
### (1) Setup
This code has been tested with Python 3.7, Mindspore 1.6.0, CUDA 10.1 and cuDNN 7.6.5 on Ubuntu 18.04.
 
- Setup python environment
```
conda create -n randlanetMs python=3.7
sh compile_op.sh
```


### (2) S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/dataset/s3dis`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Begin train & test (we provid two version of training, one for train only, and another for train and valid):

```
python train.py //train only
python train_val.py // train & valid
python test.py --model_path ./runs  //test
```


