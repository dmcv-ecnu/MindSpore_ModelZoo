# LatticeNet
This is the MindSpore version of [LatticeNet: Towards Lightweight Image Super-Resolution with Lattice Block，ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670273.pdf)
# Abstract
Deep neural networks with a massive number of layers have made a remarkable breakthrough on single image super-resolution (SR), but sacrifice computation complexity and memory storage. To address this problem, we focus on the lightweight models for fast and accurate image SR. Due to the frequent use of residual block (RB) in SR models, we pursue an economical structure to adaptively combine RBs. Drawing lessons from lattice filter bank, we design the lattice block (LB) in which two butterfly structures are applied to combine two RBs. LB has the potential of various linear combinations of two RBs. Each case of LB depends on the combination coefficients which are determined by the attention mechanism. LB favors the lightweight SR model with the reduction of about half amount of the parameters while keeping the similar SR performance. Moreover, we propose a lightweight SR model, LatticeNet, which uses series connection of LBs and the backward feature fusion. Extensive experiments demonstrate that our proposal can achieve superior accuracy on four available benchmark datasets against other state-of-the-art methods, while maintaining relatively low computation and memory requirements.
# Dependencies
* Python3.6
* MindSpore:https://www.mindspore.cn/install
* numpy
* ModelArts:https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dashboard
# Citation
Please kindly cite the references in your publications if it helps your research:
```@article{2020LatticeNet,
  title={LatticeNet: Towards Lightweight Image Super-Resolution with Lattice Block},
  author={ Luo, X.  and  Xie, Y.  and  Zhang, Y.  and  Qu, Y.  and  Fu, Y. },
  journal={European Conference on Computer Vision},
  year={2020},
}
