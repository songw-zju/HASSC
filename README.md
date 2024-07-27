# HASSC
> Song Wang, Jiawei Yu, Wentong Li, Wenyu Liu, Xiaolu Liu, Junbo Chen*, Jianke Zhu*

This is the official implementation of **Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation** (CVPR 2024)  [[Paper](https://arxiv.org/pdf/2404.11958.pdf)] [[Video](https://www.youtube.com/watch?v=UoVYkmW_N6g)].

<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>



## Preparation

### SemanticKITTI Download

- The **semantic scene completion dataset v1.1** (SemanticKITTI voxel data, 700 MB) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download).
- The **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### Environment Setup

We release the HASSC implementation with VoxFormer-T, please refer the environment setup in the [original repo](https://github.com/NVlabs/VoxFormer).



## Run and Eval

Train the SSC model **with our proposed HASSC** on 4 GPUs 

```
./tools/dist_train.sh ./projects/configs/hassc/hassc-voxformer-T.py 4
```

Eval the SSC model **with our proposed HASSC** on 4 GPUs

```
./tools/dist_test.sh ./projects/configs/hassc/hassc-voxformer-T.py ./path/to/ckpts.pth 4
```



## Acknowledgement

Many thanks to these excellent open source projects: [VoxFormer](https://github.com/NVlabs/VoxFormer), [mmdetction3d](https://github.com/open-mmlab/mmdetection3d), [PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)



## Citations
```
@inproceedings{wang2024not,
      title={Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation},
      author={Wang, Song and Yu, Jiawei and Li, Wentong and Liu, Wenyu and Liu, Xiaolu and Chen, Junbo and Zhu, Jianke},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
}
```
