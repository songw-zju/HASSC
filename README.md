# HASSC
> Song Wang, Jiawei Yu, Wentong Li, Wenyu Liu, Xiaolu Liu, Junbo Chen*, Jianke Zhu*

This is the official implementation of **Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation** (CVPR 2024)  [[Paper]()] [[Video]()].



## Abstract
Semantic scene completion, also known as semantic occupancy prediction, can provide dense geometric and semantic information for autonomous vehicles, which attracts the increasing attention of both academia and industry. Unfortunately, existing methods usually formulate this task as a voxel-wise classification problem and treat each voxel equally in 3D space during training. As the hard voxels have not been paid enough attention, the performance in some challenging regions is limited. The 3D dense space typically contains a large number of empty voxels, which are easy to learn but require amounts of computation due to handling all the voxels uniformly for the existing models. Furthermore, the voxels in the boundary region are more challenging to differentiate than those in the interior. In this paper, we propose HASSC approach to train the semantic scene completion model with hardness-aware design. The global hardness from the network optimization process is defined for dynamical hard voxel selection. Then, the local hardness with geometric anisotropy is adopted for voxel-wise refinement. Besides, self-distillation strategy is introduced to make training process stable and consistent. Extensive experiments show that our HASSC scheme can effectively promote the accuracy of the baseline model without incurring the extra inference cost.


## Framework
<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>



## Citations
```
@inproceedings{wang2024not,
      title={Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation},
      author={Wang, Song and Li, Wentong and Liu, Wenyu and Liu, Xiaolu and Zhu, Jianke},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
}
```
