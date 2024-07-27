import os
import math
import torch
import numpy as np
import torch.nn as nn
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.hassc_voxformer.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss, distill_ssc_loss
from projects.mmdet3d_plugin.hassc_voxformer.utils.ssc_metric_torch import SSCMetrics
from projects.mmdet3d_plugin.hassc_voxformer.utils.hvm_header import HardVoxelMiningHead, voxel_sample


@HEADS.register_module()
class HASSCVoxFormerHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        cross_transformer,
        self_transformer,
        positional_encoding,
        embed_dims,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag=False,
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.n_classes = 20

        self.alpha = 0.2
        self.beta = 1.0
        self.lamda = 48
        self.delta = 0.1

        self.embed_dims = embed_dims
        self.bev_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)
        self.mask_embed = nn.Embedding(1, self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)
        self.hvm_head = HardVoxelMiningHead(in_channel=148, num_classes=self.n_classes)

        self.class_names = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
                            "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
                                                        0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        self.count = 0
        self.ssc_metric = SSCMetrics(self.class_names)

    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embed.weight.to(dtype)   #[128*128*16, dim]

        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

        # Load query proposals
        proposal = img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        vox_coords, ref_3d = self.get_ref_3d()

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats, 
            bev_queries,
            self.bev_h,
            self.bev_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
        )

        # Complete voxel features by adding mask tokens
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=bev_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[1], self.embed_dims).to(dtype)

        # Diffuse voxel features by deformable self attention
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            vox_feats_flatten,
            512,
            512,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
        )
        vox_feats_diff = vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)

        input_dict = {
            "x3d": vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0),
        }

        feat_dict = {
            "image_feats": mlvl_feats[0],
            "seed_feats": seed_feats,
            "vox_feats_flatten": vox_feats_flatten,
            "vox_feats_diff": vox_feats_diff.permute(3, 0, 1, 2)
        }

        out = self.hvm_head(target, input_dict)

        input_detach = {}
        input_detach['x3d'] = input_dict['x3d'].detach()

        return out, feat_dict

    def step(self, out_dict, teacher_out_dict, target, lga, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """
        ssc_pred = out_dict["ssc_logit"]

        if step_type == "train":
            loss_dict = dict()
            class_weight = self.class_weights.type_as(target)

            # compute the original semantic scene completion loss
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                weight_mask = (target != 255)  # & (target != 0)
                loss_valid = loss_ssc[weight_mask]
                decode_loss_copy = loss_valid
                lga_valid = lga[weight_mask]
                lga_weights = 1.0 + lga_valid
                decode_loss_copy = decode_loss_copy * lga_weights
                loss_dict['loss_ssc'] = decode_loss_copy.mean()

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            # compute hard voxel mining loss in the student brunch
            refined_pred = out_dict["refined_pred"]
            sampled_voxel_coords = out_dict["sampled_voxel_coords"]
            gt_voxels = voxel_sample(
                target.float().unsqueeze(1),
                sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            lga_voxels = voxel_sample(
                lga.float().unsqueeze(1),
                sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            ce_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
            hvm_loss = ce_criterion(refined_pred, gt_voxels)

            local_hardness = self.alpha + self.beta * lga_voxels
            hvm_loss = hvm_loss * local_hardness

            flatten_targets = gt_voxels.flatten()
            valid_mask = flatten_targets != 255
            class_weights = self.class_weights.type_as(ssc_pred)
            norm_weights = class_weights[flatten_targets[valid_mask]]

            hvm_loss = hvm_loss.flatten()[valid_mask].sum() / norm_weights.sum()
            loss_dict['loss_hard_voxel_mining'] = hvm_loss * 1.0

            # compute metric for reference
            teacher_ssc_pred = teacher_out_dict["ssc_logit"]
            output_voxels_tmp = ssc_pred.clone().detach()
            teacher_voxels_tmp = teacher_ssc_pred.clone().detach()
            target_voxels_tmp = target.clone().detach()
            output_voxels_tmp = torch.argmax(output_voxels_tmp, dim=1)
            teacher_voxels_tmp = torch.argmax(teacher_voxels_tmp, dim=1)
            mask = target_voxels_tmp != 255
            tp, fp, fn = self.ssc_metric.get_score_completion(output_voxels_tmp, target_voxels_tmp, mask)
            tp_sum, fp_sum, fn_sum = self.ssc_metric.get_score_semantic_and_completion(output_voxels_tmp,
                                                                                       target_voxels_tmp, mask)
            sc_iou = tp / (tp + fp + fn)
            ssc_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-5)
            ssc_miou = ssc_iou[1:].mean()

            loss_dict['sc_iou'] = sc_iou
            loss_dict['ssc_miou'] = ssc_miou

            tp, fp, fn = self.ssc_metric.get_score_completion(teacher_voxels_tmp, target_voxels_tmp, mask)
            tp_sum, fp_sum, fn_sum = self.ssc_metric.get_score_semantic_and_completion(teacher_voxels_tmp,
                                                                                       target_voxels_tmp, mask)
            teacher_sc_iou = tp / (tp + fp + fn)
            teacher_ssc_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-5)
            teacher_ssc_miou = teacher_ssc_iou[1:].mean()

            loss_dict['teacher_sc_iou'] = teacher_sc_iou
            loss_dict['teacher_ssc_miou'] = teacher_ssc_miou

            # compute self-distillation loss
            distill_mask = (target != 255)
            loss_logit_distill = distill_ssc_loss(ssc_pred, teacher_ssc_pred, distill_mask)
            dynamic_weight = math.exp(teacher_ssc_miou)
            loss_dict['loss_logit_distill'] = loss_logit_distill * dynamic_weight * self.lamda

            # compute hard voxel mining loss in the teacher brunch
            teacher_sampled_voxel_coords = teacher_out_dict["sampled_voxel_coords"]
            student_corr_pred = voxel_sample(
                ssc_pred,
                teacher_sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1)

            teacher_gt_voxels = voxel_sample(
                target.float().unsqueeze(1),
                teacher_sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            teacher_lga_voxels = voxel_sample(
                lga.float().unsqueeze(1),
                teacher_sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            teacher_ce_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
            teacher_hvm_loss = teacher_ce_criterion(student_corr_pred, teacher_gt_voxels)

            teacher_local_hardness = self.alpha + self.beta * teacher_lga_voxels
            teacher_hvm_loss = teacher_hvm_loss * teacher_local_hardness

            flatten_targets = teacher_lga_voxels.flatten()
            valid_mask = flatten_targets != 255
            class_weights = self.class_weights.type_as(ssc_pred)
            norm_weights = class_weights[flatten_targets[valid_mask]]
            teacher_hvm_loss = teacher_hvm_loss.flatten()[valid_mask].sum() / norm_weights.sum()
            loss_dict['loss_hard_voxel_mining_teacher'] = teacher_hvm_loss * self.delta
            self.count += 1
            return loss_dict

        elif step_type == "val" or "test":
            y_true = target.cpu().numpy()
            y_pred = ssc_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            if self.save_flag:
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, teacher_out_dict, target, lga, img_metas):
        """Training step.
        """
        return self.step(out_dict, teacher_out_dict, target, lga, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, None, target, None, img_metas, "val")

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1, -1)+0.5)/self.bev_h, (yv.reshape(1,-1)+0.5)/self.bev_w, (zv.reshape(1,-1)+0.5)/self.bev_z,], axis=0).astype(np.float64).T

        return vox_coords, ref_3d

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./voxformer", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))
