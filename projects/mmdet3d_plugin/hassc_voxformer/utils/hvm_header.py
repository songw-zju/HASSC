import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer


class HardVoxelMiningHead(nn.Module):
    def __init__(self, in_channel=532, num_classes=20):
        super().__init__()
        self.t = 3
        self.omega = 0.75
        self.N = 4096

        self.refined_mlp = nn.Conv1d(in_channel, num_classes, 1)

        conv_cfg = {'type': 'Conv3d', 'bias': False}
        norm_cfg = {'type': 'GN', 'num_groups': 32, 'requires_grad': True}
        self.occ_conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=128,
                             out_channels=64, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=64,
                             out_channels=20, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, input_dict):
        output_dict = {}
        voxel_feature = input_dict["x3d"]
        coarse_prediction = self.occ_conv(voxel_feature)
        final_prediction = F.interpolate(coarse_prediction, scale_factor=2, mode="trilinear",
                                         align_corners=False).contiguous()
        output_dict["ssc_logit"] = final_prediction

        if not self.training:
            return output_dict

        sampled_voxels_coords = sampling_hard_voxels(coarse_prediction, self.N, self.t, self.omega)

        sampled_coarse_pred = voxel_sample(coarse_prediction, sampled_voxels_coords, align_corners=False)
        sampled_voxel_feature = voxel_sample(voxel_feature, sampled_voxels_coords, align_corners=False)

        sampled_voxel_feature = torch.cat([sampled_coarse_pred, sampled_voxel_feature], dim=1)

        output_dict["refined_pred"] = self.refined_mlp(sampled_voxel_feature)
        output_dict["sampled_voxel_coords"] = sampled_voxels_coords

        return output_dict


def voxel_sample(input, voxel_coords, **kwargs):
    add_dim = False
    if voxel_coords.dim() == 3:
        add_dim = True
        voxel_coords = voxel_coords.unsqueeze(2)
        voxel_coords = voxel_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * voxel_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(-1)
        output = output.squeeze(-1)
    return output


@torch.no_grad()
def sampling_hard_voxels(mask, N, t=3, omega=0.75):
    device = mask.device
    B, _, H, W, Z = mask.shape
    mask, _ = mask.sort(1, descending=True)
    random_sampled_coords = torch.rand(B, t * N, 3, device=device)
    random_sampled_voxels = voxel_sample(mask, random_sampled_coords, align_corners=False)

    # The largest and second largest probability in C classes
    p_a = random_sampled_voxels[:, 0]
    p_b = random_sampled_voxels[:, 1]

    # compute the global hardness and select corresponding voxels
    global_hardness = 1.0 / (p_a - p_b)
    _, voxel_index = global_hardness.topk(int(omega * N), dim=1)
    voxels_global_hardness_sampled = random_sampled_coords.view(-1, 3)[voxel_index.view(-1), :].view(B, int(omega * N),
                                                                                                     3)

    voxels_random_sampled = torch.rand(B, N - int(omega * N), 3, device=device)
    return torch.cat([voxels_global_hardness_sampled, voxels_random_sampled], 1).to(device)

