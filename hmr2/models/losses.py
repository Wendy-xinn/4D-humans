import torch
import torch.nn as nn

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')
        # ==================== 和 3D 完全一致的映射 ====================
        self.output_to_smpl = {
            8:  0,   
            12: 1,  
            9:  2,  
            29: 4,  
            26: 5,   
            30: 7,   
            25: 8,  
            1:  12, 
            34: 16,  
            33: 17,  
            35: 18,  
            32: 19, 
            36: 20,  
            31: 21,  
        }

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()

        # pred = pred_keypoints_2d.clone()
        # gt   = gt_keypoints_2d.clone()
        # # print("pred:", pred.shape)
        # # print("gt:", gt.shape)
        # gt = gt[..., :2]  # 去掉 conf

        # out_indices = list(self.output_to_smpl.keys())
        # smpl_indices = list(self.output_to_smpl.values())

        # pred_selected = pred[:, out_indices, :]      # [B, S, 14, 2]
        # gt_selected   = gt[:, smpl_indices, :]       # 

        # # 因为 GT 没有 confidence，默认 confidence = 1（不加权）
        # conf = torch.ones_like(gt_selected[..., :1])        # [B, S, 14, 1]

        # # ====================== 计算 2D 重投影损失 ======================
        # loss = (conf * self.loss_fn(pred_selected, gt_selected)).sum(dim=(1, 2))
        # return loss.sum()   # 返回标量 total loss


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()
    
# class Keypoint3DLoss(nn.Module):

#     def __init__(self, loss_type: str = 'l1'):
#         """
#         3D keypoint loss module（已适配你的 44→SMPL24 对齐）。
#         只对 JOINT_MAP 中能对应上的 14 个关节计算损失（这是最标准、最稳定的做法）。
#         """
#         super(Keypoint3DLoss, self).__init__()
#         if loss_type == 'l1':
#             self.loss_fn = nn.L1Loss(reduction='none')
#         elif loss_type == 'l2':
#             self.loss_fn = nn.MSELoss(reduction='none')
#         else:
#             raise NotImplementedError('Unsupported loss function')

#         # ==================== 关键：output(44) → SMPL(24) 映射 ====================
#         # 优先使用 GT 版（extra 19 个里的），质量更高
#         self.output_to_smpl = {
#             8:  0,   # OP MidHip
#             12: 1,   # OP LHip
#             9:  2,   # OP RHip
#             29: 4,   # Left Knee (GT版)
#             26: 5,   # Right Knee (GT版)
#             30: 7,   # Left Ankle (GT版)
#             25: 8,   # Right Ankle (GT版)
#             1:  12,  # OP Neck
#             34: 16,  # Left Shoulder (GT版)
#             33: 17,  # Right Shoulder (GT版)
#             35: 18,  # Left Elbow (GT版)
#             32: 19,  # Right Elbow (GT版)
#             36: 20,  # Left Wrist (GT版)
#             31: 21,  # Right Wrist (GT版)
#         }
#         self.pelvis_smpl_id = 0      # SMPL 24 个关节里，pelvis 永远是第 0 个
#         self.pelvis_output_idx = 39

#     def forward(self, 
#                 pred_keypoints_3d: torch.Tensor, 
#                 gt_keypoints_3d: torch.Tensor):
#         """
#         Args:
#             pred_keypoints_3d: [B, S, 44, 3]   ← 你的 model output（前44个关节点）
#             gt_keypoints_3d:   [B, S, 24, 3]   ← SMPL 24 个关节点 
#         Returns:
#             torch.Tensor: scalar loss
#         """

#         pred = pred_keypoints_3d.clone()   # [B, N, 3]
#         gt   = gt_keypoints_3d[..., :3].clone()   # [B, N, 3]
#         # print("pred:", pred.shape)
#         # print("gt:", gt.shape)

#         # ====================== 1. pelvis 对齐（完全照原逻辑） ======================
#         pred = pred - pred[:, self.pelvis_output_idx, :].unsqueeze(1)
#         gt   = gt   - gt[:, self.pelvis_smpl_id, :].unsqueeze(1)

#         # # ====================== 2. 只选对应关节 ======================
#         out_indices = list(self.output_to_smpl.keys())
#         smpl_indices = list(self.output_to_smpl.values())

#         pred_selected = pred[:, out_indices, :]   # [B, 14, 3]
#         gt_selected   = gt[:, smpl_indices, :]    # [B, 14, 3]

#         # ====================== 3. 没有 conf → 全1 ======================
#         conf = torch.ones_like(gt_selected[..., :1])  # [B, 14, 1]

#         # ====================== 4. 完全照原始 loss ======================
#         loss = (conf * self.loss_fn(pred_selected, gt_selected)).sum(dim=(1, 2))

#         return loss.sum()

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        return loss_param.sum()
