import torch
import pytorch_lightning as pl
import numpy as np
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer, eval_pose
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smpl_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import SMPL
from hmr2.configs import CACHE_DIR_4DHUMANS
DEFAULT_CHECKPOINT=f'{CACHE_DIR_4DHUMANS}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'

log = get_pylogger(__name__)

class HMR2(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HMR2 model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        self.is_ego = self.cfg.MODEL.SMPL_HEAD.get('IS_EGO', 'false') 
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        self.smpl_head = build_smpl_head(cfg, is_ego='false')
        self.smpl_head_ego = build_smpl_head(cfg, is_ego='true')
        # if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
        #     log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
        #     self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])
        if self.cfg.MODEL.BACKBONE.load_pretrained:
            ckpt = torch.load(DEFAULT_CHECKPOINT, map_location='cpu')
            state_dict = ckpt['state_dict']  # Lightning checkpoint
            vit_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    vit_state_dict[k.replace('backbone.', '')] = v
            self.backbone.load_state_dict(vit_state_dict, strict=False)
            head_dict = {k.replace('smpl_head.', ''): v for k,v in state_dict.items() if k.startswith('smpl_head.')}
            self.smpl_head.load_state_dict(head_dict, strict=True)
            self.smpl_head_ego.load_state_dict(head_dict, strict=False)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        for p in self.smpl_head.parameters():
            p.requires_grad = False
        self.smpl_head.eval()

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()

        # Instantiate SMPL model
        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        # Buffer that shows whether we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smpl.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.smpl_head.parameters())
        all_params += list(self.smpl_head_ego.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        optimizers = []
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizers.append(optimizer)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                                lr=self.cfg.TRAIN.LR,
                                                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
            optimizers.append(optimizer_disc)

        return optimizers

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        pred_smpl_params_ego, pred_cam_ego, _ = self.smpl_head_ego(conditioning_feats)
        pred_smpl_params, pred_cam, _ = self.smpl_head(conditioning_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['pred_cam_ego'] = pred_cam_ego
        output['pred_smpl_params_ego'] = {k: v.clone() for k,v in pred_smpl_params_ego.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        focal_length_ego = self.cfg.EXTRA.FOCAL_LENGTH_EGO * torch.ones(batch_size, 2, device=device, dtype=dtype)   # 这里不确定
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length
        # 计算 Ego 的位移 (pred_cam_t_ego)
        pred_cam_t_ego = torch.stack([pred_cam_ego[:, 1], pred_cam_ego[:, 2], 
                                    2*focal_length_ego[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam_ego[:, 0] +1e-9)], dim=-1)
        output['pred_cam_t_ego'] = pred_cam_t_ego
        output['focal_length_ego'] = focal_length_ego

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        pred_smpl_params_ego['global_orient'] = pred_smpl_params_ego['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params_ego['body_pose'] = pred_smpl_params_ego['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params_ego['betas'] = pred_smpl_params_ego['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        smpl_output_ego = self.smpl(**{k: v.float() for k,v in pred_smpl_params_ego.items()}, pose2rot=False)
        pred_keypoints_3d_ego = smpl_output_ego.joints
        pred_vertices_ego = smpl_output_ego.vertices
        output['pred_keypoints_3d_ego'] = pred_keypoints_3d_ego.reshape(batch_size, -1, 3)
        output['pred_vertices_ego'] = pred_vertices_ego.reshape(batch_size, -1, 3)
        # output['pred_keypoints_2d_ego'] = torch.zeros((batch_size, 44, 2), device=device)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                translation=pred_cam_t,
                                                focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        output['pred_keypoints_2d_ego'] = torch.zeros_like(output['pred_keypoints_2d'])

        return output

    def compute_bone_direction_loss(self, pred_j3d, gt_j3d):
        """
        针对你提供的 JOINT_MAP 索引：
        1: OP LHip, 4: OP LKnee, 7: OP LAnkle
        2: OP RHip, 5: OP RKnee, 8: OP RAnkle
        0: OP MidHip, 12: OP Neck
        """
        # 定义关键骨骼对 (起始点, 结束点)
        bone_pairs = [
            (1, 4), (4, 7),   # 左腿: 胯 -> 膝, 膝 -> 踝
            (2, 5), (5, 8),   # 右腿: 胯 -> 膝, 膝 -> 踝
            (0, 12),          # 躯干: 盆骨 -> 脖子
        ]
        
        gt_j3d = gt_j3d[:, :, :3]  # gt多一维置信度
        direction_loss = 0
        for p1, p2 in bone_pairs:
            # 计算预测和 GT 的骨骼向量
            vec_pred = pred_j3d[:, p2] - pred_j3d[:, p1]
            vec_gt = gt_j3d[:, p2] - gt_j3d[:, p1]
            
            # 使用余弦相似度强制方向对齐
            # cos=1 表示方向完全一致，loss=0
            cos_sim = torch.nn.functional.cosine_similarity(vec_pred, vec_gt, dim=-1)
            direction_loss += (1 - cos_sim).mean()
            
        return direction_loss
    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        prefix = '_ego' if self.is_ego else ''

        pred_smpl_params = output[f'pred_smpl_params{prefix}']
        pred_keypoints_2d = output[f'pred_keypoints_2d{prefix}']           #[:, :24, :]   # 改成只取前24个
        pred_keypoints_3d = output[f'pred_keypoints_3d{prefix}']           #[:, :24, :]   # 改成只取前24个
        # print("pred", pred_smpl_params["global_orient"])
        # print("gt", batch['smpl_params']['global_orient'])

        batch_size = pred_smpl_params['body_pose'].shape[0]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        # is_axis_angle = batch['smpl_params_is_axis_angle']
        # print("gt3d", gt_keypoints_3d)
        # print("gt2d", gt_keypoints_2d)
        # print("pre3d", pred_keypoints_3d)
        # print("pre2d", pred_keypoints_2d)

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14)
        # loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d)
        # loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        loss_bone = self.compute_bone_direction_loss(pred_keypoints_3d, gt_keypoints_3d)
        w_bone = self.cfg.LOSS_WEIGHTS.get('BONE_DIRECTION', 1.0)

        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(batch_size, -1)
            # if is_axis_angle[k].all():
            if k in ["global_orient", "body_pose"]:
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                # print('gt_rot', gt)
            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)
        
        w_2d = 0.0 if self.is_ego else self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D']
        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d + \
               w_2d * loss_keypoints_2d + \
               w_bone * loss_bone + \
               sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])

        losses = {
            f'loss{prefix}': loss.detach(),
            f'loss_keypoints_3d{prefix}': loss_keypoints_3d.detach(),
            f'loss_bone_{prefix}': loss_bone.detach(),
        }
        if not self.is_ego:
            losses['loss_keypoints_2d'] = loss_keypoints_2d.detach()

        for k, v in loss_smpl_params.items():
            losses[f'loss_{k}{prefix}'] = v.detach()

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_image=False, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['img'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        pred_vertices_ego = output['pred_vertices_ego'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        focal_length_ego = output['focal_length_ego'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_cam_t_ego = output['pred_cam_t_ego'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        if write_image:
            predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                                pred_cam_t[:num_images].cpu().numpy(),
                                                                images[:num_images].cpu().numpy(),
                                                                pred_keypoints_2d[:num_images].cpu().numpy(),
                                                                gt_keypoints_2d[:num_images].cpu().numpy(),
                                                                focal_length=focal_length[:num_images].cpu().numpy())
            predictions_ego = self.mesh_renderer.visualize_tensorboard_ego(pred_vertices_ego[:num_images].cpu().numpy(),
                                                                batch['vertices'].detach().cpu().numpy(),
                                                                pred_cam_t_ego[:num_images].cpu().numpy(),
                                                                images[:num_images].cpu().numpy(),
                                                                focal_length_ego=focal_length_ego[:num_images].cpu().numpy())
            combined_predictions = torch.cat([predictions, predictions_ego], dim=2)
            if write_to_summary_writer:
                summary_writer.add_image('%s/predictions' % mode, combined_predictions, step_count)

            return combined_predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch = joint_batch['img']
        # mocap_batch = joint_batch['mocap']
        batch = joint_batch
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True, write_image=False)
        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.VIS_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True, write_image=True)

        prefix = '_ego' if self.is_ego else ''
        loss_key = f'loss{prefix}'
        self.log(f'train/{loss_key}', output['losses'][loss_key], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False, write_image=True)
        # pred_keypoints_3d = output['pred_keypoints_3d'].detach()
        # pred_keypoints_3d = pred_keypoints_3d[:,None,:,:]
        # batch_size = pred_keypoints_3d.shape[0]
        # num_samples = pred_keypoints_3d.shape[1]
        # gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)

        # # Align predictions and ground truth such that the pelvis location is at the origin
        # pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.cfg.EXTRA.PELVIS_IND]]
        # gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.cfg.EXTRA.PELVIS_IND]]
        # keypoint_list = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43]
        # # EVAL_JOINT_MAP = {
        # #     8: 0, 12: 1, 9: 2, 29: 4, 26: 5, 30: 7, 25: 8,
        # #     1: 12, 34: 16, 33: 17, 35: 18, 32: 19, 36: 20, 31: 21
        # # }
        # # PRED_EVAL_IDX = list(EVAL_JOINT_MAP.keys())   # [8, 12, 9, 29, 26, 30, 25, 1, 34, 33, 35, 32, 36, 31]
        # # GT_EVAL_IDX   = list(EVAL_JOINT_MAP.values()) # [0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
        # # # 1. 索引对齐 → 统一为 [B, 14, 3]
        # # pred_sel = pred_keypoints_3d[:, PRED_EVAL_IDX, :]
        # # gt_sel   = gt_keypoints_3d[:, GT_EVAL_IDX, :]
        
        # # # 2. 调用你原有的 eval_pose（内部会做 Procrustes 对齐）
        # mpjpe, pa_mpjpe = eval_pose(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3)[:, self.keypoint_list], gt_keypoints_3d.reshape(batch_size * num_samples, -1 ,3)[:, keypoint_list])
        # mpjpe = mpjpe.reshape(batch_size, num_samples)
        # pa_mpjpe = pa_mpjpe.reshape(batch_size, num_samples)
        # batch_mpjpe = mpjpe.mean(axis=1)      
        # batch_pa_mpjpe = pa_mpjpe.mean(axis=1) 
        # mean_mpjpe = float(np.mean(batch_mpjpe))
        # mean_pa_mpjpe = float(np.mean(batch_pa_mpjpe))

        # # Compute 2d keypoint errors
        # pred_keypoints_2d = output['pred_keypoints_2d'].detach()
        # pred_keypoints_2d = pred_keypoints_2d[:,None,:,:]
        # gt_keypoints_2d = batch['keypoints_2d'][:,None,:,:].repeat(1, num_samples, 1, 1)
        # conf = gt_keypoints_2d[:, :, :, -1].clone()
        # kp_err = torch.nn.functional.mse_loss(
        #                 pred_keypoints_2d,
        #                 gt_keypoints_2d[:, :, :, :-1],
        #                 reduction='none'
        #             ).sum(dim=3)
        # kp_l2_loss = (conf * kp_err).mean(dim=2)
        # batch_kp_l2 = kp_l2_loss.mean(dim=1)       # [B]
        # mean_kp_l2 = float(batch_kp_l2.mean().cpu().numpy())
        
        # self.log('val/mpjpe', mean_mpjpe, sync_dist=True, on_epoch=True, prog_bar=True)
        # self.log('val/pa_mpjpe', mean_pa_mpjpe, sync_dist=True, on_epoch=True, prog_bar=True)
        # self.log('val/kp2d_l2', mean_kp_l2, sync_dist=True, on_epoch=True, prog_bar=True)

        return output
