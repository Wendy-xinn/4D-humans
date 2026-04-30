import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder

def build_smpl_head(cfg):
    smpl_head_type = cfg.MODEL.SMPL_HEAD.get('TYPE', 'hmr')
    if  smpl_head_type == 'transformer_decoder':
        return SMPLTransformerDecoderHead(cfg)
    else:
        raise ValueError('Unknown SMPL head type: {}'.format(smpl_head_type))

class SMPLTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMPL_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        transformer_args = dict(
            num_tokens=2,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = (transformer_args | dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER))
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.SMPL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        # 初始值
        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

        # 注册 Ego 初始值
        # 1. 旋转：让 Ego 初始状态背对相机（绕Y轴转180度）
        init_ego_pose = init_body_pose.clone()
        if self.joint_rep_type == '6d':
            # 6D 旋转：取反前两个基向量实现约 180 度旋转
            init_ego_pose[:, :3] = -init_ego_pose[:, :3] 
        elif self.joint_rep_type == 'aa':
            # 轴角旋转：绕 Y 轴旋转 180 度 [0, pi, 0]
            init_ego_pose[:, 1] += np.pi
        
        # 2. 相机：Ego 就在相机后面，scale设大(1.0)，位移设在相机后方
        init_ego_cam = torch.tensor([[1.0, 0.0, 0.2]]) 

        self.register_buffer('init_ego_body_pose', init_ego_pose)
        self.register_buffer('init_ego_cam', init_ego_cam)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        init_body_pose2 = self.init_ego_body_pose.expand(batch_size, -1)
        init_betas2 = self.init_betas.expand(batch_size, -1)
        init_cam2 = self.init_ego_cam.expand(batch_size, -1)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        pred_body_pose2 = init_body_pose2
        pred_betas2 = init_betas2
        pred_cam2 = init_cam2
        pred_body_pose_list2 = []
        pred_betas_list2 = []
        pred_cam_list2 = []
        for i in range(self.cfg.MODEL.SMPL_HEAD.get('IEF_ITERS', 1)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token1 = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:,None,:]
                token2 = torch.cat([pred_body_pose2, pred_betas2, pred_cam2], dim=1)[:,None,:]
                token = torch.stack([token1, token2], dim=1)
            else:
                token = torch.zeros(batch_size, 2, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            # token_out = token_out.squeeze(1) # (B, C)
            token_out1 = token_out[:, 0, :]
            token_out2 = token_out[:, 1, :]

            # Readout from token_out
            pred_body_pose = self.decpose(token_out1) + pred_body_pose
            pred_betas = self.decshape(token_out1) + pred_betas
            pred_cam = self.deccam(token_out1) + pred_cam
            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)
            pred_body_pose2 = self.decpose(token_out2) + pred_body_pose2
            pred_betas2 = self.decshape(token_out2) + pred_betas2
            pred_cam2 = self.deccam(token_out2) + pred_cam2
            pred_body_pose_list2.append(pred_body_pose2)
            pred_betas_list2.append(pred_betas2)
            pred_cam_list2.append(pred_cam2)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smpl_params_list = {}
        pred_smpl_params_list['body_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0)
        pred_smpl_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, self.cfg.SMPL.NUM_BODY_JOINTS+1, 3, 3)

        pred_smpl_params = {'global_orient': pred_body_pose[:, [0]],
                            'body_pose': pred_body_pose[:, 1:],
                            'betas': pred_betas}
        pred_smpl_params_list2 = {}
        pred_smpl_params_list2['body_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list2], dim=0)
        pred_smpl_params_list2['betas'] = torch.cat(pred_betas_list2, dim=0)
        pred_smpl_params_list2['cam'] = torch.cat(pred_cam_list2, dim=0)
        pred_body_pose2 = joint_conversion_fn(pred_body_pose2).view(batch_size, self.cfg.SMPL.NUM_BODY_JOINTS+1, 3, 3)

        pred_smpl_params2 = {'global_orient': pred_body_pose2[:, [0]],
                            'body_pose': pred_body_pose2[:, 1:],
                            'betas': pred_betas2}
        return pred_smpl_params, pred_cam, pred_smpl_params_list, pred_smpl_params2, pred_cam2, pred_smpl_params_list2
