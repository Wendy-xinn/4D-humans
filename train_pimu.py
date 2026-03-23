import os
import time
import torch
torch.set_float32_matmul_precision('high')
import argparse
import numpy as np
import hydra
from glob import glob
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as checkpoint
import pickle
import pandas as pd
import cv2
from smplx.lbs import batch_rodrigues
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from hmr2.utils import recursive_to
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models.backbones import create_backbone
from hmr2.models.heads import build_smpl_head
from hmr2.models.discriminator import Discriminator
from hmr2.models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from hmr2.models import SMPL
from hmr2.utils import SkeletonRenderer, MeshRenderer
from hmr2.utils.geometry import perspective_projection, aa_to_rotmat
from hmr2.datasets.utils import (
    generate_image_patch_cv2,
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
)
from skimage.filters import gaussian
from yacs.config import CfgNode
import pytorch_lightning as pl
import copy
import pyrootutils
from torchvision import transforms
root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )

LIGHT_BLUE = (0.65, 0.74, 0.86)

class ImageIMUDataset(Dataset):
    def __init__(self, image_root,  pt_root, cfg):
        self.image_root = os.path.abspath(image_root)
        self.pt_root = pt_root
        self.cfg = cfg
        self.img_size = tuple(cfg.MODEL.IMAGE_SIZE)
        # self.img_size = (192, 256)  # (W, H)
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std  = 255. * np.array(cfg.MODEL.IMAGE_STD)

        extensions = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}

        self.samples = []
        self._pt_cache = {}

        # 遍历 image_root/s*/action*_cam*/
        subjects = sorted(os.listdir(self.image_root))
        for subject in subjects:
            subj_dir = os.path.join(self.image_root, subject)
            if not os.path.isdir(subj_dir):
                continue

            for action_cam in sorted(os.listdir(subj_dir)):
                img_dir = os.path.join(subj_dir, action_cam)
                if not os.path.isdir(img_dir):
                    continue
                # action 名（去掉 cam）
                action = "_".join(action_cam.split("_")[:-1])
                key = (subject, action)

                pt_file = os.path.join(self.pt_root, f"{subject}_{action}.pt")
                if not os.path.exists(pt_file):
                    print(f"[WARN] missing pt: {pt_file}")
                    continue
                # 读取 pt（一次）
                data = torch.load(pt_file, map_location="cpu")
                N_pt = data["acc"].shape[0]

                # 收集图像
                images = sorted([
                    p for p in glob(os.path.join(img_dir, "*"))
                    if os.path.splitext(p)[1][1:].lower() in extensions
                ])

                if len(images) == 0:
                    continue

                # 统一长度（核心）
                N = min(len(images), N_pt)

                for i in range(N):
                    self.samples.append({
                        "img": images[i],
                        # "subject": subject,
                        # "action": action,
                        "pt_path": pt_file,
                        "frame_idx": i
                    })

        print(f"[Dataset] total samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError("No valid image–IMU pairs found")

    def __len__(self):
        return len(self.samples)

    def _load_pt(self, pt_path):
        if pt_path not in self._pt_cache:
            self._pt_cache[pt_path] = torch.load(pt_path, map_location="cpu")
        return self._pt_cache[pt_path]
    def __getitem__(self, idx):
        s = self.samples[idx]

        data = self._load_pt(s["pt_path"])
        k = s["frame_idx"]
        
         # ---------- load image ----------
        cvimg = cv2.imread(s["img"])
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

        H, W, _ = cvimg.shape

        # ========= 1. fake bbox（整图）=========
        center_x = W / 2.0
        center_y = H / 2.0
        bbox_size = max(H, W)

        # ========= 2. anti-aliasing（和 ViTDet 一样）抗锯齿处理=========
        input_size = min(self.img_size)   # scalar
        downsampling_factor = bbox_size / input_size  / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(
                cvimg,
                sigma=(downsampling_factor - 1) / 2,
                channel_axis=2,
                preserve_range=True
            )

        # ========= 3. crop + resize =========
        img_patch_cv, _ = generate_image_patch_cv2(
            cvimg,
            center_x, center_y,
            bbox_size, bbox_size,
            # self.img_size[0], self.img_size[1],
            256, 192,
            False, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT
        )

        # BGR → RGB（generate_image_patch 用的是 cv2）
        img_patch_cv = img_patch_cv[:, :, ::-1]

        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # ========= 4. normalization =========
        for c in range(3):
            img_patch[c] = (img_patch[c] - self.mean[c]) / self.std[c]

        # -------- imu --------
        acc = data["acc"][k]        # (6, 3)
        ori = data["ori"][k]        # (6, 3, 3)

        
        # -------- gt --------
        pose = data["pose"][k]
        tran = data["tran"][k]
        

        return {
            "image": img_patch,
            "imu_acc": acc.float(),
            "imu_ori": ori.float(),
            "gt_pose": pose.float(),
            "gt_tran": tran.float(),
        }

class ImageIMUDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_set = ImageIMUDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.TRAIN_PT_ROOT,
            self.cfg
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
        )
        


class IMUProjector(nn.Module):
    def __init__(self, imu_dim=72, embed_dim=1280, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.imu_dim = imu_dim

        self.proj = nn.Sequential(
            nn.Linear(imu_dim, embed_dim // 2),
            nn.ReLU(),  
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * num_tokens)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, imu_data):
        if imu_data is None or imu_data.numel() == 0:
            return None
        if imu_data.dim() == 3:
            imu_data = imu_data.squeeze(1)
        assert imu_data.shape[1] == self.imu_dim, \
            f"Expected imu_dim={self.imu_dim}, but got {imu_data.shape}"

        imu_data = imu_data.to(dtype=self.proj[0].weight.dtype)
        x = self.proj(imu_data)
        x = x.view(-1, self.num_tokens, self.embed_dim)
        return self.norm(x)
    

class HMR2WithIMU(pl.LightningModule):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True, load_pretrained=True):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['init_renderer', 'load_pretrained'])
        self.cfg = cfg
        self.cam_translation = torch.tensor(cfg.MODEL.CAMERA.TRANSLATION, dtype=torch.float32)
        self.cam_focal = torch.tensor(cfg.MODEL.CAMERA.FOCAL_LENGTH, dtype=torch.float32)
        self.cam_center = torch.tensor(cfg.MODEL.CAMERA.CENTER, dtype=torch.float32)
        self.cam_rotation = torch.tensor(cfg.MODEL.CAMERA.ROTATION, dtype=torch.float32)

        self.backbone_ori = create_backbone(cfg) # 这个参数要冻结
        self.backbone_copy = create_backbone(cfg)
        self.IMUEncoder = IMUProjector(cfg.MODEL.IMU.INPUT_DIM, cfg.MODEL.IMU.EMBED_DIM)
        
        # Zero-initialize IMUEncoder parameters
        for param in self.IMUEncoder.parameters():
            nn.init.zeros_(param)

        # ViT 有多少个 Block，我们就需要多少个 Zero-Conv
        num_blocks = len(self.backbone_ori.blocks)
        embed_dim = self.backbone_ori.embed_dim
        self.zero_convs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_blocks)
        ])
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

        self.fusion_mlp_linear = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.fusion_mlp_linear.weight)
        nn.init.zeros_(self.fusion_mlp_linear.bias)

        if load_pretrained:
            ckpt = torch.load(DEFAULT_CHECKPOINT, map_location='cpu')
            state_dict = ckpt['state_dict']  # Lightning checkpoint
            vit_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    vit_state_dict[k.replace('backbone.', '')] = v
            self.backbone_ori.load_state_dict(vit_state_dict, strict=False)
            self.backbone_copy = copy.deepcopy(self.backbone_ori)  
            
        for p in self.backbone_ori.parameters():
            p.requires_grad = False
        self.backbone_ori.eval()
        for p in self.backbone_copy.parameters():
            p.requires_grad = True  # copy 是可训练的
            
        # Create SMPL head
        self.smpl_head = build_smpl_head(cfg)
        for p in self.smpl_head.parameters():
            p.requires_grad = False
        self.smpl_head.eval()
        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

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


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone_copy.parameters()},
                {"params": self.IMUEncoder.parameters()},
                {"params": self.fusion_mlp_linear.parameters()},
                {"params": self.zero_convs.parameters()},
            ],
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer

    def forward(self, images, imu, gt_pose, gt_tran):
        """
        images: [B, 3, H, W]
        imu:    [B, imu_dim]
        """

        # resize = Resize((256, 192))
        # images = resize(images)

        B = images.shape[0]


        # ===== 1. backbone_ori：reference visual tokens =====
        with torch.no_grad():
            # feat_ori = self.backbone_ori.forward_features(images)
            # ViT: [B, N, C]
            x, (Hp, Wp) = self.backbone_ori.patch_embed(images)  # [B, N, C]

            if self.backbone_ori.pos_embed is not None:
                x = x + self.backbone_ori.pos_embed[:, 1:] # + self.backbone_ori.pos_embed[:, :1]    # 加上位置编码

            for blk in self.backbone_ori.blocks:
                if self.backbone_ori.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            feat_ori = self.backbone_ori.last_norm(x)  # [B, N, C]

        B, N, C = feat_ori.shape
        # ===== 2. IMU token =====
        imu_token = self.IMUEncoder(imu)        # [B, 4(num_tokens), C]

        # ControlNet 标准做法：token → patch-wise control
        imu_control = imu_token.mean(dim=1, keepdim=True)   # [B, 1, C]
        # imu_control = imu_control.expand(B, N, C)             # [B, N, C]

        # ===== 3. ControlNet-style：逐层注入 =====
        # 利用广播机制，不需要显式 expand (省显存)
        tokens = feat_ori
        for i, blk in enumerate(self.backbone_copy.blocks):
            tokens = blk(tokens)
            tokens = tokens + self.zero_convs[i](imu_control)

        feat_fuse = self.backbone_copy.last_norm(tokens)

        delta = self.fusion_mlp_linear(feat_fuse)  # nn.Linear(C, C)

        # ===== Residual add (ControlNet core) =====
        fused_features = feat_ori + delta
        B, N, C = fused_features.shape
        # H_w = W_w = int(N**0.5)
        # if H_w*W_w != N:
        #     # pad tokens to nearest square
        #     pad = H_w*W_w - N
        #     fused_features = F.pad(fused_features, (0,0,0,pad), value=0)
        # x_smpl = fused_features.permute(0,2,1).reshape(B, C, H_w, W_w)
        x_smpl = fused_features.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        pred_smpl_params, pred_cam, _ = self.smpl_head(x_smpl) 
        batch_size = images.shape[0]
        # Store useful regression outputs to the output dict
        output = {}

        # print(f"gt_pose shape: {gt_pose.shape}, gt_tran shape: {gt_tran.shape}")
        gt_shape = torch.zeros(gt_pose.shape[0], 10, device=gt_pose.device)
        pose_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(gt_pose.shape[0], 24, 3, 3)
        smpl_output = self.smpl(
            betas=gt_shape,         # [B, 10]
            body_pose=pose_rotmat[:, 1:],  # [B, 23, 3, 3]
            global_orient=pose_rotmat[:, :1],  # [B, 1, 3, 3]
            transl=gt_tran          # [B, 3]
        )
        gt_keypoints_3d = smpl_output.joints  # [B, J, 3]
        # print(f"gt_keypoints_3d shape: {gt_keypoints_3d.shape}")
        # print(f"gt_keypoints_3d: {gt_keypoints_3d}")
        output['gt_keypoints_3d'] = gt_keypoints_3d
        device = gt_pose.device
        # gt_keypoints_2d = perspective_projection(gt_keypoints_3d,
        #                        translation=self.cam_translation.to(device).unsqueeze(0).expand(batch_size, -1),  # [B, 3]
        #                        focal_length=self.cam_focal.to(device).unsqueeze(0).expand(batch_size, -1),  # [B, 2]
        #                        camera_center=self.cam_center.to(device).unsqueeze(0).expand(batch_size, -1),  # [B, 2]
        #                        rotation=self.cam_rotation.to(device).unsqueeze(0).expand(batch_size, -1, -1),  # [B, 3, 3]
        #                  )
       
        # # 归一化
        # img_w, img_h = self.cfg.MODEL.IMAGE_SIZE[0], self.cfg.MODEL.IMAGE_SIZE[1]
        # gt_keypoints_2d[..., 0] = gt_keypoints_2d[..., 0] / img_w - 0.5
        # gt_keypoints_2d[..., 1] = gt_keypoints_2d[..., 1] / img_h - 0.5
        
        # print(f"gt_keypoints_3d shape: {gt_keypoints_3d.shape}, gt_keypoints_2d shape: {gt_keypoints_2d.shape}")
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                      pred_cam[:, 2],
                      2*focal_length[:, 0]/(self.cfg.MODEL.VIT_IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length
        gt_keypoints_2d = perspective_projection(gt_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.VIT_IMAGE_SIZE)
        # print(f"gt_keypoints_2d shape: {gt_keypoints_2d.shape}")
        # print(f"gt_keypoints_2d: {gt_keypoints_2d}")

        output['gt_keypoints_2d'] = gt_keypoints_2d
        # gt的相机参数
        # min_row, max_row, min_col, max_col    0 1079 0 1919
        # fx, fy, cx, cy    1284.32 1286.38 959.5 539.5
        # distortion params   1.40869e-05
        # 3x3 Rotation matrix R    [-0.99713 0.00504186 -0.0755413,  0.0221672 -0.93461 -0.354982, -0.0723915 -0.355637 0.931816 ] 
        # 3x1 translation t      0.820506 0.59704 5.33591 
        # (Such that a world point p in the camera coordinate frame is given by p' = Rp + t) 
        # (Such that a project point for a perfect pinhole camera with no distortion is u = fx* p'_x/p'_zworld point in the camera coordinate frame is given by p' = Rp + t) 
         
        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        # print(f"pred_keypoints_3d shape: {output['pred_keypoints_3d'].shape}")
        # print(f"pred_keypoints_3d: {output['pred_keypoints_3d']}")

        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.VIT_IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        # print(f"pred_keypoints_2d shape: {output['pred_keypoints_2d'].shape}")
        # print(f"pred_keypoints_2d: {output['pred_keypoints_2d']}")
        # 可视化gt
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # img = images[0].cpu().numpy().transpose(1, 2, 0) 
        # keypoints_2d = output['gt_keypoints_2d'][0].detach().cpu().numpy() 
        # plt.imshow(img.astype('uint8'))
        # plt.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='r', s=10)
        # plt.savefig("debug_2d.png")
        # plt.close()
        # keypoints_3d = output['gt_keypoints_3d'][0].detach().cpu().numpy() 
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], c='b', s=10)
        # plt.savefig("debug_3d.png")
        # plt.close()
        
        return output

    def render_prediction_only(self, batch, output, max_items=1):
        images = batch["image"][:max_items]           # [B, 3, H, W]
        verts  = output["pred_vertices"][:max_items]  # [B, V, 3]
        # cam_t  = output["pred_cam_t"][:max_items]     # [B, 3]
        
        # 获取焦距
        focal_length = output['focal_length']
        if torch.is_tensor(focal_length):
            # 如果是 [B, 2] 形状，先取第一个样本，再取第一个分量 (fx)
            if focal_length.dim() == 2:
                focal_length = focal_length[0, 0].item()
            # 如果是 [2] 形状，直接取第一个分量
            elif focal_length.dim() == 1:
                focal_length = focal_length[0].item()
            else:
                focal_length = focal_length.item()

        # 反标准化图像用于显示
        mean = torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(3, 1, 1).to(images.device)
        std = torch.tensor(self.cfg.MODEL.IMAGE_STD).view(3, 1, 1).to(images.device)
        images_unnorm = images * std + mean
        
        images_np = images_unnorm.permute(0, 2, 3, 1).cpu().numpy()
        verts_np  = verts.detach().cpu().numpy()
        cam_t = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        combined_vis = []

        for i in range(len(images_np)):
            img_overlay = self.mesh_renderer(
                vertices=verts_np[i],
                camera_translation=cam_t,   # ✅ 每个样本同一个固定相机
                image=images_np[i],
                focal_length=focal_length,
                side_view=False
            )

            img_side = self.mesh_renderer(
                vertices=verts_np[i],
                camera_translation=cam_t,
                image=images_np[i],
                focal_length=focal_length,
                side_view=True
            )

            viz_combined = np.concatenate(
                [images_np[i], img_overlay, img_side],
                axis=1
            )

            combined_vis.append(
                torch.from_numpy(viz_combined).permute(2,0,1)
            )

        return torch.stack(combined_vis) # [B, 3, H, W*3]

    def simple_mesh_vis(self, batch, output, max_items=1):
        images = batch["image"][:max_items]          # [B,3,H,W]
        verts  = output["pred_vertices"][:max_items] # [B,V,3]

        # unnormalize
        mean = torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=images.device).view(3,1,1)
        std  = torch.tensor(self.cfg.MODEL.IMAGE_STD,  device=images.device).view(3,1,1)
        images = images * std + mean
        images = images.permute(0,2,3,1).cpu().numpy()

        verts = verts.detach().cpu().numpy()

        vis_out = []

        for i in range(len(images)):
            v = verts[i].copy()

            # --- 1. 居中人体 ---
            v -= v.mean(0)

            # --- 2. 固定缩放 ---
            scale = 1.8 / (v[:,1].max() - v[:,1].min())
            v *= scale

            # --- 3. 投影到2D (正交) ---
            h, w, _ = images[i].shape
            proj = v[:, :2]
            proj[:,0] = proj[:,0] * w/2 + w/2
            proj[:,1] = -proj[:,1] * h/2 + h/2

            # --- 4. rasterize (最简单点云) ---
            canvas = images[i].copy()

            for x,y in proj.astype(int):
                if 0 <= x < w and 0 <= y < h:
                    canvas[y, x] = [1, 0, 0]  # 红色mesh点

            vis = np.concatenate([images[i], canvas], axis=1)
            vis_out.append(torch.from_numpy(vis).permute(2,0,1))

        return torch.stack(vis_out)

    
    def compute_loss(self, batch, output, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        losses = {}

        pred_smpl_params = output['pred_smpl_params']
        batch_size = pred_smpl_params['body_pose'].shape[0]

        # gt_pose: (B, 24, 3)
        gt_smpl_params = batch['gt_pose']
        # (B*24, 3) → (B*24, 3, 3)
        gt_pose_rotmat = aa_to_rotmat(gt_smpl_params.view(-1, 3)).view(batch_size, 24, 3, 3)
        gt_global_orient = gt_pose_rotmat[:, 0:1]    # (B, 1, 3, 3)
        gt_body_pose     = gt_pose_rotmat[:, 1:24]   # (B, 23, 3, 3)
        pred_global_orient = pred_smpl_params["global_orient"]  # (B, 1, 3, 3)
        pred_body_pose     = pred_smpl_params["body_pose"]      # (B, 23, 3, 3)

        # 全局旋转损失（根节点）
        losses['root_ori_loss'] = F.mse_loss(
            pred_global_orient,
            gt_global_orient,
            reduction="mean"
        )

        # smpl pose损失
        losses['pose_loss'] = F.mse_loss(
            pred_body_pose,
            gt_body_pose,
            reduction="mean"
        )

        # 2D/3D关键点损失
        losses['keypoints_3d_loss'] = F.mse_loss(
            output['pred_keypoints_3d'],
            output['gt_keypoints_3d'],
            reduction="mean"
        )
        losses['keypoints_2d_loss'] = F.mse_loss(
            output['pred_keypoints_2d'],
            output['gt_keypoints_2d'],
            reduction="mean"
        )

        # 总损失
        total_loss = (losses['root_ori_loss'] * self.cfg.LOSS_WEIGHTS.GLOBAL_ORIENT +
                      losses['pose_loss'] * self.cfg.LOSS_WEIGHTS.BODY_POSE +
                      losses['keypoints_3d_loss'] * self.cfg.LOSS_WEIGHTS.KEYPOINTS_3D +
                      losses['keypoints_2d_loss'] * self.cfg.LOSS_WEIGHTS.KEYPOINTS_2D
                    )
        losses['total_loss'] = total_loss
        return losses
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        imu_acc = batch["imu_acc"]
        imu_ori = batch["imu_ori"]

        imu = torch.cat([
            imu_acc.view(images.size(0), -1),
            imu_ori.view(images.size(0), -1),
        ], dim=1)
        gt_pose = batch["gt_pose"]
        gt_tran = batch["gt_tran"]
        batch_size = batch['image'].shape[0]
        optimizer = self.optimizers(use_pl_optimizer=True)

        output = self.forward(images, imu, gt_pose, gt_tran)
        pred_smpl_params = output['pred_smpl_params']

        losses = self.compute_loss(batch, output, train=True)
        loss = losses['total_loss']

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        self.log("train/loss_pose", losses['pose_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_root_ori", losses['root_ori_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_keypoints_3d", losses['keypoints_3d_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_keypoints_2d", losses['keypoints_2d_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/total_loss", losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        summary_writer = self.logger.experiment
        for loss_name, val in losses.items():
            summary_writer.add_scalar('train/' + loss_name, val.detach().item(), self.global_step)

        num_images = min(2, batch_size)  # 最多可视化4张图
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)
        gt_keypoints_2d = output['gt_keypoints_2d']
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        # 每 N step 可视化一次
        if self.global_step % self.cfg.GENERAL.VIS_STEPS == 0:
            if self.mesh_renderer is not None:
                with torch.no_grad():
                    # print(gt_keypoints_2d.shape, pred_keypoints_2d.shape, pred_cam_t.shape, focal_length.shape)
                    # print(gt_keypoints_2d)
                    # print(pred_keypoints_2d)
                    print(pred_keypoints_2d.min(), pred_keypoints_2d.max())
                    print(gt_keypoints_2d.min(), gt_keypoints_2d.max())
                    predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
                    # 写入tensorboard
                    if hasattr(self.logger, "experiment"):
                        self.logger.experiment.add_image(
                            'train/predictions', predictions, self.global_step
                        )
            # if self.skeleton_renderer is not None:
            #     vis = self.skeleton_renderer(pred_keypoints_3d=output['pred_keypoints_3d'],
            #                                 images=batch['image'].detach().cpu().numpy())
            #     self.logger.experiment.add_images("train/skeleton_pred", torch.tensor(vis).permute(2,0,1), self.global_step)
        return loss
def train(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = HMR2WithIMU(cfg)
    model.to(device)
    datamodule = ImageIMUDataModule(cfg)
    
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(cfg.paths.output_dir, "tensorboard"),
        name="",
        version="",
    )
    loggers = [logger]
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor="train/total_loss",           # 保存最优模型的指标
    )
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
    ]

    trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers, 
    )

    # trainer.fit(model, datamodule=datamodule, ckpt_path="last")
    trainer.fit(model, datamodule=datamodule)



@hydra.main(version_base="1.2", config_path=str(root/"hmr2/configs_hydra"), config_name="train.yaml")
def main(cfg):
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()

