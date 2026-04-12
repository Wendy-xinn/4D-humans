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
from hmr2.datasets.utils import get_example, expand_to_aspect_ratio
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
body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

class ImageIMUDataset(Dataset):
    def __init__(self, image_root,  pt_root, cfg, train = True):
        self.image_root = os.path.abspath(image_root)
        self.pt_root = pt_root
        self.cfg = cfg
        self.train = train

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std  = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        # 读取所有 pt 文件
        pt_files = sorted(glob(os.path.join(pt_root, "*.pt")))
        self.samples = []
        self._pt_cache = {}

        for pt_file in pt_files:
            data = torch.load(pt_file, map_location="cpu")
            N = data["poses"].shape[0]
            for k in range(N):
                self.samples.append({
                    "pt_path": pt_file,
                    "frame_idx": k,
                    "sequence_name": data.get("sequence_name")
                })
                    

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found")
        print(f"[Dataset] total samples: {len(self.samples)}")
       

    def __len__(self):
        return len(self.samples)

    # def project_smpl_to_2d(self, joints_3d):
    #     """
    #     joints_3d: [J, 3] (SMPL 输出的世界坐标)
    #     """
    #     # 1. 统一维度为 Batch 模式 [B=1, J, 3]
    #     points = joints_3d.unsqueeze(0) 
    #     batch_size = 1
    #     device = points.device
    #     dtype = points.dtype

    #     # 2. 准备外参 (R, t)
    #     # 按照官方公式 p' = Rp + t，rotation 对应 R，translation 对应 t
    #     rotation = self.R.to(device).unsqueeze(0)     # [1, 3, 3]
    #     translation = self.t.to(device).unsqueeze(0)  # [1, 3]

    #     # 3. 准备内参 (K)
    #     focal_length = torch.tensor([[self.fx, self.fy]], device=device, dtype=dtype) # [1, 2]
    #     camera_center = torch.tensor([[self.cx, self.cy]], device=device, dtype=dtype) # [1, 2]

    #     # --- 开始执行透视投影逻辑 ---

    #     # A. 坐标变换 (World -> Camera): points = R @ points + t
    #     # 注意: points 是 [B, J, 3], rotation 是 [B, 3, 3]
    #     # 使用 einsum 确保矩阵乘法作用在坐标维度上
    #     points = torch.einsum('bij,bkj->bki', rotation, points) 
    #     points = points + translation.unsqueeze(1) # [1, J, 3]

    #     # B. 归一化深度 (Perspective Distortion): [x/z, y/z, 1]
    #     # 这里的 points[:,:,-1] 就是深度 Z
    #     projected_points = points / points[:, :, -1].unsqueeze(-1)

    #     # C. 应用内参矩阵 K
    #     # 构造 K 矩阵
    #     K = torch.zeros([batch_size, 3, 3], device=device, dtype=dtype)
    #     K[:, 0, 0] = focal_length[:, 0]
    #     K[:, 1, 1] = focal_length[:, 1]
    #     K[:, 2, 2] = 1.
    #     K[:, :-1, -1] = camera_center

    #     # 应用 K: res = K @ projected_points
    #     projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    #     # 4. 返回结果，去掉 Batch 维度，保留 [J, 3] (x, y, 1)
    #     return projected_points.squeeze(0)

    def project_to_2d(self, joints_3d, cam_intrinsics, cam_pose):
        """
        joints_3d: [J, 3], world coordinates
        cam_intrinsics: [3, 3]
        cam_pose: [4, 4], world->camera
        """
        # 坐标变换到相机系
        J = joints_3d.shape[0]
        device = joints_3d.device

        # 1. 转为齐次坐标 [J,4]
        joints_h = torch.cat([joints_3d, torch.ones(J,1,device=device)], dim=-1)

        # 2. 外参变换
        joints_cam = (cam_pose @ joints_h.T).T  # [J,4]
        X_c, Y_c, Z_c = joints_cam[:,0], joints_cam[:,1], joints_cam[:,2]

        # 3. 投影
        u = (cam_intrinsics[0,0] * X_c / Z_c) + cam_intrinsics[0,2]
        v = (cam_intrinsics[1,1] * Y_c / Z_c) + cam_intrinsics[1,2]

        return torch.stack([u, v, torch.ones_like(u)], dim=-1)

    def _load_pt(self, pt_path):
        if pt_path not in self._pt_cache:
            self._pt_cache[pt_path] = torch.load(pt_path, map_location="cpu")
        return self._pt_cache[pt_path]
    
    def __getitem__(self, idx):
        s = self.samples[idx]

        data = self._load_pt(s["pt_path"])
        k = s["frame_idx"]
        actor_idx = s["actor_idx"]
        # ===== 检查相机有效性 =====
        if not data["campose_valid"][k]:
            return None     

        # SMPL 参数
        pose = data["pose"][k]          # [72]
        # trans = data["trans"][k]        # [3]
        betas = data["shape"]         # [10]
        # gender = data["genders"]
        cam_intrinsics = data["cam_intrinsics"][k]
        cam_pose = data["cam_poses"][k]

        # # SMPL forward
        # pose_rotmat = batch_rodrigues(torch.from_numpy(pose).float().view(-1,3)).unsqueeze(0)  # [1,24,3,3]
        # betas_tensor = torch.from_numpy(betas).float().unsqueeze(0)                             # [1,10]
        # trans_tensor = torch.from_numpy(trans).float().unsqueeze(0)                             # [1,3]
        # smpl_out = self.smpl(betas=betas_tensor, body_pose=pose_rotmat[:,1:], global_orient=pose_rotmat[:,:1], transl=trans_tensor)
        # joints_3d = smpl_out.joints[0]  # [24,3]
        joints_3d = data["jointPositions"][k].view(-1,3) 

        # 图像
        img_id = data["img_ids"][k]
        img_name = f"image_{img_id:05d}.jpg"
        img_path = os.path.join(self.image_root, s["sequence_name"], img_name)
        cvimg = cv2.imread(img_path)
        # cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        H, W, _ = cvimg.shape

        smpl_params = {'global_orient': pose[:3],
                       'body_pose': pose[3:],
                       'betas': betas
                      }

        has_smpl_params = {'global_orient': True,
                           'body_pose': True,
                           'betas': True
                           }

        # 2D keypoints
        keypoints_2d = self.project_to_2d(joints_3d, cam_intrinsics, cam_pose)

        # 使用 get_example 裁剪图像 patch
        xmin, ymin = keypoints_2d[:,0].min(), keypoints_2d[:,1].min()
        xmax, ymax = keypoints_2d[:,0].max(), keypoints_2d[:,1].max()

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        bbox_size = max(xmax - xmin, ymax - ymin) * 1.2
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(
            np.array([bbox_size, bbox_size]),
            target_aspect_ratio=BBOX_SHAPE
        ).max()

        augm_config = self.cfg.DATASETS.CONFIG
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            cvimg, center_x, center_y,
            bbox_size, bbox_size,
            keypoints_2d.cpu().numpy(),
            joints_3d.cpu().numpy(),
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            False,
            augm_config
        )
        gt_pose = np.concatenate([smpl_params["global_orient"].reshape(-1), smpl_params["body_pose"].reshape(-1)])

        # IMU
        acc = torch.from_numpy(data["vacc"][actor_idx][k]).float()
        ori = torch.from_numpy(data["vrot"][actor_idx][k]).float()

        return {
            "image": img_patch,
            "imu_acc": acc,
            "imu_ori": ori,
            "gt_pose": torch.from_numpy(gt_pose).float(),
            "keypoints_2d": torch.from_numpy(keypoints_2d).float(),
            "keypoints_3d": torch.from_numpy(keypoints_3d).float(),
            "smpl_params": smpl_params,
            "has_smpl_params": has_smpl_params,
        }

    
       
        # save_path = "/home/zhanghongwen/wxPro2/4D-Humans1"
        # # 调用可视化函数
        # visualize_full_frame(cvimg, keypoints_2d, gt_keypoints_3d, save_path)
        
class ImageDataset(Dataset):
    def __init__(self, image_root,  pt_root, cfg, train = True):
        self.image_root = os.path.abspath(image_root)
        self.pt_root = pt_root
        self.cfg = cfg
        self.train = train

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std  = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        # 读取所有 pt 文件
        pt_files = sorted(glob(os.path.join(pt_root, "*.pt")))
        self.samples = []
        self._pt_cache = {}

        for pt_file in pt_files:
            data = torch.load(pt_file, map_location="cpu")
            N = data["poses"].shape[0]
            for k in range(N):
                self.samples.append({
                    "pt_path": pt_file,
                    "frame_idx": k,
                    "sequence_name": data.get("sequence_name")
                })
                    

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found")
        print(f"[Dataset] total samples: {len(self.samples)}")
       

    def __len__(self):
        return len(self.samples)


    def project_to_2d(self, joints_3d, cam_intrinsics, cam_pose):
        """
        joints_3d: [J, 3], world coordinates
        cam_intrinsics: [3, 3]
        cam_pose: [4, 4], world->camera
        """
        # 坐标变换到相机系
        J = joints_3d.shape[0]
        device = joints_3d.device

        # 1. 转为齐次坐标 [J,4]
        joints_h = torch.cat([joints_3d, torch.ones(J,1,device=device)], dim=-1)

        # 2. 外参变换
        joints_cam = (cam_pose @ joints_h.T).T  # [J,4]
        X_c, Y_c, Z_c = joints_cam[:,0], joints_cam[:,1], joints_cam[:,2]

        # 3. 投影
        u = (cam_intrinsics[0,0] * X_c / Z_c) + cam_intrinsics[0,2]
        v = (cam_intrinsics[1,1] * Y_c / Z_c) + cam_intrinsics[1,2]

        return torch.stack([u, v, torch.ones_like(u)], dim=-1)

    def _load_pt(self, pt_path):
        if pt_path not in self._pt_cache:
            self._pt_cache[pt_path] = torch.load(pt_path, map_location="cpu")
        return self._pt_cache[pt_path]
    
    def __getitem__(self, idx):
        s = self.samples[idx]

        data = self._load_pt(s["pt_path"])
        k = s["frame_idx"]
        # ===== 检查相机有效性 =====
        if not data["campose_valid"][k]:
            return None     

        # SMPL 参数
        pose = data["poses"][k]          # [72]
        # trans = data["trans"][k]        # [3]
        betas = data["shape"]         # [10]
        # gender = data["genders"]
        cam_intrinsics = data["cam_intrinsics"]
        cam_pose = data["cam_poses"][k]
        # print("cam_intrinsics:", cam_intrinsics.shape)
        # print("cam_pose:", cam_pose.shape)

        joints_3d = data["jointPositions"][k].view(-1,3) 
        # print("joints_3d.shape:", joints_3d.shape)

        # 图像
        # img_id = data["img_ids"][k]
        img_name = f"image_{k:05d}.jpg"
        img_path = os.path.join(self.image_root, s["sequence_name"], img_name)
        cvimg = cv2.imread(img_path)
        # cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)    # get_example()会变回去
        H, W, _ = cvimg.shape

        pose_np = pose.cpu().numpy()
        betas_np = betas.cpu().numpy()
        smpl_params = {'global_orient': pose_np[:3],
                       'body_pose': pose_np[3:],
                       'betas': betas_np
                      }

        has_smpl_params = {'global_orient': True,
                           'body_pose': True,
                           'betas': True
                           }

        # 2D keypoints
        keypoints_2d = self.project_to_2d(joints_3d, cam_intrinsics, cam_pose)

        # 使用 get_example 裁剪图像 patch
        xmin, ymin = keypoints_2d[:,0].min(), keypoints_2d[:,1].min()
        xmax, ymax = keypoints_2d[:,0].max(), keypoints_2d[:,1].max()

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        bbox_size = max(xmax - xmin, ymax - ymin) * 1.5
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(
            np.array([bbox_size, bbox_size]),
            target_aspect_ratio=BBOX_SHAPE
        ).max()

        augm_config = self.cfg.DATASETS.CONFIG
        keypoints_3d = np.concatenate([joints_3d.cpu().numpy(), np.ones((joints_3d.shape[0], 1))],axis=-1)  # [J,4]
        keypoints_2d = keypoints_2d.cpu().numpy()
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            cvimg, center_x, center_y,
            bbox_size, bbox_size,
            keypoints_2d, keypoints_3d,
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            self.train,
            augm_config
        )
        gt_pose = np.concatenate([smpl_params["global_orient"].reshape(-1), smpl_params["body_pose"].reshape(-1)])
        # print({k: (type(v), v.shape) for k,v in smpl_params.items()})
  
        return {
            "image": img_patch,
            "gt_pose": torch.from_numpy(gt_pose).float(),
            "keypoints_2d": torch.from_numpy(keypoints_2d).float(),
            "keypoints_3d": torch.from_numpy(keypoints_3d).float(),
        }


class ImageIMUDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_set = ImageIMUDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.TRAIN_PT_ROOT,
            self.cfg,
            train=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
        )

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_set = ImageDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.TRAIN_PT_ROOT,
            self.cfg,
            train=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
        )

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_sample_to_file(img_patch, keypoints_2d, keypoints_3d, save_path, show_3d=True, figsize=(10, 5)):
    
    # 1. 还原维度 (C,H,W) -> (H,W,C)
    img = np.array(img_patch)
    img = img.transpose(1, 2, 0)

    # 2. 反归一化 (使用 RGB 顺序的均值)
    # 这里的 mean/std 量级取决于你的 self.mean/std 是 0.4 还是 123
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    
    if img.min() < 0: # 确认是归一化后的数据
        img = (img * std) + mean
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    fig = plt.figure(figsize=figsize)

    # ------------------- 2. 2D 散点图 (无连线) -------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title("2D Keypoints (Scatter)")
    ax1.axis("off")

    # 处理 keypoints_2d 类型
    kp2d = keypoints_2d.detach().cpu().numpy() if torch.is_tensor(keypoints_2d) else np.array(keypoints_2d)

    for i in range(len(kp2d)):
        # get_example 后的坐标在 [-0.5, 0.5] 之间，0 为中心
        x_norm, y_norm, v = kp2d[i]
        if v > 0:
            # 还原到像素坐标
            xi = (x_norm + 0.5) * w
            yi = (y_norm + 0.5) * h
            ax1.scatter(xi, yi, c='red', s=15, edgecolors='white', linewidths=0.5)

    # ------------------- 3. 3D 散点图 (无连线) -------------------
    if show_3d:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("3D Keypoints (Scatter)")
        
        kp3d = keypoints_3d.detach().cpu().numpy() if torch.is_tensor(keypoints_3d) else np.array(keypoints_3d)
        
        # 只取前三列 X, Y, Z
        X, Y, Z = kp3d[:, 0], kp3d[:, 1], kp3d[:, 2]
        ax2.scatter(X, Y, Z, c='blue', s=15)

        # 保持 3D 比例
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (X.max()+X.min())*0.5, (Y.max()+Y.min())*0.5, (Z.max()+Z.min())*0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 调整视角让它看起来像站立的
        ax2.view_init(elev=20, azim=-70)

    # ------------------- 4. 保存 -------------------
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
def train(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = HMR2WithIMU(cfg)
    # model.to(device)
    # datamodule = ImageIMUDataModule(cfg)
    # dataset = ImageIMUDataset("/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/imageFiles", "/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/train", cfg, train=True)
    dataset = ImageDataset("/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/imageFiles", "/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/train", cfg, train=True)
    for idx in range(500,505):
        sample = dataset[idx]
    #     if sample is None:
    #         continue   # 跳过
    #     img_patch = sample["image"]           # tensor [3,H,W]
    #     keypoints_2d = sample["keypoints_2d"]
    #     keypoints_3d = sample["keypoints_3d"]
    #     save_path = os.path.join("./", f"sample2_{idx:03d}.png")
    #     # 调用可视化函数
    #     visualize_sample_to_file(img_patch, keypoints_2d, keypoints_3d, save_path)

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