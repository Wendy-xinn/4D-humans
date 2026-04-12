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
    def __init__(self, image_root,  pt_root, bbox_root, cfg, train = True):
        self.image_root = os.path.abspath(image_root)
        self.pt_root = pt_root
        self.bbox_root = bbox_root
        self.cfg = cfg
        self.train = train

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)

        self.img_size = cfg.MODEL.IMAGE_SIZE
        # self.img_size = (192, 256)  # (W, H)
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std  = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        # Camera intrinsics (TotalCapture)
        self.fx, self.fy = 1284.32, 1286.38
        self.cx, self.cy = 959.5, 539.5
        # Camera extrinsics (R, t)
        self.R = torch.tensor([
            [-0.99713, 0.00504186, -0.0755413],
            [0.0221672, -0.93461, -0.354982],
            [-0.0723915, -0.355637, 0.931816]
        ], dtype=torch.float32)
        self.t = torch.tensor([0.820506, 0.59704, 5.33591], dtype=torch.float32)

        extensions = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}

        self.samples = []
        self._pt_cache = {}
        self._bbox_cache = {}

        # 遍历 image_root/s*/action*_cam*/
        # subjects = sorted(os.listdir(self.image_root))
        subjects = ["s1" ]
        for subject in subjects:
            subj_dir = os.path.join(self.image_root, subject)
            # print(subj_dir)
            if not os.path.isdir(subj_dir):
                continue

            # for action_cam in sorted(os.listdir(subj_dir)):
            for action_cam in ["acting1_cam1"]:
                img_dir = os.path.join(subj_dir, action_cam)
                print(img_dir)
                if not os.path.isdir(img_dir):
                    continue
                # action 名（去掉 cam）
                action = "_".join(action_cam.split("_")[:-1])
                pt_file = os.path.join(self.pt_root, f"{subject}_{action}.pt")
                bbox_file = os.path.join(bbox_root, f"{subject}_{action_cam}.pt")
                if not os.path.exists(pt_file) or not os.path.exists(bbox_file):
                    print(f"[WARN] missing pt or bbox: {subject}/{action_cam}")
                    continue

                # 读取 pt（一次）
                data = torch.load(pt_file, map_location="cpu")
                # bbox_dict = torch.load(bbox_file)
                N_pt = data["acc"].shape[0]
                key = (subject, action)

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
                        "img_path": images[i],
                        # "subject": subject,
                        # "action": action,
                        "pt_path": pt_file,
                        "bbox_path": bbox_file,
                        "frame_idx": i
                    })

        print(f"[Dataset] total samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError("No valid image–IMU pairs found")

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

    def project_smpl_to_2d(self, joints_3d):
        """
        注意：此时输入的 joints_3d 必须已经是相机系下的坐标（Depth 5m左右）
        """
        device = joints_3d.device
        
        # 1. 提取 XYZ (J, 3)
        # 不要再做 R @ p + t 了！
        X = joints_3d[:, 0]
        Y = joints_3d[:, 1]
        Z = joints_3d[:, 2] # 这里的 Z 应该就是你 print 出来的 5.12 左右

        # 2. 透视投影 (Perspective Projection)
        # 标准公式：u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy
        
        # 3. 返回 [J, 3] 格式，保持与你原代码一致
        return torch.stack([u, v, torch.ones_like(u)], dim=-1)

    def _load_pt(self, pt_path):
        if pt_path not in self._pt_cache:
            self._pt_cache[pt_path] = torch.load(pt_path, map_location="cpu")
        return self._pt_cache[pt_path]
    def _load_bbox(self, bbox_path):
        if bbox_path not in self._bbox_cache:
            self._bbox_cache[bbox_path] = torch.load(bbox_path, map_location="cpu")
        return self._bbox_cache[bbox_path]
    def __getitem__(self, idx):
        s = self.samples[idx]

        data = self._load_pt(s["pt_path"])
        bbox_dict = self._load_bbox(s["bbox_path"])
        bbox = bbox_dict.get(s["img_path"], None)
        k = s["frame_idx"]
        
         # ---------- load image ----------
        cvimg = cv2.imread(s["img_path"])
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

        H, W, _ = cvimg.shape

        # ========= bbox =========
        if bbox is None:
            # fallback to full image
            center_x = W/2; center_y = H/2
            width = max(H,W)
            height = width
        else:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2)/2
            center_y = (y1 + y2)/2
            width = x2 - x1
            height = y2 - y1
            # bbox_size = max(width, height)
            bbox_size = expand_to_aspect_ratio(np.array([width, height]), target_aspect_ratio=(1.0,1.0)).max()

        # --------- SMPL 3D joints ---------
        pose = data["pose"][k]        # torch
        pose_np = pose.cpu().numpy()
        tran = data["tran"][k]        # torch  现在是相对位移
        # print(tran)
        # 尝试映射：Vicon_X -> SMPL_X, Vicon_Y -> SMPL_Y (高度), Vicon_Z -> SMPL_Z (深度)
        # 如果投影出来人是倒着的或平躺的，再微调这个顺序
        target_tran = torch.stack([tran[0], -tran[1], tran[2]])
        target_trans = self.t.to(tran.device)  + target_tran  # 回到相机视野
        smpl_params = {"global_orient": pose_np[:3], "body_pose": pose_np[3:]}
        has_smpl_params = {"global_orient": True, "body_pose": True}

        gt_shape = torch.zeros(1, 10, device=pose.device)
        pose_rotmat = batch_rodrigues(pose.view(-1, 3)).unsqueeze(0) 
        target_trans = target_trans.unsqueeze(0)
        # 构造一个绕 Y 轴旋转 180 度的旋转矩阵
        R_y_180 = torch.tensor([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=torch.float32)

        # 假设 pose_rotmat[0, 0] 是原来的 3x3 global_orient
        new_global_orient = R_y_180 @ pose_rotmat[0, 0] @ R_y_180 @ R_y_180
        temp_output = self.smpl(
            betas=gt_shape,
            body_pose=pose_rotmat[:, 1:],
            global_orient=new_global_orient.unsqueeze(0),
            # global_orient=pose_rotmat[:, :1], 
            transl=torch.zeros_like(target_trans)          # ← 关键：先不加平移
        )
        joints_temp = temp_output.joints[0]               # [24, 3] （或你SMPL的joint数）
        gt_keypoints_3d = (joints_temp - joints_temp[0]) + target_trans
        print(f"Depth check: {gt_keypoints_3d[0, 2].item():.2f}m")
        # root_idx = 0                                      # SMPL root = pelvis
        # spine_idx = 3
        # offset = joints_temp[spine_idx] - joints_temp[root_idx]
        # target_tran_corrected = target_trans - offset.unsqueeze(0)
        # smpl_output = self.smpl(
        #     betas=gt_shape,         # [B, 10]
        #     body_pose=pose_rotmat[:, 1:],  # [B, 23, 3, 3]
        #     # global_orient=pose_rotmat[:, :1],  # [B, 1, 3, 3]
        #     global_orient=new_global_orient,
        #     transl=target_tran_corrected        # [B, 3]
        # )
        # gt_keypoints_3d = smpl_output.joints[0]  # [J, 3]   世界坐标系
        # gt_keypoints_3d = gt_keypoints_3d.clone()
        # gt_keypoints_3d[..., 0] *= -1  # 翻转 x 轴
        # gt_keypoints_3d[..., 2] *= -1   # 同时翻转 Z（前后方向）
        keypoints_2d = self.project_smpl_to_2d(gt_keypoints_3d)  # (N,3)
       
        # save_path = "/home/zhanghongwen/wxPro2/4D-Humans1"
        # # 调用可视化函数
        # visualize_full_frame(cvimg, keypoints_2d, gt_keypoints_3d, save_path)
        # 转 numpy
        keypoints_2d = keypoints_2d.cpu().numpy()
        gt_keypoints_3d = gt_keypoints_3d.cpu().numpy()
        keypoints_3d = np.concatenate([gt_keypoints_3d, np.ones((gt_keypoints_3d.shape[0], 1))],axis=-1)  # [J,4]

        augm_config = self.cfg.DATASETS.CONFIG
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            cvimg,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoints_2d, keypoints_3d,
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            False,
            augm_config,
        )
        gt_pose = np.concatenate([smpl_params["global_orient"].reshape(-1), smpl_params["body_pose"].reshape(-1)])

        # -------- imu --------
        acc = data["acc"][k]        # (6, 3)
        ori = data["ori"][k]        # (6, 3, 3)
        

        return {
            "image": img_patch,
            "imu_acc": acc.float(),
            "imu_ori": ori.float(),
            "gt_pose": torch.from_numpy(gt_pose).float(),
            # "gt_tran": tran.float(),
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
            self.cfg.DATA.TRAIN_BBOX_ROOT,
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

def visualize_full_frame(img_rgb, kp2d, kp3d, save_path):
    """
    专门用于在全图（未裁剪）上查看投影是否准确
    """
    import matplotlib.pyplot as plt
    
    # 确保图像是 numpy 且为 uint8
    if torch.is_tensor(img_rgb):
        img = img_rgb.detach().cpu().numpy().astype(np.uint8)
    else:
        img = np.array(img_rgb).astype(np.uint8)
        
    # 强制 RGB 格式显示
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    fig = plt.figure(figsize=(15, 7))
    
    # 2D 部分
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img) # 此时 img 是 0-255 的 uint8，imshow 不会黑屏
    
    # 转换坐标点
    pts = kp2d.detach().cpu().numpy() if torch.is_tensor(kp2d) else kp2d
    
    # 直接画像素点，不要加任何 (x+0.5)*w 的偏移！
    # 因为 project_smpl_to_2d 出来的已经是像素坐标了
    ax1.scatter(pts[:, 0], pts[:, 1], c='r', s=10, edgecolors='white', linewidths=0.3)
    ax1.set_title("Full Frame Projection")
    ax1.axis('off')

    # 3D 部分
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_box_aspect([1,1,1]) # 强制比例 1:1:1
    p3d = kp3d.detach().cpu().numpy() if torch.is_tensor(kp3d) else kp3d
    ax2.scatter(p3d[:, 0], p3d[:, 1], p3d[:, 2], c='b', s=10)
    ax2.view_init(elev=-90, azim=-90) # SMPL 常用视角
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
def visualize_sample_to_file(img_patch, keypoints_2d, keypoints_3d, save_path, show_3d=True, figsize=(10, 5)):
    
    # ------------------- 1. 准备图像 (修复维度报错) -------------------
    if torch.is_tensor(img_patch):
        img = img_patch.detach().cpu().numpy()
    else:
        img = np.array(img_patch)

    # 重点：如果维度是 (3, H, W)，强制转为 (H, W, 3)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    # 反标准化 (Un-normalize)
    # HMR2 默认使用 ImageNet 均值方差，这里将其还原回 [0, 255]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if img.max() <= 5: # 如果是标准化的浮点数
        img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

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
    dataset = ImageIMUDataset("/home/zhanghongwen/wxPro2/sam-3d-body/data/TotalCapture/extracted_images", "/home/zhanghongwen/wxPro2/sam-3d-body/data/TotalCapture/preprocessed_gt", "/home/zhanghongwen/wxPro2/sam-3d-body/data/TotalCapture/bbox_cache", cfg, train=False)
    for idx in range(500,505):
        sample = dataset[idx]
        img_patch = sample["image"]           # tensor [3,H,W]
        keypoints_2d = sample["keypoints_2d"]
        keypoints_3d = sample["keypoints_3d"]
        save_path = os.path.join("./", f"sample_{idx:03d}.png")
        # 调用可视化函数
        visualize_sample_to_file(img_patch, keypoints_2d, keypoints_3d, save_path)

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