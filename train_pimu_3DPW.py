import os
import time
import torch
import json
from pathlib import Path
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
from scipy.spatial.transform import Rotation as R
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
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

class ImageIMUDataset(Dataset):
    def __init__(self, image_root,  pt_root, cfg, train = True, imu_window_size=5):
        self.image_root = os.path.abspath(image_root)
        self.pt_root = pt_root
        self.cfg = cfg
        self.train = train
        self.imu_window_size = imu_window_size  # 奇数，如5→取中心帧±2帧

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
            # ✅ 关键：加载对齐索引
            img_ids = data['img_ids'].long() # [N_img] 30Hz→60Hz映射
            campose_valid = data["campose_valid"]
            
            N_img = len(img_ids)
            for k in range(N_img):
                if not campose_valid[k]:
                    continue
                
                # ✅ 记录：图像帧k → 60Hz中心帧idx
                center_60hz = img_ids[k].item()   # 如: k=0 → center=0; k=1 → center=2  tensor -> int
                self.samples.append({
                    "pt_path": pt_file,
                    "img_idx_30hz": k,           # 图像帧索引(30Hz)
                    "center_60hz": center_60hz,  # ✅ 对应60Hz中心帧
                    "sequence_name": data.get("sequence_name"),
                    "total_60hz_frames": data["pose_60Hz"].shape[0]  # 用于边界检查
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
        
        # 🔑 关键索引
        center_60hz = s["center_60hz"]  # 该图像对应的60Hz中心帧
        total_60hz = s["total_60hz_frames"]
        w = self.imu_window_size
        half_w = w // 2

        # ✅ 1. 提取 60Hz IMU 窗口 [center-half_w : center+half_w+1]
        start = max(0, center_60hz - half_w)
        end = min(total_60hz, center_60hz + half_w + 1)
        
        imu_rot_window = data['vrot'][start:end]   # [W_imu, 6, 3, 3]
        imu_acc_window = data['vacc'][start:end]   # [W_imu, 6, 3]
        # ✅ 边界填充（如果窗口越界）
        if imu_rot_window.shape[0] < w:
            pad_len = w - imu_rot_window.shape[0]
            # 简单复制边界帧填充（也可用零填充/镜像）
            imu_rot_window = torch.cat([
                imu_rot_window[[0]].expand(pad_len//2, -1, -1, -1) if pad_len//2 > 0 else torch.empty(0),
                imu_rot_window,
                imu_rot_window[[-1]].expand(pad_len - pad_len//2, -1, -1, -1) if pad_len - pad_len//2 > 0 else torch.empty(0)
            ], dim=0)
            imu_acc_window = torch.cat([
                imu_acc_window[[0]].expand(pad_len//2, -1, -1),
                imu_acc_window,
                imu_acc_window[[-1]].expand(pad_len - pad_len//2, -1, -1)
            ], dim=0)

        # SMPL 参数
        # ✅ 2. GT: 取中心帧的 60Hz pose/trans/joints (用于监督)
        pose_60hz = data["pose_60Hz"][center_60hz]      # [72]
        trans_60hz = data["trans_60Hz"][center_60hz]    # [3]
        joints_3d = data["jointPositions"][s["img_idx_30hz"]].view(-1, 3)  # [J, 3]
        betas = data["shape"]                            # [10]
        # ✅ 3. 相机参数 (30Hz图像帧对应)
        cam_intrinsics = data["cam_intrinsics"]
        cam_pose = data["cam_poses"][s["img_idx_30hz"]]  # 注意：cam_poses是30Hz!
        # print("cam_intrinsics:", cam_intrinsics.shape)
        # print("cam_pose:", cam_pose.shape)


        # 2. 【关键修复】转换 global_orient 到 +Y向下系
        global_orient_4dhumans = convert_global_orient_yup_to_ydown_np(pose_60hz[:3])
        # pose_fixed = pose.clone()
        # pose_fixed[:3] = torch.from_numpy(global_orient_4dhumans)
        # pose_rotmat = batch_rodrigues(pose_fixed.view(-1, 3)).view(1, 24, 3, 3)
        # smpl_output = self.smpl(
        #     betas=betas.unsqueeze(0),
        #     body_pose=pose_rotmat[:, 1:],
        #     global_orient=pose_rotmat[:, :1], 
        #     transl=trans.unsqueeze(0)
        # )
        # joints_3d = smpl_output.joints.squeeze(0)
        # joints_3d_rh = joints_3d.clone()
        # joints_3d_rh[:, 0] *= -1
        # conf = torch.ones((joints_3d_rh.shape[0], 1), dtype=joints_3d.dtype)
        # keypoints_3d = torch.cat([joints_3d_rh, conf], dim=1).cpu().numpy()  # [J, 4]
        # ================= 坐标系转换 =================
        # 1. 世界系 → 相机系
        # ⚠️ 注意：cam_pose是30Hz的
        joints_h = torch.cat([joints_3d, torch.ones_like(joints_3d[:, :1])], dim=-1)
        joints_cam_3dpw = (cam_pose @ joints_h.T).T[:, :3]

        # print("joints_3d.shape:", joints_3d.shape)
        

        # 图像
        # img_id = data["img_ids"][k]
        img_name = f"image_{s['img_idx_30hz']:05d}.jpg"
        img_path = os.path.join(self.image_root, s["sequence_name"], img_name)
        cvimg = cv2.imread(img_path)
        # cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)    # get_example()会变回去
        H, W, _ = cvimg.shape

        pose_np = pose_60hz.cpu().numpy()
        betas_np = betas.cpu().numpy()
        smpl_params = {'global_orient':  global_orient_4dhumans,            #pose_np[:3],
                       'body_pose': pose_np[3:],
                       'betas': betas_np
                      }

        # has_smpl_params = {'global_orient': True,
        #                    'body_pose': True,
        #                    'betas': True
        #                    }
        has_smpl_params = {
            'global_orient': np.array([1.0], dtype=np.float32),
            'body_pose': np.array([1.0], dtype=np.float32),
            'betas': np.array([1.0], dtype=np.float32),
        }
    
        # 2D keypoints
        # keypoints_2d = self.project_to_2d(joints_3d, cam_intrinsics, cam_pose).cpu().numpy()

        # 2D 投影 (全透视)
        X, Y, Z = joints_cam_3dpw.unbind(-1)
        Z = torch.clamp(Z, min=1e-5)
        u = cam_intrinsics[0,0] * X / Z + cam_intrinsics[0,2]
        v = cam_intrinsics[1,1] * Y / Z + cam_intrinsics[1,2]
        keypoints_2d = torch.stack([u, v], dim=-1)

        # # 补齐维度 [J, 4]
        J = joints_cam_3dpw.shape[0]
        keypoints_3d_input = torch.cat([joints_cam_3dpw, torch.ones(J, 1)], dim=-1).cpu().numpy()
        keypoints_2d_input = keypoints_2d.cpu().numpy()
        if keypoints_2d_input.shape[1] == 2:
            keypoints_2d_input = np.concatenate([keypoints_2d_input, np.ones((J, 1))], axis=1)

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
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            cvimg, center_x, center_y,
            bbox_size, bbox_size,
            keypoints_2d_input, keypoints_3d_input,
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            # self.train,
            False,
            augm_config
        )
        # gt_pose = np.concatenate([smpl_params["global_orient"].reshape(-1), smpl_params["body_pose"].reshape(-1)])
        # print({k: (type(v), v.shape) for k,v in smpl_params.items()})

        item = {
            "img": img_patch,
            "smpl_params": {
                "global_orient": torch.from_numpy(smpl_params["global_orient"]).float(),  # (3,)
                "body_pose": torch.from_numpy(smpl_params["body_pose"]).float(),          # (69,)
                "betas": torch.from_numpy(smpl_params["betas"]).float()                   # (10,)
        },
            "keypoints_2d": torch.from_numpy(keypoints_2d).float(),
            "keypoints_3d": torch.from_numpy(keypoints_3d).float(),
            "has_smpl_params": { k: torch.tensor([1.0 if v else 0.0])
                                for k, v in has_smpl_params.items()},
            # 🆕 新增 IMU 窗口输入
            "imu_rot": imu_rot_window.float(),   # [W, 6, 3, 3]
            "imu_acc": imu_acc_window.float(),   # [W, 6, 3]
            "imu_window_size": w,
            "center_60hz_idx": torch.tensor(center_60hz),  # 可选：用于调试/损失mask
        }
        # print({k: type(v) for k, v in item.items()})
        return item
        
# ================= 放在文件顶部，class ImageIMUDataset 外部 =================

def convert_global_orient_yup_to_ydown_np(rotvec_np):
    """
    将 rotation vector 从 +Y up 坐标系转换到 +Y down 坐标系（纯 NumPy 实现）
    Args:
        rotvec_np: numpy array, shape (3,)
    Returns:
        numpy array, shape (3,)
    """
    rotvec_np = np.asarray(rotvec_np, dtype=np.float32)
    theta = np.linalg.norm(rotvec_np)
    if theta < 1e-8:
        return rotvec_np.copy()

    r = rotvec_np / theta
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1.0 - c

    # 构建旋转矩阵 R_old
    r_cross = np.array([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=np.float32)
    R_old = c * np.eye(3, dtype=np.float32) + s * r_cross + t * np.outer(r, r)

    # 坐标系变换: R_new = M @ R_old @ M, M = diag(1, -1, 1)
    M = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    R_new = M @ R_old @ M

    # 旋转矩阵 -> Rodrigues 向量
    trace = np.trace(R_new)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    theta_new = np.arccos(cos_theta)

    if theta_new < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = np.array([R_new[2,1] - R_new[1,2],
                     R_new[0,2] - R_new[2,0],
                     R_new[1,0] - R_new[0,1]], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    return (theta_new * axis).astype(np.float32)

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
                if not data["campose_valid"][k]:
                    continue   # 直接跳过
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
        trans = data["trans"][k]
        # 2. 【关键修复】转换 global_orient 到 +Y向下系
        global_orient_4dhumans = convert_global_orient_yup_to_ydown_np(pose[:3])
        # pose_fixed = pose.clone()
        # pose_fixed[:3] = torch.from_numpy(global_orient_4dhumans)
        # pose_rotmat = batch_rodrigues(pose_fixed.view(-1, 3)).view(1, 24, 3, 3)
        # smpl_output = self.smpl(
        #     betas=betas.unsqueeze(0),
        #     body_pose=pose_rotmat[:, 1:],
        #     global_orient=pose_rotmat[:, :1], 
        #     transl=trans.unsqueeze(0)
        # )
        # joints_3d = smpl_output.joints.squeeze(0)
        # joints_3d_rh = joints_3d.clone()
        # joints_3d_rh[:, 0] *= -1
        # conf = torch.ones((joints_3d_rh.shape[0], 1), dtype=joints_3d.dtype)
        # keypoints_3d = torch.cat([joints_3d_rh, conf], dim=1).cpu().numpy()  # [J, 4]
        # ================= 坐标系转换 =================
        # 1. 世界系 → 相机系
        joints_h = torch.cat([joints_3d, torch.ones_like(joints_3d[:, :1])], dim=-1)
        joints_cam_3dpw = (cam_pose @ joints_h.T).T[:, :3]

        
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
        smpl_params = {'global_orient':  global_orient_4dhumans,            #pose_np[:3],
                       'body_pose': pose_np[3:],
                       'betas': betas_np
                      }

        # has_smpl_params = {'global_orient': True,
        #                    'body_pose': True,
        #                    'betas': True
        #                    }
        has_smpl_params = {
            'global_orient': np.array([1.0], dtype=np.float32),
            'body_pose': np.array([1.0], dtype=np.float32),
            'betas': np.array([1.0], dtype=np.float32),
        }

        # 2D keypoints
        # keypoints_2d = self.project_to_2d(joints_3d, cam_intrinsics, cam_pose).cpu().numpy()

        # 2D 投影 (全透视)
        X, Y, Z = joints_cam_3dpw.unbind(-1)
        Z = torch.clamp(Z, min=1e-5)
        u = cam_intrinsics[0,0] * X / Z + cam_intrinsics[0,2]
        v = cam_intrinsics[1,1] * Y / Z + cam_intrinsics[1,2]
        keypoints_2d = torch.stack([u, v], dim=-1)

        # # 补齐维度 [J, 4]
        J = joints_cam_3dpw.shape[0]
        keypoints_3d_input = torch.cat([joints_cam_3dpw, torch.ones(J, 1)], dim=-1).cpu().numpy()
        keypoints_2d_input = keypoints_2d.cpu().numpy()
        if keypoints_2d_input.shape[1] == 2:
            keypoints_2d_input = np.concatenate([keypoints_2d_input, np.ones((J, 1))], axis=1)

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
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            cvimg, center_x, center_y,
            bbox_size, bbox_size,
            keypoints_2d_input, keypoints_3d_input,
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            # self.train,
            False,
            augm_config
        )
        # gt_pose = np.concatenate([smpl_params["global_orient"].reshape(-1), smpl_params["body_pose"].reshape(-1)])
        # print({k: (type(v), v.shape) for k,v in smpl_params.items()})

        item = {
            "img": img_patch,
            "smpl_params": {
                "global_orient": torch.from_numpy(smpl_params["global_orient"]).float(),  # (3,)
                "body_pose": torch.from_numpy(smpl_params["body_pose"]).float(),          # (69,)
                "betas": torch.from_numpy(smpl_params["betas"]).float()                   # (10,)
        },
            "keypoints_2d": torch.from_numpy(keypoints_2d).float(),
            "keypoints_3d": torch.from_numpy(keypoints_3d).float(),
            "has_smpl_params": { k: torch.tensor([1.0 if v else 0.0])
                                for k, v in has_smpl_params.items()}
        }
        # print({k: type(v) for k, v in item.items()})
        return item

class EgoDataset(Dataset):
    def __init__(self, root_dir, split_file, cfg, mode='train'):
        """
        root_dir: 数据集根目录
        split_file: data_splits.csv 路径
        mode: 'train' or 'val'
        """
        self.root_dir = Path(root_dir)
        self.split_file = split_file
        self.cfg = cfg

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)
        self.img_size = cfg.MODEL.IMAGE_SIZE   # 目标尺寸224
        self.img_size_ori = cfg.MODEL.IMAGE_SIZE_ORI      # 原始长边1920
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std  = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)
        
        # 1. 从 CSV 获取序列名称
        df_split = pd.read_csv(self.split_file)
        if mode in df_split.columns:
            self.seq_names = df_split[mode].dropna().tolist()
        else:
            raise ValueError(f"Mode '{mode}' 不在 CSV 的列名中。可选列: {df_split.columns.tolist()}")
        
        # 定义子文件夹路径
        self.color_root = self.root_dir / "egocentric_color"
        self.calib_root = self.root_dir / "calibrations"
        self.smpl_root = self.root_dir / f"smpl_camera_wearer_{mode}"
        # self.seq_names = ["recording_20210907_S02_S01_01", "recording_20210921_S11_S10_01"]
        
        self.data_list = []
        self._prepare_data()

    def _prepare_data(self):
        """ 遍历所有序列，建立索引表 """
        for seq in self.seq_names:
            seq_path = self.color_root / seq
            if not seq_path.exists(): continue
            
            # 处理“不用管名称”的子文件夹 (通常是时间戳文件夹)
            sub_folders = [f for f in seq_path.iterdir() if f.is_dir()]
            if not sub_folders: continue
            seq_content_path = sub_folders[0] 
            
            # 读取 pv.txt 获取位姿和内参
            pv_txt_path = list(seq_content_path.glob("*_pv.txt"))[0]
            pv_info = self._parse_pv_txt(pv_txt_path)
            
            # 读取该序列的 calibration
            calib_path = self.calib_root / seq / "cal_trans" / "holo_to_kinect12.json"
            with open(calib_path, 'r') as f:
                holo_to_kinect = json.load(f) # 包含trans(四元数)
                T_h2m = np.array(holo_to_kinect['trans'])
                T_m2h = np.linalg.inv(T_h2m)
            
            # 遍历图像文件夹 PV
            img_dir = seq_content_path / "PV"
            for img_path in img_dir.glob("*.jpg"):
                img_name = img_path.stem 
                
                # 分割文件名获取 timestamp 和 frame_id
                parts = img_name.split('_', 1)
                if len(parts) < 2: continue
                timestamp, frame_id = parts[0], parts[1]
                # print(timestamp)
                # print(frame_id)
                
                # 匹配 SMPL 文件路径
                # 路径示例: RECORDING_NAME/body_idx_x/results/frame_xxxxx/000.pkl
                # 注意: 这里假设 body_idx_0 是 wearer，具体需根据数据确认
                smpl_search_pattern = f"{seq}/body_idx_*/results/{frame_id}/000.pkl"
                matching_pkls = list(self.smpl_root.glob(smpl_search_pattern))
                
                if matching_pkls and timestamp in pv_info:
                    self.data_list.append({
                        'img_path': str(img_path),
                        'pv_data': pv_info[timestamp],
                        'cx_cy': pv_info['meta'], # 这是个字典
                        'T_m2h': T_m2h,
                        'smpl_path': str(matching_pkls[0])
                    })

    def _parse_pv_txt(self, path):
        """ 解析 pv.txt 文件 """
        info = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            # 第一行: cx, cy, w, h
            meta = [float(x) for x in lines[0].strip().split(',')]
            info['meta'] = {'cx': meta[0], 'cy': meta[1]}
            
            # 后续行: timestamp, fx, fy, pv2world_transform
            for line in lines[1:]:
                data = line.strip().split(',')
                ts = data[0]
                fx, fy = float(data[1]), float(data[2])
                # 4x4 变换矩阵
                trans_mat = np.array([float(x) for x in data[3:]]).reshape(4, 4)
                T_w2c = np.linalg.inv(trans_mat)
                info[ts] = {'fx': fx, 'fy': fy, 'T_w2c': T_w2c}    # 这里的transform是pv camera坐标系到holo世界坐标系
        return info

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print(self.__len__())
        item = self.data_list[idx]
        
        # 1. 读取图像(可以后面直接用图像的路径进行图像处理)
        # image = cv2.imread(item['img_path'])
        # 2. 读取 SMPL 参数，得到3d坐标
        with open(item['smpl_path'], 'rb') as f:
            smpl_data = pickle.load(f)
        # 需要从axis-angle转换成矩阵的形式
        # OpenCV (Y down, Z forward) -> OpenGL (Y up, Z back)  这里不对，训练应该用opencv的坐标系
        # flip_yz = torch.tensor([
        #     [1,  0,  0, 0],
        #     [0, -1,  0, 0],
        #     [0,  0, -1, 0],
        #     [0,  0,  0, 1]
        # ], dtype=torch.float32)
        T_w2c = torch.tensor(item['pv_data']['T_w2c'], dtype=torch.float32)
        T_m2h = torch.tensor(item['T_m2h'], dtype=torch.float32)
        # T_total = flip_yz @ T_w2c @ T_m2h  # (4, 4)
        T_total = T_w2c @ T_m2h  # (4, 4)

        global_orient = torch.tensor(smpl_data['global_orient'], dtype=torch.float32).view(-1, 3)
        global_orient = batch_rodrigues(global_orient)   # (B, 3, 3)

        R_m2c = T_total[:3, :3].clone().detach().float()
        global_orient_c = R_m2c @ global_orient
        r_obj = R.from_matrix(global_orient_c.numpy().squeeze())   # 转换成axis angle
        global_orient_camera = r_obj.as_rotvec().astype(np.float32)

        global_orient_ = global_orient.view(-1, 1, 3, 3)  # (B, 1, 3, 3)

        body_pose = torch.tensor(smpl_data['body_pose'], dtype=torch.float32).view(-1, 3)
        body_pose = batch_rodrigues(body_pose)           # (B*23, 3, 3)
        body_pose = body_pose.view(-1, 23, 3, 3)         # (B, 23, 3, 3)

        betas = torch.tensor(smpl_data['betas'], dtype=torch.float32).view(1, -1)
        transl = torch.tensor(smpl_data['transl'], dtype=torch.float32).view(1, 3)
        smpl_output = self.smpl(
            global_orient=global_orient_,
            body_pose=body_pose,
            betas=betas,
            transl = transl,
            pose2rot=False 
        )  # master坐标系下
        # joints_3d_master = smpl_output.joints.detach().cpu().numpy().squeeze(0)
        joints_3d_master = smpl_output.joints.squeeze(0)
        vertices_master = smpl_output.vertices
        # 扩展为齐次坐标 [B, 6890, 4]
        ones = torch.ones(vertices_master.shape[0], vertices_master.shape[1], 1).to(vertices_master.device)
        vertices_master_ = torch.cat([vertices_master, ones], dim=-1)
        vertices_cam = (T_total @ vertices_master_.transpose(1, 2)).transpose(1, 2).squeeze(0)[:, :3] # 截取回 [6890, 3]

        # print(joints_3d_master)
        # print(joints_np)
        # 手动构造齐次坐标: (N, 3) -> (N, 4)
        joints_homo = np.ones((joints_3d_master.shape[0], 4))
        joints_homo[:, :3] = joints_3d_master
        joints_3d_camera = (T_total @ joints_homo.T).T[:, :3] # (J, 3)
        joints_3d_np = joints_3d_camera.detach().cpu().numpy()
        # print(joints_3d_camera)

        # 3. 透视投影: Camera 3D -> Image 2D Pixel(取消2d的loss，因为z太小会导致u/v很大，没办法计算)
        fx, fy = item['pv_data']['fx'], item['pv_data']['fy']
        cx, cy = item['cx_cy']['cx'], item['cx_cy']['cy'] # 注意索引方式
        z = joints_3d_np[:, 2]
        u = (joints_3d_np[:, 0] * fx) / z + cx
        v = (joints_3d_np[:, 1] * fy) / z + cy
        # # print('u:', u)
        # # print('v:', v)
        
        # 组装为 (N, 3) 格式，最后一列是 visibility 标志位
        # 判定有效性
        # z > 0 是基础，0.01 是为了避开分母过小导致的数值不稳定
        valid_mask = (z > 0.1) 

        # 对于无效的点，把它们坐标设为 0 或者一个安全值，并将 visibility 设为 0
        u[~valid_mask] = 0
        v[~valid_mask] = 0

        # 组装时，最后一列是 visibility (1.0 代表有效，0.0 代表无效)
        keypoints_2d_input = np.stack([u, v, valid_mask.astype(np.float32)], axis=-1)
        J = joints_3d_np.shape[0]
        keypoints_3d_input = np.concatenate([joints_3d_np, np.ones((J, 1))], axis=-1)
        
        smpl_params = {'global_orient':  global_orient_camera,            #pose_np[:3],
                       'body_pose': smpl_data['body_pose'],
                       'betas': smpl_data['betas']
                      }
        has_smpl_params = {
            'global_orient': np.array([1.0], dtype=np.float32),
            'body_pose': np.array([1.0], dtype=np.float32),
            'betas': np.array([1.0], dtype=np.float32),
        }

        augm_config = self.cfg.DATASETS.CONFIG
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            item['img_path'], item['cx_cy']['cx'], item['cx_cy']['cy'],
            self.img_size_ori, self.img_size_ori,
            keypoints_2d_input, keypoints_3d_input,
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            # self.train,
            False,
            augm_config
        )
        
            
        # 3. 准备返回数据
        sample = {
            'img': img_patch,
            "smpl_params": {
                "global_orient": torch.tensor(smpl_params['global_orient'], dtype=torch.float32),  # (3,)
                "body_pose": torch.tensor(smpl_params['body_pose'], dtype=torch.float32),          # (69,)
                "betas": torch.tensor(smpl_params['betas'], dtype=torch.float32)                   # (10,)
        },
            "keypoints_2d": torch.tensor(keypoints_2d, dtype=torch.float32),
            "keypoints_3d": torch.tensor(keypoints_3d, dtype=torch.float32),
            "has_smpl_params": { k: torch.tensor([1.0 if v else 0.0])
                                for k, v in has_smpl_params.items()},
            # "cx": torch.tensor(item['cx_cy']['cx'], dtype=torch.float32),
            # "cy": torch.tensor(item['cx_cy']['cy'], dtype=torch.float32)
            'vertices': vertices_cam.detach().cpu().numpy().astype(np.float32),
        }

        return sample
        

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, cfg, dataset_cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = ImageIMUDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.TRAIN_PT_ROOT,
            self.cfg,
            train=True
        )
        self.val_dataset = ImageIMUDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.TRAIN_VAL_ROOT,
            self.cfg,
            train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            # num_workers=0,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
        )
    

class HMR2DataModule_ego(pl.LightningDataModule):
    def __init__(self, cfg, dataset_cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = EgoDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.SPLIT_FILE,
            self.cfg,
            mode='train'
        )
        self.val_dataset = EgoDataset(
            self.cfg.DATA.TRAIN_IMG_ROOT,
            self.cfg.DATA.SPLIT_FILE,
            self.cfg,
            mode='val'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            # num_workers=0,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
        )


import matplotlib.pyplot as plt

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

# def debug_canvas(kps_2d, cx, cy):
#     canvas = np.zeros((5000, 5000, 3), dtype=np.uint8)
#     offset = 2500
#     for kp in kps_2d:
#         # 将坐标平移到画布中心
#         u, v = int(kp[0] - cx + offset), int(kp[1] - cy + offset)
#         print(u,v)
#         if 0 <= u < 5000 and 0 <= v < 5000:
#             cv2.circle(canvas, (u, v), 10, (0, 255, 255), -1)
#     return canvas

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_joints(idx, joints_3d):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    joints_pure = np.array(joints_3d.tolist(), dtype=np.float32)
    
    x = joints_pure[:, 0]
    y = joints_pure[:, 1]
    z = joints_pure[:, 2]
    
    # 绘制关节
    ax.scatter(x, y, z, c='r', marker='o')
    
    # 为了方便观察，标出原点（相机位置）
    ax.scatter([0], [0], [0], c='blue', marker='X', s=100, label='Camera')
    
    # 标注坐标轴
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down/Up)')
    ax.set_zlabel('Z (Forward)')
    ax.legend()
    
    # 强制等比例，否则人形会变形
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig(f"debug_canvas_{idx}.jpg")
    plt.close(fig)

def train(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model = HMR2WithIMU(cfg)
    # model.to(device)
    # datamodule = ImageIMUDataModule(cfg)
    # dataset = ImageIMUDataset("/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/imageFiles", "/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/train", cfg, train=True)
    # dataset = ImageDataset("/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/imageFiles", "/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/train", cfg, train=True)
    dataset = EgoDataset(cfg.DATA.TRAIN_IMG_ROOT, cfg.DATA.SPLIT_FILE, cfg, mode='train')
    for idx in range(0,5):
        sample = dataset[idx]
        keypoints_3d = sample["keypoints_3d"].detach().cpu().numpy()
        visualize_3d_joints(idx, keypoints_3d)
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