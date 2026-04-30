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