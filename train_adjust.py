from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from yacs.config import CfgNode
from hmr2.configs import dataset_config, CACHE_DIR_4DHUMANS, get_config
from hmr2.datasets import HMR2DataModule
# from hmr2.models.hmr2 import HMR2
from hmr2.models.hmr2pimu import HMR2pimu
from hmr2.utils.pylogger import get_pylogger
from hmr2.utils.misc import task_wrapper, log_hyperparameters
from train_pimu_3DPW import ImageDataModule
from datetime import datetime

# HACK reset the signal handling so the lightning is free to set it
# Based on https://github.com/facebookincubator/submitit/issues/1709#issuecomment-1246758283
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

DEFAULT_CHECKPOINT=f'{CACHE_DIR_4DHUMANS}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
CHECKPOINT = "/media/zhanghongwen/Elements1/wxPro2/4D-Humans/logs/train/runs/hmr2_adjust/checkpoints/epoch=35-step=50000.ckpt"

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = ImageDataModule(cfg, dataset_cfg)

    # Setup model
    model = HMR2pimu(cfg)
    # print(model.smpl.joint_map)
    checkpoint_path = CHECKPOINT
    log.info(f"Loading pretrained checkpoint from {checkpoint_path}")
    # model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    # model_cfg = get_config(model_cfg, update_cachedir=True)
    # model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False )

    # ckpt = torch.load(checkpoint_path, map_location='cpu')
    # missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)

    # model = HMR2.load_from_checkpoint(checkpoint_path, cfg=cfg)  # 里面有discriminator会报错
   
    # 🔑 关键：区分两种场景
    if cfg.get('RESUME_FROM_CHECKPOINT', False):
        # 🔄 场景A: 恢复训练（从 last.ckpt 继续）
        # 让 Lightning 自动处理，但需确保 model.on_load_checkpoint 已重写
        ckpt_path = 'last'  # 或具体路径
        log.info(f"Resuming training from {ckpt_path}")
    else:
        # 🚀 场景B: 首次训练（从预训练 backbone 开始）
        # __init__ 中已手动加载预训练权重，这里禁用自动恢复
        ckpt_path = None
        log.info(f"Starting new training with pretrained backbone")

    # Setup Tensorboard logger
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name=f"run_{run_id}", version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    rich_callback = pl.callbacks.RichProgressBar()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
        # rich_callback
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers, 
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else None), # Submitit uses SIGUSR2
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path=str(root/"hmr2/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()


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