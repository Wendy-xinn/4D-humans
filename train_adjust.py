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
