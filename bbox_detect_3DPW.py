from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hmr2
from pathlib import Path    
from tqdm import tqdm

import cv2
import os
import torch

# 1. 配置 Detectron2 预训练模型
cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)


# 2. 遍历图片，检测 bbox

img_root = "/home/zhanghongwen/wxPro2/sam-3d-body/data/3DPW/imageFiles"
save_root = "/home/zhanghongwen/wxPro2/sam-3d-body/data/data/3DPW/bbox_cache"
os.makedirs(save_root, exist_ok=True)


# ✅ bbox_dict 就是最终结果
# key: 图像名，value: [x1, y1, x2, y2] 或 None（没有检测到）
# torch.save(bbox_dict, "bbox_dict.pt")
