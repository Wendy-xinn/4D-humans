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

img_root = "/home/zhanghongwen/wxPro2/sam-3d-body/data/TotalCapture/extracted_images"
save_root = "/home/zhanghongwen/wxPro2/sam-3d-body/data/TotalCapture/bbox_cache"
os.makedirs(save_root, exist_ok=True)
for subject in sorted(os.listdir(img_root)):
    subj_dir = os.path.join(img_root, subject)
    if not os.path.isdir(subj_dir):
        continue

    # 遍历 subject 下所有动作摄像头文件夹
    for action_cam in sorted(os.listdir(subj_dir)):
        img_dir = os.path.join(subj_dir, action_cam)
        if not os.path.isdir(img_dir):
            continue
        
        bbox_dict = {}

        # 遍历动作文件夹下的图像
        for img_name in tqdm(sorted(os.listdir(img_dir)), desc=f"{subject}/{action_cam}"):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                bbox_dict[img_path] = None
                continue

            # 1. 检测
            outputs = detector(img)
            det_instances = outputs["instances"]

            # 2. 只保留 person 且 score>0.5
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)

            if valid_idx.sum() == 0:
                bbox_dict[img_path] = None
                continue

            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            scores = det_instances.scores[valid_idx].cpu().numpy()

            # 3. 取置信度最高的
            best_idx = scores.argmax()
            best_box = boxes[best_idx]

            bbox_dict[img_path] = best_box.tolist()
            # print(bbox_dict)
            # img = cv2.imread("/home/zhanghongwen/wxPro2/sam-3d-body/data/TotalCapture/extracted_images/s1/acting1_cam1/frame_000000.jpg")
            # bbox = [954.7147, 256.3385, 1378.4514, 687.4528]
            # x1, y1, x2, y2 = map(int, bbox)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imwrite("bbox_result.jpg", img)

        action_name = f"{subject}_{action_cam}.pt"
        save_path = os.path.join(save_root, action_name)
        torch.save(bbox_dict, save_path)
        print(f"[Saved] {save_path}")

# ✅ bbox_dict 就是最终结果
# key: 图像名，value: [x1, y1, x2, y2] 或 None（没有检测到）
# torch.save(bbox_dict, "bbox_dict.pt")
