import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import torch.nn.functional as F

from .render_openpose import render_openpose, render_gt_24_keypoints

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class MeshRenderer:

    def __init__(self, cfg, faces=None):
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.focal_length_ego = cfg.EXTRA.FOCAL_LENGTH_EGO
        self.img_res = cfg.MODEL.IMAGE_SIZE
        # self.img_res = cfg.MODEL.VIT_IMAGE_SIZE
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.img_res,
                                       viewport_height=self.img_res,
                                       point_size=1.0)
        
        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces

    def visualize(self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    # 原图/mesh正视图/mesh侧视图/预测关键点/GT关键点
    def visualize_tensorboard(self, vertices, camera_translation, images, pred_keypoints, gt_keypoints, focal_length=None, nrow=5, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
        pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        # if gt_keypoints.shape[-1] == 2:
        #     gt_keypoints = np.concatenate((gt_keypoints, np.ones_like(gt_keypoints)[:, :, [0]]), axis=-1)
        gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
        # gt_keypoints[..., 0] = (gt_keypoints[..., 0] + 0.5) * 256
        # gt_keypoints[..., 1] = (gt_keypoints[..., 1] + 0.5) * 192
        keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9), (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
            body_keypoints = pred_keypoints[i, :25]
            extra_keypoints = pred_keypoints[i, -19:]
            for pair in keypoint_matches:
                body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
            # print(body_keypoints)
            pred_keypoints_img = render_openpose(255 * images_np[i].copy(), body_keypoints) / 255
            body_keypoints = gt_keypoints[i, :25]
            extra_keypoints = gt_keypoints[i, -19:]
            for pair in keypoint_matches:
                if extra_keypoints[pair[1], -1] > 0 and body_keypoints[pair[0], -1] == 0:  # 置信度筛选
                    body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
            # print(body_keypoints)
            gt_keypoints_img = render_gt_24_keypoints(255*images_np[i].copy(), body_keypoints) / 255
            # gt_keypoints_img = render_openpose(255*images_np[i].copy(), body_keypoints) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs
    
    def visualize_tensorboard_ego(self, vertices, gt_vertices, camera_translation, images, focal_length_ego=None, nrow=4, padding=2):
        batch_size = vertices.shape[0]
        rend_imgs = []
        
        # 动态调整网格高度：根据 GT 的 Y 均值动态设置地平线，防止“悬浮”
        y_floor = gt_vertices[0].mean(0)[1] + 0.9 # 假设脚底在重心下方 0.9m
        grid_points = self.create_ground_grid(y_pos=y_floor, size=5, res=10)
        
        for i in range(batch_size):
            # --- 1. 原始 Pred (包含模型预测的平移) ---
            pred_verts_cam = vertices[i] + camera_translation[i]
            
            # --- 2. 重心对齐的 Pred (诊断 Pose 用) ---
            # 强制将 Pred 的重心移动到 GT 的重心，看姿态是否重合
            pred_center = vertices[i].mean(0)
            gt_center = gt_vertices[i].mean(0)
            pred_verts_aligned = vertices[i] - pred_center + gt_center
            
            # 调用渲染
            # A: 预测 (上帝视角)
            rend_pred_3d = self.render_ego_bird_view(pred_verts_cam, None, grid_points)
            # B: GT (上帝视角)
            rend_gt_3d = self.render_ego_bird_view(None, gt_vertices[i], grid_points)
            # C: 姿态对比 (强制对齐重心后的红+绿)
            # 只要这个图里的红绿重合了，就说明 Pose 学对了，只是 Translation 没对齐
            rend_pose_comp = self.render_ego_bird_view(pred_verts_aligned, gt_vertices[i], grid_points)

            def to_tensor(x):
                # 记得在这里做 np.flip(x, axis=0) 如果你的渲染器没翻转
                return torch.from_numpy(np.transpose(x, (2, 0, 1))).float()

            rend_imgs.extend([
                torch.from_numpy(images[i]),    # 第一人称原图
                to_tensor(rend_pred_3d),        # 原始预测位置
                to_tensor(rend_gt_3d),          # GT 
                to_tensor(rend_pose_comp)       # 强制重心对齐后的对比 (核心诊断图)
            ])

        grid = make_grid(rend_imgs, nrow=4, padding=padding)
        return grid

    def create_ground_grid(self, y_pos=1.7, size=4, res=8):
        """ 生成地平面网格顶点 """
        grid_points = []
        ticks = np.linspace(-size/2, size/2, res+1)
        for t in ticks:
            # 沿着 Z 轴的线
            grid_points.append(np.stack([np.full(res+1, t), np.full(res+1, y_pos), ticks], axis=1))
            # 沿着 X 轴的线
            grid_points.append(np.stack([ticks, np.full(res+1, y_pos), np.full(res+1, t)], axis=1))
        return np.concatenate(grid_points, axis=0)

    def render_ego_bird_view(self, pred_vertices=None, gt_vertices=None, grid_vertices=None):
        import pyrender
        import trimesh.transformations as tf

        # 创建渲染器 (RGBA, 256x256)
        renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0], ambient_light=(0.4, 0.4, 0.4))

        # 1. 材质定义保持
        mat_pred = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.8, 0.3, 0.3, 1.0))
        mat_gt = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.3, 0.8, 0.3, 1.0))

        # 【2. 彻底移除 flip_rot 逻辑】
        # 只要从 Dataset 出来的物体，一律不做任何旋转，原样丢进 scene
        if pred_vertices is not None:
            # 如果模型预测也飞了，检查 pred_cam_t 的单位和方向
            m_pred = trimesh.Trimesh(pred_vertices.copy(), self.faces)
            scene.add(pyrender.Mesh.from_trimesh(m_pred, material=mat_pred))

        if gt_vertices is not None:
            m_gt = trimesh.Trimesh(gt_vertices.copy(), self.faces)
            scene.add(pyrender.Mesh.from_trimesh(m_gt, material=mat_gt))

        if grid_vertices is not None:
            # 网格保持原样
            grid_homo = np.ones((grid_vertices.shape[0], 4))
            grid_homo[:, :3] = grid_vertices
            pc = pyrender.Mesh.from_points(grid_vertices, colors=[[0.5, 0.5, 0.5]]*len(grid_vertices))
            scene.add(pc)

        # --- 3. 修改相机设置 (OpenCV 空间 Look-at) ---
        
        # Target (瞄准点) 直接用原始均值即可 (满足 OpenCV Y 向下)
        if gt_vertices is not None:
            target = gt_vertices.mean(0)
        elif pred_vertices is not None:
            target = pred_vertices.mean(0)
        else:
            target = np.array([0.0, 0.0, 1.5]) # 默认看向相机前方的合理深度

        # 设置相机位置 (相对于 Target 的 OpenCV 偏移)
        # 我们希望相机站在右后方，高处鸟瞰
        # OpenCV: X右, Y下, Z前。
        # cam_offset[0]=2.0 (右); cam_offset[2]=-3.0 (后退)
        # cam_offset[1]=-2.0 (高处鸟瞰，OpenCV Y 向上是负数)
        cam_offset = np.array([2.0, -2.0, -3.0]) 
        cam_pos = target + cam_offset

        # 手动计算 Look-at 矩阵
        # 【4. 核心关键】：UP 向量在 OpenCV 世界系里。
        # 我们看着一个头朝下 ($Y$向下) 的物体。
        # 为了让渲染出来的画面中头朝上，我们相机的头顶必须对着世界系下的 $-Y$ 轴。
        def get_lookat_pose(eye, target, up=[0, -1, 0]): # <--- 修正 UP 向量
            z = eye - target
            z /= np.linalg.norm(z)
            x = np.cross(up, z) # 如果 up 是 [0,-1,0]，x 会指向 OpenCV 系的左方。
            x /= np.linalg.norm(x)
            y = np.cross(z, x) # 重新计算完美的 Camera Y-up
            pose = np.eye(4)
            pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
            return pose

        # 使用 UP=[0, -1, 0] 计算 Camera Pose
        camera_pose = get_lookat_pose(cam_pos, target, up=[0, -1, 0])

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0) # 约45度视角，人大小合适
        scene.add(camera, pose=camera_pose)

        # 4. 灯光与渲染
        from hmr2.utils.mesh_renderer import create_raymond_lights
        for node in create_raymond_lights():
            scene.add_node(node)

        # 只取 RGB (256, 256, 3) 供 make_grid 使用
        color, _ = renderer.render(scene)
        # 使用 np.flip 或 [::-1] 将图像上下颠倒，对齐 OpenCV/PIL 习惯
        color = np.flip(color, axis=0)
        color_rgb = color.astype(np.float32)[:,:,:3] / 255.0
        renderer.delete()
        return color_rgb

    # def visualize_tensorboard_ego(self, vertices, gt_vertices, camera_translation, images, focal_length_ego=None, nrow=5, padding=2):
    #     """
    #     针对 Ego 任务优化的可视化：对比预测与 GT
    #     layout: 原图 | Pred Overlay | GT Overlay | Pred Side | GT Side
    #     """
    #     images_np = np.transpose(images, (0, 2, 3, 1))
    #     rend_imgs = []
        
    #     # 准备一个全 0 的平移向量，用于渲染已经在相机系下的 GT 顶点
    #     null_cam_t = np.zeros(3)
        
    #     for i in range(vertices.shape[0]):
    #         fl = self.focal_length_ego
            
    #         # --- 1. 预测渲染 (Prediction) ---
    #         # 正面
    #         rend_pred = torch.from_numpy(np.transpose(
    #             self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), 
    #             (2, 0, 1))).float()
    #         # 侧面
    #         rend_pred_side = torch.from_numpy(np.transpose(
    #             self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), 
    #             (2, 0, 1))).float()
                
    #         # --- 2. GT 渲染 (Ground Truth) ---
    #         # 重要：gt_vertices[i] 已经是相机坐标系绝对坐标，所以 camera_translation 传全 0
    #         # 正面
    #         rend_gt = torch.from_numpy(np.transpose(
    #             self.__call__(gt_vertices[i], null_cam_t, images_np[i], focal_length=fl, side_view=False), 
    #             (2, 0, 1))).float()
    #         # 侧面
    #         rend_gt_side = torch.from_numpy(np.transpose(
    #             self.__call__(gt_vertices[i], null_cam_t, images_np[i], focal_length=fl, side_view=True), 
    #             (2, 0, 1))).float()
            
    #         # 按组添加：5张图构成一行
    #         rend_imgs.extend([
    #             torch.from_numpy(images[i]), # 原图
    #             rend_pred,                   # 预测正面
    #             rend_gt,                     # GT 正面
    #             rend_pred_side,              # 预测侧面
    #             rend_gt_side                 # GT 侧面
    #         ])
            
    #     # 调整每行显示 5 张图
    #     rend_imgs = make_grid(rend_imgs, nrow=5, padding=padding)
    #     return rend_imgs

    def __call__(self, vertices, camera_translation, image, focal_length=5000, text=None, resize=None, side_view=False, baseColorFactor=(1.0, 1.0, 0.9, 1.0), rot_angle=90):
        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=baseColorFactor)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]
        if resize is not None:
            output_img = cv2.resize(output_img, resize)

        output_img = output_img.astype(np.float32)
        renderer.delete()
        return output_img
