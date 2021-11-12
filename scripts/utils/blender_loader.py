"""NeRF data loader utils
Developer: Dong Quoc Tranh
Created at: 29/10/2021
"""
import os
import json
import cv2
import torch
import imageio
import numpy as np


def translate_t(t):
    """Get translation matrix"""
    return torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]).float()


def rotate_phi(phi):
    """Get x-axis rotation matrix"""
    return torch.tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]
    ]).float()


def rotate_theta(theta):
    """Get y-axis rotation matrix | correspond to """
    return torch.tensor([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ]).float()


def spheric_pose(theta, phi, t):
    """Get camera-to-world matrix"""
    c2w = translate_t(t)
    c2w = rotate_phi(phi / 180. * np.pi) @ c2w
    c2w = rotate_theta((theta / 180. * np.pi)) @ c2w
    return torch.Tensor([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]) @ c2w


def load_blender_data(base_dir, half_res=False, step=1):
    """Load blender-type dataset.

    Args:
        base_dir (str): Directory which contains dataset
        half_res (bool): Use half of image resolution
        step (int): Step between two samples

    Returns:
        imgs (ndarray | NxHxWx4)
        poses (ndarray | Nx4x4): Camera extrinsic parameters (transform matrix)
        [H, W, focal]: Camera intrinsic parameters

    """
    splits = ['train', 'val', 'text']
    all_imgs, all_poses, counts = [], [], [0]
    for s in splits:
        with open(os.path.join(base_dir, f'transforms_{s}.json'), 'r') as f:
            meta = json.load(f)

        imgs, poses = [], []
        if s == 'train' or step == 0:
            step = 1

        for frame in meta['frames'][::step]:
            fname = os.path.join(base_dir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = (W / 2 * np.tan(camera_angle_x / 2))
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        imgs = np.array([cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) for img in imgs])

    render_poses = torch.stack(
        [spheric_pose(angle, -30., 4.) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0
    )
    return imgs, poses, [H, W, focal], render_poses, i_split
