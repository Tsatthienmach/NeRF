import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.utils import load_blender_data

imgs, poses, [H, W, focal] = load_blender_data('.data/nerf_synthetic/lego',
                                               annot_name='transforms_train.json',
                                               half_res=False, step=1)

assert imgs.shape[0] == poses.shape[0], \
    f'Number samples of image and pose should be equal, but got {imgs.shape[0]} | {poses.shape[0]}'

###################################################
# Test half_res
###################################################

t_imgs, t_poses, [t_H, t_W, t_focal] = load_blender_data('.data/nerf_synthetic/lego',
                                                         annot_name='transforms_train.json',
                                                         half_res=True, step=1)

assert t_imgs.shape[0] == t_poses.shape[0], \
    f'Number samples of image and pose should be equal, but got {t_imgs.shape[0]} | {t_poses.shape[0]}'
assert t_H == H // 2, f"H should be {H // 2}, but got {t_H}"
assert t_W == W // 2, f"W should be {W // 2}, but got {t_W}"
assert t_focal == focal / 2, f"Focal length should be {focal / 2}, but got {t_focal}"

###################################################
# Test step
###################################################

t_imgs, t_poses, [t_H, t_W, t_focal] = load_blender_data('.data/nerf_synthetic/lego',
                                                         annot_name='transforms_train.json',
                                                         half_res=False, step=5)

assert t_imgs.shape[0] == t_poses.shape[0], \
    f'Number samples of image and pose should be equal, but got {t_imgs.shape[0]} | {t_poses.shape[0]}'
assert t_imgs.shape[0] == np.ceil(imgs.shape[0] / 5)
