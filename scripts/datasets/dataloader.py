"""NeRF data loader
Developer: Dong Quoc Tranh
Created at: 29/10/2021
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor()
from ..utils import load_blender_data, load_llff_data, get_ray_directions


class NeRFDataset(Dataset):
    """LLFF dataset.

    Args:
        base_dir (str): data directory
        TODO
    """
    def __init__(
            self, base_dir, res_factor=4, recenter=True, bd_factor=0.75, spherify=False,
            transforms=None, data_type='llff', llff_hold=8, test_skip=8, K=None,
            half_res=True, white_bkgd=True, N_rand=1024, precrop_iters=None
    ):
        super(LLFFDataset, self).__init__()
        self.split=None
        self.precrop_iters = precrop_iters
        self.N_rand = N_rand
        self.transforms = transforms if transforms is not None else ToTensor()
        if data_type == 'llff':
            imgs, poses, bds, render_poses, i_test = load_llff_data(
                base_dir, factor=res_factor, recenter=recenter, bd_factor=bd_factor,
                spherify=spherify
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if llff_hold > 0:
                i_test = np.arange(imgs.shape[0])[::llff_hold]
            i_val = i_test
            i_train = np.arange([i for i in np.arange(int(imgs.shape[0])) \
                                 if i not in i_test and i not in i_val])
            if no_ndc:
                near = np.ndarray.min(bds) * 0.9
                far = np.ndarray.max(bds) * 1.
            else:
                near = 0.
                far = 1.

            print(f'Dataset: Train: {i_train} | Val: {i_val} | Test: {i_test}')
            print(f'Near boundary: {near} | Far boundary: {far}')

        elif data_type == 'synthetic':
            imgs, poses, hwf, render_poses, i_split = load_blender_data(
                base_dir, half_res=half_res, step=test_skip
            )
            i_train, i_val, i_test = i_split
            near = 2.
            far = 6.
            imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:]) if white_bkgd else imgs[..., :3]
        else:
            raise ValueError('Inappropriate datatype')

        hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
        if K is None:
            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])
        self.imgs = imgs
        self.poses = poses
        self.render_poses = render_poses
        self.hwf = hwf
        self.i_train = i_train
        self.i_val = i_val
        self.i_test = i_test

    def __len__(self):
        if self.split is None:
            raise ValueError('No define on split')

    def __getitem__(self, index):
        if self.split == 'train':
            target = self.imgs[index]
            pose = self.poses[index]
            if self.N_rand is not None:
                rays_o, rays_d = get_rays(*self.hwf, torch.Tensor(pose))  # HxWx3 - HxWx3
                coords = torch.stack(
                    torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1
                )

            coords = torch.reshape(coords, [-1, 2])
            select_ids = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)
            select_coords = coords[select_ids].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[: , 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[: , 1]]
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]


        elif self.split == 'val':
            pass

        elif self.split == 'test':
            pass

        else:
            raise ValueError('No define on split')



        return sample

    def copy(self, split='val'):
        pass

    def train(self):
        self.split = 'train'
        self.imgs = self.imgs[self.i_train]
        self.poses = self.poses[self.i_train]
        self.render_poses = None

    def eval(self):
        self.split = 'val'
        self.imgs = self.imgs[self.i_val]
        self.poses = self.poses[self.i_val]
        self.render_poses = None

    def test(self):
        self.split = 'test'
        # Render pose