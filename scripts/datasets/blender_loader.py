import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from ..utils.ray_utils import get_rays, get_ray_dirs


class BlenderDataset(Dataset):
    """Blender dataset module

    Args:
        root_dir (str): dataset directory
        split (str | train/val): training set or validation test
        img_wh (tuple): Scaling width, height
        transforms (object): transformer
        white_bg (bool): transparent background to white background
        val_step (int): validation item getting step
    """
    def __init__(self, root_dir, split='train', img_wh=(500, 500),
                 transforms=None, white_bg=True, val_step=1):
        self.root_dir = root_dir
        self.split = split
        self.transforms = T.ToTensor() if transforms is None else transforms
        self.img_wh = img_wh
        self.white_bg = white_bg
        self.val_step = val_step
        self.read_meta()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        else:
            return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {
                'rays': self.all_rays[idx],
                'rgbs': self.all_rgbs[idx]
            }
            return sample

        else:
            # Create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            img = Image.open(os.path.join(self.root_dir,
                                          f'{frame["file_path"]}.png'))
            W, H = self.img_wh
            if img.size != (W, H):
                img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transforms(img)
            valid_mask = (img[-1] > 0).flatten()  # (HxW) valid color area
            img = img.view(4, -1).permute(1, 0)  # (HxW, 4)
            img = self.blend_rgb(img)
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = self.to_rays(rays_o, rays_d)
            return {
                'rays': rays,
                'rgbs': img,
                'c2w': c2w,
                'valid_mask': valid_mask
            }

    def read_meta(self):
        with open(os.path.join(self.root_dir, f'transforms_{self.split}.json'),
                  'r') as f:
            self.meta = json.load(f)
            self.meta['frames'] = self.meta['frames'][::self.val_step]
        W, H = self.img_wh
        self.focal = 0.5 * W / np.tan(0.5 * self.meta['camera_angle_x'])
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        self.directions = get_ray_dirs(H, W, self.focal)
        if self.split == 'train':
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)
                image_path = os.path.join(self.root_dir,
                                          f'{frame["file_path"]}.png')
                self.poses.append(pose)
                self.image_paths.append(image_path)
                img = Image.open(image_path)
                if img.size != (W, H):
                    img = img.resize(self.img_wh, Image.LANCZOS)

                img = self.transforms(img)  # (4, H, W)
                img = img.view(4, -1).permute(1, 0)  # (HxW, 4) RGBA
                img = self.blend_rgb(img)
                self.all_rgbs.append(img)
                rays_o, rays_d = get_rays(self.directions, c2w)
                self.all_rays.append(self.to_rays(rays_o, rays_d))  # (HxW, 8)

            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)

    def blend_rgb(self, rgba):
        """Blend a RGB image from RGBA image
        Args:
            rgba (HxW, 4): RGBA image

        Returns:
            rgb (HxW, 3)
        """
        if self.white_bg:
            img = rgba[:, :3] * rgba[:, -1:] + (1 - rgba[:, -1:])
        else:
            img = rgba[:, :3]

        return img

    def to_rays(self, rays_o, rays_d):
        """Form rays"""
        return torch.cat([
            rays_o,
            rays_d,
            self.near * torch.ones_like(rays_o[:, :1]),
            self.far * torch.ones_like(rays_o[:, :1])
        ], 1)
