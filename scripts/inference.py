import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
import torch
import argparse
import numpy as np
from threading import Thread
from scripts.models import NeRF, Embedder
from scripts.utils.video_utils import VideoWriter
from scripts.utils.llff_utils import create_spiral_poses, spheric_pose
from scripts.datasets import LLFFDataset
from scripts.utils.ray_utils import get_rays
from scripts.utils.rendering import render_rays


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_freqs', type=int, default=10)
    parser.add_argument('--dir_freqs', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--no_log_scale', default=False, action='store_true')
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--hid_layers', type=int, default=256)
    parser.add_argument('--skips', nargs="+", type=int, default=[4])
    parser.add_argument('--chunk', default=3000, type=int)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    return parser.parse_args()


class NeRFInfer(torch.nn.Module):
    """NeRF inference module

    Args:
        ckpt (str): checkpoint path
        dvc: torch device
    """
    def __init__(self, ckpt, dvc, chunk=1000, pos_freqs=10, dir_freqs=4,
                 in_channels=3, log_scale=True, depth=8, hid_layers=256,
                 skips=[4]):
        super().__init__()
        self.device = dvc
        self.chunk = chunk
        self.next_idx = 0
        self.current_rgb = None
        self.pos_embedder = Embedder(pos_freqs, in_channels, log_scale)
        self.dir_embedder = Embedder(dir_freqs, in_channels, log_scale)
        self.coarse_model = NeRF(depth, hid_layers,
                                 in_channels * (1 + 2 * pos_freqs),
                                 in_channels * (1 + 2 * dir_freqs),
                                 skips=skips)
        self.fine_model = NeRF(depth, hid_layers,
                               in_channels * (1 + 2 * pos_freqs),
                               in_channels * (1 + 2 * dir_freqs),
                               skips=skips)
        self.coarse_model.to(self.device)
        self.fine_model.to(self.device)
        self.coarse_model.eval()
        self.fine_model.eval()
        self.ckpt = torch.load(ckpt)
        self.load_weight()

    def inference(self, radius=None, theta=0, phi=0):
        rays = self.create_rays(radius, theta, phi)
        self.render_rgbs(rays)

    def show_cv2(self, im):
        im = im.reshape(self.H, self.W, 3)
        cv2.imshow('rgb', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def render_rgbs(self, rays):
        ray_len = len(rays)
        tmp_img = np.ones((self.H * self.W, 3)).astype(np.uint8) * 255
        for i in range(0, ray_len, self.chunk):
            mini_rays = rays[i:i + self.chunk]
            mini_rays_len = len(mini_rays)
            sub_rays = mini_rays[::2]
            results = self(sub_rays.to(self.device))
            with torch.no_grad():
                rgb = results['rgb_fine'].view(-1, 3).cpu()
            tmp_img[i: i + mini_rays_len][::2] = np.array(rgb * 255,
                                                          dtype=np.uint8)
            self.show_cv2(tmp_img)

    def forward(self, rays):
        results = render_rays(
            models=[self.coarse_model, self.fine_model],
            embedders=[self.pos_embedder, self.dir_embedder],
            rays=rays, N_samples=64, perturb=1, N_importance=64,
            chunk=self.chunk, white_bg=True, test_mode=True
        )
        return results

    def load_weight(self):
        self.coarse_model.load_state_dict(self.ckpt['models'][0])
        self.fine_model.load_state_dict(self.ckpt['models'][1])
        self.test_info = self.ckpt['test_info']
        self.W = self.test_info['WH'][0]
        self.H = self.test_info['WH'][1]
        print(20 * '-')
        print('Epoch: ', self.ckpt['epoch'])
        print('PSNR: ', self.ckpt['best_psnr'])
        print(f'Radius boundary: {self.test_info["min_bound"] } ' +
              f'| {self.test_info["max_bound"]}')
        print('Resolution: ', self.W, self.H)
        print(20 * '-')

    def create_rays(self, radius=None, theta=0, phi=30):
        """Generate rays for a pose
        
        Args:
            radius (float)
            theta (degree)
            phi (degree)
        
        Returns:
            rays
            WH: resolution of output image
        """
        if not radius:
            radius = self.test_info['radius']

        pose = torch.FloatTensor(spheric_pose(theta / 180 * np.pi,
                                              phi / 180 * np.pi,
                                              radius))
        rays_o, rays_d = get_rays(self.test_info['directions'], pose)
        near = self.test_info['min_bound']
        far = min(8 * near, self.test_info['max_bound'])
        rays = LLFFDataset.to_rays(rays_o, rays_d, near, far)
        return rays


if __name__ == '__main__':
    # PARAMS
    params = get_opts()
    params.gpu = 0
    params.weight = 'logs/face_removed_bg/checkpoint_best_psnr.pth'
    device = torch.device(f'cuda:{params.gpu}')if params.gpu > -1 else \
        torch.device('cpu')
    nerf = NeRFInfer(ckpt=params.weight, dvc=device, chunk=params.chunk)
    for p in np.linspace(-180, 0, 18):
        # for theta in np.linspace(-360, 0, 5):
        theta = 180
        print(f'--------> theta: None | phi: {p} | theta: {theta}')
        nerf.inference(None, theta, p)
