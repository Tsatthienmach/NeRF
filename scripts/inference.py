import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

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
        self.current_index = 0
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
        t1 = Thread(target=self.render_rgbs, args=(rays,))
        t2 = Thread(target=self.show)
        t1.start()
        t2.start()

    def show(self):
        i = 0
        image = np.zeros((H * W, 3)).astype(np.uint8)
        while i < int(self.H * self.W) - 1:
            if i <= self.current_index and \
               self.current_rgb is not None:    
                image[i: self.current_index] = \
                    np.array(self.current_rgb).astype(np.uint8)
                i = self.index
                self.current_rgb = None

            img = image.reshape(self.H, self.W, 3)
            cv2.imshow('rgb', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def render_rgbs(self, rays):
        for i in range(0, len(rays), self.chunk):
            self.current_index = i + self.chunk
            mini_rays = rays[i:i + self.chunk]
            with torch.no_grad():
                results = self(mini_rays.to(self.device))

            self.current_rgb = \
                results['rgb_fine'].view(self.chunk, 3).cpu()

    def forward(self, rays):
        results = render_rays(
            models=[self.coarse_model, self.fine_model],
            rays=rays, N_samples=64, perturb=1, N_importance=64,
            chunk=self.chunk, white_bg=True, test_mode=True
        )

    def load_weight(self):
        self.coarse_model.load_state_dict(self.ckpt['models'][0])
        self.fine_model.load_state_dict(self.ckpt['models'][1])
        self.test_info = self.ckpt['test_info']
        self.WH = self.test_info['WH']
        print(20 * '-')
        print('Epoch: ', self.ckpt['epoch'])
        print('PSNR: ', self.ckpt['best_psnr'])
        print(f'Radius boundary: {self.test_info["min_bound"] } ' +
              f'- {self.test_info["max_bound"]}')
        print('Resolution: ', self.WH)
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
    # params.gpu=0
    # params.weight='/mnt/datadrive/tranhdq/NeRF/NeRF/logs/face_removed_bg/checkpoint_best_psnr.pth'


    device = torch.device(f'cuda:{params.gpu}')if params.gpu > -1 else \
        torch.device('cpu')

    nerf = NeRFInfer(ckpt=params.weight, dvc=device)
    nerf.inference()
