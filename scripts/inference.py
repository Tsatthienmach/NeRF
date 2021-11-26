import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import argparse
from scripts.models import NeRF, Embedder
from scripts.utils.video_utils import VideoWriter
from scripts.utils.llff_utils import create_spiral_poses, spheric_pose
from scripts.dataset import LLFFDataset


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
    def __init__(self, ckpt, dvc, pos_freqs=10, dir_freqs=4,
                 in_channels=3, log_scale=True, depth=8, hid_layers=256,
                 skips=[4]):
        super().__init__()
        self.device = dvc
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

    def forward(self, pose):
        pass

    def load_weight(self):
        self.coarse_model.load_state_dict(self.ckpt['models'][0])
        self.fine_model.load_state_dict(self.ckpt['models'][1])
        self.test_info = self.ckpt['test_info']
        print(20 * '-')
        print('Epoch: ', self.ckpt['epoch'])
        print('PSNR: ', self.ckpt['best_psnr'])
        print('Raidus bound: ', self.test_info['min_bound'] - self.test_info['max_bound'])
        print(20 * '-')

    def create_pose(self, radius=None, theta=0, phi=30):
        """Generate rays for a pose
        
        Args:
            radius (float)
            theta (degree)
            phi (degree)
        
        Returns:
            rays:
        """
        if not radius:
            radius = self.test_info['radius']

        pose = torch.FloatTensor(spheric_pose(theta, phi, radius))
        rays_o, rays_d = get_rays(self.test_info['directions'], pose)
        near = self.test_info['min_bound']
        far = min(8 * near, self.test_info['max_bound'])
        rays = LLFFDataset.to_rays(rays_o, rays_d, near, far)




if __name__ == '__main__':
    # PARAMS
    params = get_opts()
    # params.gpu=0
    # params.weight='/mnt/datadrive/tranhdq/NeRF/NeRF/logs/face_removed_bg/checkpoint_best_psnr.pth'


    device = torch.device(f'cuda:{params.gpu}')if params.gpu > -1 else \
        torch.device('cpu')

    nerf = NeRFInfer(ckpt=params.weight, dvc=device)
