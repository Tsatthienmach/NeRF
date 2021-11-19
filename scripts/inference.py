import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import argparse
from scripts.models import NeRF, Embedder
from scripts.utils.video_utils import VideoWriter
from scripts.utils.llff_utils import create_spheric_poses, create_spiral_poses


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
    return parser


class NeRFInfer(torch.nn.Module):
    """NeRF inference module

    Args:
        ckpt (str): checkpoint path
        dvc: torch device
    """
    def __init__(self, ckpt, dvc, pos_freqs=10, dir_freqs=4,
                 in_channels=3, log_scale=True, depth=8, hid_layers=256,
                 skips=[4]):
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
        self.coarse_model.to(device)
        self.fine_model.to(device)
        self.coarse_model.eval()
        self.fine_model.eval()
        self.ckpt = torch.load(ckpt)
        self.coarse_model.load_state_dict(self.ckpt['coarse'])
        self.fine_model.load_state_dict(self.ckpt['fine'])
        self.test_info = self.ckpt['test_info']

    def forward(self, pose):
        pass


if __name__ == '__main__':
    # PARAMS
    params = get_opts()
    device = torch.device(f'cuda:{params.gpu}')if params.gpu > -1 else \
        torch.device('cpu')
    weight = torch.load(params.weight)
