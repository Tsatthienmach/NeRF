import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.datasets.llff_loader import LLFFDataset

dataset = LLFFDataset(
    '/home/tranhdq/ws/NeRF/.data/nerf_llff_data/fern',
    # split='train',
    split='test',
    spheric_poses=False,
)

print('Len:', dataset.__len__())
for k, v in dataset[0].items():
    print(k, v.shape, v[0:2])
