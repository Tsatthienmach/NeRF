import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.datasets.blender_loader import BlenderDataset

dataset = BlenderDataset(
    '/home/tranhdq/ws/NeRF/.data/nerf_synthetic/lego',
    # split='train',
    split='train',
    img_wh=(500, 500),
)

print('Len:', dataset.__len__())
for k, v in dataset[0].items():
    print(k, v.shape, v[0:2])
