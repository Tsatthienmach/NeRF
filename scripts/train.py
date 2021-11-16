import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scripts.datasets import dataset_dict
from scripts.losses import loss_dict
from scripts.loggers import Writer, ModelCheckPoint
from scripts.metrics import SSIM, PSNR
from scripts.models import NeRF, Embedder
from scripts.trainer import Trainer


if __name__ == '__main__':
    loss = loss_dict['mse']

    # DATASET
    dataset_module = dataset_dict['llff']
    train_set = dataset_module(
        root_dir='.data/nerf_llff_data/fern', split='train', img_wh=(504, 378),
        spheric_poses=False, val_num=1, transforms=T.Compose([T.ToTensor()]),
        res_factor=8)
    val_set = dataset_module(
        root_dir='.data/nerf_llff_data/fern', split='val', img_wh=(504, 378),
        spheric_poses=False, val_num=1, transforms=T.Compose([T.ToTensor()]),
        res_factor=8
    )
    test_set = dataset_module(
        root_dir='.data/nerf_llff_data/fern', split='test', img_wh=(504, 378),
        spheric_poses=False, val_num=1, transforms=T.Compose([T.ToTensor()]),
        res_factor=8)

    train_loader = DataLoader(
        train_set, shuffle=True, num_workers=4, batch_size=1024 * 20,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, shuffle=False, num_workers=4, batch_size=1, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, shuffle=False, num_workers=4, batch_size=1,
        pin_memory=True
    )

    for i in ['train', 'val', 'test']:
        c_loader = vars()[f'{i}_set']
        for k, v in list(c_loader)[0].items():
            print(k, v.shape)
        print()
