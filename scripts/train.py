import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scripts.datasets import dataset_dict
from scripts.losses import loss_dict
from scripts.loggers import Writer, ModelCheckPoint
from scripts.metrics import SSIM, PSNR
from scripts.models import NeRF, Embedder
from scripts.trainer import Trainer


if __name__ == '__main__':
    # PARAMS
    device = torch.device('cpu')
    exp_name = 'test'
    save_dir = 'logs'
    sfx = '4time_downscale'

    LOSS = 'mse'
    DATASET = 'llff'
    DATASET_DIR = '.data/nerf_llff_data/fern'

    # LOSS
    loss = loss_dict[LOSS]()

    # DATASET
    dataset_module = dataset_dict[DATASET]
    train_set = dataset_module(
        root_dir=DATASET_DIR, split='train', img_wh=(504, 378),
        spheric_poses=False, transforms=T.Compose([T.ToTensor()]),
        res_factor=8, val_step=10
    )
    val_set = dataset_module(
        root_dir=DATASET_DIR, split='val', img_wh=(504, 378),
        spheric_poses=False, transforms=T.Compose([T.ToTensor()]),
        res_factor=8, val_step=10
    )
    test_set = dataset_module(
        root_dir=DATASET_DIR, split='test', img_wh=(504, 378),
        spheric_poses=False, transforms=T.Compose([T.ToTensor()]),
        res_factor=8, val_step=10
    )

    train_loader = DataLoader(
        train_set, shuffle=False, num_workers=4, batch_size=1024,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, shuffle=False, num_workers=4, batch_size=1, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, shuffle=False, num_workers=4, batch_size=1, pin_memory=True
    )

    # MODELS
    embedders = {
        'pos': Embedder(N_freqs=10, in_channels=3, log_scale=True).to(device),
        'dir': Embedder(N_freqs=4, in_channels=3, log_scale=True).to(device)
    }
    models = {
        'coarse': NeRF(
            D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]
        ).to(device),
        'fine': NeRF(
            D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]
        ).to(device)
    }

    # OPTIMIZER AND LR SCHEDULER
    parameters = []
    for name, model in models.items():
        parameters += list(model.parameters())

    optimizer = Adam(parameters, lr=5e-4, eps=1e-8, weight_decay=0)
    lr_scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)

    # METRICS
    metrics = {
        'psnr': PSNR(),
        'ssim': SSIM()
    }

    # LOGGERS
    writer = Writer(
        exp_name=exp_name, save_dir=save_dir, sfx=sfx, i_image=1
    )
    model_ckpt = ModelCheckPoint(
        exp_name=exp_name, save_dir=save_dir, sfx=sfx, i_save=1
    )

    # TRAINER
    trainer = Trainer(
        train_set=train_loader,
        val_set=val_loader,
        test_set=test_loader,
        embedders=embedders,
        models=models,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        writer=writer,
        model_ckpt=model_ckpt,
        load_weight=False,
        device=device,
        chunk=1024
    )

    trainer.fit()