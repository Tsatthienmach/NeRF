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
from scripts.utils.video_utils import VideoWriter
from scripts.utils.opt import get_opts


if __name__ == '__main__':
    params = get_opts()
    print('Params: ', params)

    # PARAMS
    if torch.cuda.is_available() and params.gpu > -1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print('DEVICE: ', device)

    # LOSS
    loss = loss_dict[params.loss]()

    # DATASET
    dataset_module = dataset_dict[params.data_type]
    train_set = dataset_module(
        root_dir=params.data_dir, split='train', img_wh=params.img_wh,
        spheric_poses=params.spheric, transforms=T.Compose([T.ToTensor()]),
        res_factor=params.res_factor, val_step=params.val_step,
        n_poses=params.N_poses, white_bg=params.white_bg,
        render_train=params.render_train
    )
    val_set = dataset_module(
        root_dir=params.data_dir, split='val', img_wh=params.img_wh,
        spheric_poses=params.spheric, transforms=T.Compose([T.ToTensor()]),
        res_factor=params.res_factor, val_step=params.val_step,
        n_poses=params.N_poses, white_bg=params.white_bg
    )
    test_set = dataset_module(
        root_dir=params.data_dir, split='test', img_wh=params.img_wh,
        spheric_poses=params.spheric, transforms=T.Compose([T.ToTensor()]),
        res_factor=params.res_factor, val_step=params.val_step,
        n_poses=params.N_poses, white_bg=params.white_bg
    )
    train_loader = DataLoader(
        train_set, shuffle=True, num_workers=0, batch_size=params.batch_size,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, shuffle=False, num_workers=0, batch_size=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, shuffle=False, num_workers=0, batch_size=1, pin_memory=True
    )

    # MODELS
    embedders = {
        'pos': Embedder(N_freqs=params.pos_freqs,
                        in_channels=params.in_channels,
                        log_scale=params.log_scale),
        'dir': Embedder(N_freqs=params.dir_freqs,
                        in_channels=params.in_channels,
                        log_scale=params.log_scale)
    }
    models = {
        'coarse': NeRF(
            D=params.depth, W=params.hid_layers,
            in_channels_xyz=params.in_channels * (1 + 2*params.pos_freqs),
            in_channels_dir=params.in_channels * (1 + 2*params.dir_freqs),
            skips=params.skips
        ),
        'fine': NeRF(
            D=params.depth, W=params.hid_layers,
            in_channels_xyz=params.in_channels * (1 + 2 * params.pos_freqs),
            in_channels_dir=params.in_channels * (1 + 2 * params.dir_freqs),
            skips=params.skips
        ) if params.N_importance > 0 else None
    }

    # OPTIMIZER AND LR SCHEDULER
    parameters = []
    for name, model in models.items():
        parameters += list(model.parameters())

    optimizer = Adam(parameters, lr=params.lr, eps=params.eps,
                     weight_decay=params.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)

    # METRICS
    metrics = {
        'psnr': PSNR(),
        'ssim': SSIM()
    }

    # LOGGERS
    writer = Writer(
        exp_name=params.exp_name, save_dir=params.log_dir, sfx=params.exp_sfx,
        i_image=params.i_image
    )
    model_ckpt = ModelCheckPoint(
        exp_name=params.exp_name, save_dir=params.log_dir, sfx=params.exp_sfx,
        i_save=params.i_save
    )
    video_writer = VideoWriter(exp_name=params.exp_name, sfx=params.exp_sfx,
                               save_dir=params.log_dir, fps=params.fps,
                               res=params.img_wh)

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
        device=device,
        model_ckpt=model_ckpt,
        video_writer=video_writer,
        N_samples=params.N_samples,
        N_importance=params.N_importance,
        chunk=params.chunk,
        epochs=params.num_epochs,
        perturb=params.perturb,
        noise_std=params.noise_std,
        use_disp=params.use_disp,
        white_bg=params.white_bg,
        i_test=params.i_test,
        weight=params.weight,
        load_weight=params.load_weight,
        test_info=test_set.test_info,
        i_batch_save=params.i_batch_save
    )
    if params.render_train:
        trainer.render(train_set)
        exit()

    if not params.eval:
        trainer.fit()
    else:
        trainer.evaluate()

