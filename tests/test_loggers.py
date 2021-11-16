import os
import sys
import torch
import torchvision.models as t_models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.loggers import ModelCheckPoint, Writer


exp_name = 'test'

# Check model checkpoint
model_checkpoint = ModelCheckPoint(exp_name=exp_name)

# Check writer
writer = Writer(exp_name=exp_name)
writer.save_loss(0.02, 2, 'train')
writer.save_metrics({
    'psnr': 23.3,
    'ssim': 41.2
}, 3, 'valid')

writer.save_metrics({'psnr': 23.3}, 3, 'train')

writer.save_imgs(torch.rand((5, 3, 100, 100)).to(torch.float32),
                 torch.rand((5, 3, 100, 100)).to(torch.float32),
                 1, sfx='X')

models = {
    'coarse': t_models.resnet18(),
    'fine': t_models.resnet18()
}

params = []
for k, v in models.items():
    params += list(v.parameters())

optimizer = torch.optim.Adam(params=params,
                             lr=5e-4,
                             eps=1e-8,
                             weight_decay=0)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[20],
                                                 gamma=0.1)

model_checkpoint.save(models, optimizer, scheduler, 45.5, 5)
model, optimizer, scheduler, epoch, best_psnr = ModelCheckPoint.load(
    'logs/test/checkpoint.pth', models, optimizer, scheduler
)

print(best_psnr)
