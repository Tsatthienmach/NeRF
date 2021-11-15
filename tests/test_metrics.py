import os
import sys
import torch
from torch.nn import MSELoss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.metrics.metrics import MSE, PSNR


gt = torch.rand(2, 2, 3).to(torch.float32)
pred = torch.rand(2, 2, 3).to(torch.float32)

# MSE testing
mse = MSE()
torch_MSE = MSELoss()

# Test one pair
mse.update(pred, gt)
assert mse.compute() == torch_MSE(pred, gt)
mse.reset()

# Test multiple pairs
for i in range(5):
    mse.update(pred, gt)

assert mse.compute().data == torch_MSE(torch.tile(pred, [5, 1, 1, 1]),
                                       torch.tile(gt, [5, 1, 1, 1])).data

# Test PSNR
psnr = PSNR()
psnr.update(pred, gt)
print(psnr.compute())
psnr.reset()

for i in range(5):
    psnr.update(pred, gt)

print(psnr.compute())
