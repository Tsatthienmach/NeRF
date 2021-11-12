import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.datasets.dataloader import LLFFDataset


llff_dataset = LLFFDataset('.data/nerf_llff_data/fern', res_factor=4)


# # Test the corresponding between images, poses, and boundaries
# assert len(img_paths) == len(poses)
# assert len(o_img_paths) == len(o_poses)
#
# # Test the appropriate dim of images, poses, boundaries and rendered poses
# assert poses.ndim == 3
# assert bounds.ndim == 2
#
# # Test the length of dim
# assert poses.shape[-2:] == (3, 4)
# assert bounds.shape[-1] == 2
#
# # Test res_factor
# assert H == o_H // 4
# assert W == o_W // 4
