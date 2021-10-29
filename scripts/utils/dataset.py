"""NeRF data loader utils
Developer: Dong Quoc Tranh
Created at: 29/10/2021
"""
import os
import json
import torch
import imageio
import numpy as np


def translate_t(t):
    """Get translation matrix"""
    return torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]).float()


def rotate_phi(phi):
    """Get x-axis rotation matrix"""
    return torch.tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]
    ]).float()


def rotate_theta(theta):
    """Get y-axis rotation matrix | correspond to """
    return torch.tensor([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ]).float()


def pose_spherical(theta, phi, t):
    """Get camera-to-world matrix"""
    c2w = translate_t(t)
    c2w = rotate_phi(phi / 180. * np.pi) @


