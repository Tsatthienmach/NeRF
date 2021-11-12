import numpy as np
import torch
from kornia import create_meshgrid


def get_rays(H, W, focal, c2w):
    """Get ray directions for all pixels in camera coordinate.

    Args:
        H, W, focal: image height, width and focal length
        c2w (ndarray | TODO): camera coord to world coord

    Returns:
        TODO
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack(
        [(i - W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1
    )
    rays_d = torch.sum(directions[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Transform rays from world coord to NDC.
    NDC: space such that canvas is a cube with sides [-1, 1] in each axis

    Args:
        H (int): image height
        W (int): image width
        focal: focal length
        near (N or float): the depth of the near plane
        rays_o (ndarray | Nx3): the origin of the rays in world coord
        rays_d (ndarray | Nx3): the direction of the rays in world coord

    Returns:
        rays_o (ndarray | Nx3): the origin of the rays in NDC
        rays_d (ndarray | Nx3): the direction of the rays in NDC
    """
    # Shift ray origin to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]
    # Projection
    o0 = -1./(W/(2. * focal)) * ox_oz
    o1 = -1./(H/(2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]
    d0 = -1./(W/(2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1./(H/(2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2
    rays_o = torch.stack([o0, o1, o2], -1)  # Bx3
    rays_d = torch.stack([d0, d1, d2], -1)  # Bx3
    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """TODO"""
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]
    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d
