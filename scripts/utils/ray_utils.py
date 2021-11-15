import torch
from kornia import create_meshgrid


def get_ray_dirs(H, W, focal):
    """Get ray directions for all pixels in camera coord
    Args:
        H (int): image height
        W (int): image width
        focal (int): focal length

    Returns:
        directions (H, W, 2): the direction of the rays in camera coord
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack(
        [(i - W/2)/focal, -(j - H/2)/focal, -torch.ones_like(i)], -1
    )
    return directions


def get_rays(directions, c2w):
    """Get ray origin and normalized directions in world coord for all pixels
    in one image.

    Args:
        directions (H, W, 3): precomputed ray directions in camera coord
        c2w (3, 4): Transformation matrix from camera coord to world coord

    Returns:
        rays_o (HxW, 3): the origin of the rays in the world coord
        rays_d (HxW, 3): the direction of the rays in the world coord
    """
    # Rotate ray directions from camera coord to the world coord
    rays_d = directions @ c2w[:, :3].T
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in the world coord
    rays_o = c2w[:, 3].expand(rays_d.shape)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Transform rays from the world coord to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
        H (int): image height
        W (int): image width
        focal (int): focal length
        near (float): the depths of the near plane
        rays_o (N_rays, 3): the origin of the rays in the world coord
        rays_d (N_rays, 3): the direction of the rays in the world coord

    Returns:
        rays_o (N_rays, 3): the origin of the rays in NDC
        rays_d (N_rays, 3): the direction of the rays in NDC
    """
    # Shift ray origin to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]  # compute on z axis
    rays_o = rays_o + t[..., None] * rays_d
    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]
    # Projection
    o0 = -1. / (W/(2. * focal)) * ox_oz
    o1 = -1. / (H/(2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]
    d0 = -1. / (W/(2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H/(2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d
