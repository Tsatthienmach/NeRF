import torch

from .ray import get_rays, get_ndc_rays


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1., use_viewdirs=False, ):
    """TODO"""
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        view_dirs = rays_d
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        view_dirs = torch.reshape(view_dirs, [-1, 3]).float()

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = get_ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, view_dirs], -1)



