import torch


def render_rays(models, embedders, rays, N_samples=64, use_disp=False,
                perturb=0, noise_std=1, N_importance=0, chunk=1024 * 32,
                white_bg=False, test_mode=False):
    """Render rays by computing the output of @model applied on @rays

    Args:
        models: dictionary that contains models
        embedders: position and direction embedders
        rays (N_rays, 3 + 3 + 2): ray origins, ray directions, near and far
            depth bounds
        N_samples (int): number of points sampled in a ray by coarse model
        use_disp (bool): whether to sample in disparity space (inverse
            depth) TODO: check the way it works
        perturb: factor to perturb the sampling position on the ray
            (coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance (int): number of fine samples per ray
        chunk (int): the chunk size in batched inference
        white_bg (bool): whether the background is white
        test_mode (bool): If True, it will not do inference on coarse rgb
            to save time.

    Returns:
        result: dictionary containing final rgbs and depth maps for coarse and
            fine models.
    """
    # Get models
    coarse_model = models[0]
    xyz_embedder = embedders[0]
    dir_embedder = embedders[1]
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # (N_rays, 1)
    dir_embedded = dir_embedder(rays_d)
    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in the depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    if perturb > 0:  # perturb sampling depth (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # Get middle
        # Get intervals between sample
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], dim=-1)
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
    if test_mode:
        weights_coarse = inference(
            coarse_model, xyz_embedder, xyz_coarse_sampled, rays_d,
            dir_embedded, z_vals, chunk=chunk, noise_std=noise_std,
            weights_only=True, white_bg=white_bg
        )
        result = {
            'opacity_coarse': weights_coarse.sum(1)
        }
    else:
        rgb_coarse, depth_coarse, weights_coarse = inference(
            coarse_model, xyz_embedder, xyz_coarse_sampled, rays_d,
            dir_embedded, z_vals, chunk=chunk, noise_std=noise_std,
            weights_only=False, white_bg=white_bg
        )
        result = {
            'rgb_coarse': rgb_coarse,
            'depth_coarse': depth_coarse,
            'opacity_coarse': weights_coarse.sum(1)
        }

    if N_importance > 0:
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1: -1],
                             N_importance, det=(perturb == 0)).detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)
        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = inference(
            model_fine, xyz_embedder, xyz_fine_sampled, rays_d, dir_embedded,
            z_vals, chunk=chunk, noise_std=noise_std, weights_only=False,
            white_bg=white_bg
        )
        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result


def inference(model, xyz_embedder, xyz_, dir_, dir_embedded, z_vals,
              chunk=1024*32, noise_std=1, weights_only=False, white_bg=False):
    """Helper function that performs model inference

    Args:
        model: NeRF model (coarse or fine)
        xyz_embedder: embedding module for xyz
        xyz_ (N_rays, N_samples_, 3): sampled positions
            N_samples_ is the number of sampled points in each ray
                = N_samples for coarse model
                = N_samples + N_importance for fine model
        dir_ (N_rays, 3): ray directions
        dir_embedded (N_rays, embed_dir_channels): embedded directions
        z_vals (N_rays, N_samples_): depths of the sampled positions
        weights_only: do inference on sigma only or not
        noise_std: factor to perturb the model's prediction of sigma
        chunk (int): the chunk size in batched inference
        white_bg (bool): whether the background is white

    Returns:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3): the final rgb image
            depth _final: (N_rays) depth map
            weights (N_rays, N_samples_): weights of each sample
    """
    N_rays, N_samples_ = xyz_.shape[:2]
    xyz_ = xyz_.view(-1, 3)  # (N-rays * N_samples_, 3)
    if not weights_only:
        dir_embedded = torch.repeat_interleave(
            dir_embedded, repeats=N_samples_, dim=0
        )  # (N_rays * N_samples_, embed_dir_channels)

    B = xyz_.shape[0]
    out_chunks = []
    for i in range(0, B, chunk):
        xyz_embedded = xyz_embedder(xyz_[i: i + chunk])
        if not weights_only:
            xyz_dir_embedded = torch.cat([
                xyz_embedded, dir_embedded[i: i + chunk]
            ], dim=1)
        else:
            xyz_dir_embedded = xyz_embedded

        out_chunks += [model(xyz_dir_embedded, sigma_only=weights_only)]

    out = torch.cat(out_chunks, dim=0)
    if weights_only:
        sigmas = out.view(N_rays, N_samples_)
    else:
        rgb_sigmas = out.view(N_rays, N_samples_, 4)
        rgbs = rgb_sigmas[..., :3]
        sigmas = rgb_sigmas[..., -1]

    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_ - 1)
    # The last delta is infinity
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1)
    deltas = torch.cat([deltas, delta_inf], dim=-1)
    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions)
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    # Compute alpha by the formula (3)
    alphas = 1 - torch.exp(- deltas * torch.relu(sigmas + noise))
    alpha_shifted = torch.cat([
        torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10
    ], -1)  # [1, a1, a2, ...]
    # weights: (N_rays, N_samples_)
    weights = alphas * torch.cumprod(alpha_shifted, -1)[:, :-1]
    weights_sum = weights.sum(1)  # (N_rays)
    if weights_only:
        return weights

    rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
    depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
    if white_bg:
        rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

    return rgb_final, depth_final, weights


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """Sample @N_importance samples from @bins with distribution defined by
    @weights.
    Hierarchical sampling (section 5.2)

    Args:
        bins (N_rays, N_samples_ + 1) where N_samples_ is the number of coarse
            samples per ray -2
        weights (N_rays, N_samples_)
        N_importance (int): the number of samples to draw from the distribution
        det (bool): deterministic or not
        eps: a small number to prevent division by zero

    Returns:
        samples: the sampled points
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)  # (N_rays, N_samples)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_sampled = torch.stack([below, above], dim=-1).view(N_rays,
                                                            2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples
