"""NeRF data loader utils
Developer: Dong Quoc Tranh
Created at: 29/10/2021
"""
import os
import cv2
import imageio
import numpy as np
from glob import glob
from .blender_loader import spheric_pose, translate_t, rotate_theta, rotate_phi


def normalize(x):
    """Normalize a vector."""
    return x / np.linalg.norm(x)


def average_poses(poses):
    """Calculate the average pose, which is then used to center all poses. The computation is as follows:
    1. Compute the center: the average of pose centers
    2. Compute the z axis: the normalized average z axis
    3. Compute axis y': the average y axis
    4. Compute the x': y' cross product z, then normalize it as the x axis
    5. Compute the y axis: z cross product x

    Args:
        poses (ndarray | Nx4x4)

    Returns:
        pose_avg (4x4): the avg pose
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    z = normalize(poses[:, :3, 2].sum(0))
    y_ = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([view_matrix(z, y_, center), hwf], 1)
    return c2w


def load_llff_data(base_dir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_z_flat=False):
    """Load llff-type dataset.

    Args:
        base_dir (str): Directory contains dataset
        factor (int): Scale factor
        recenter (bool): Recenter poses
        bd_factor (float): Boundary factor
        spherify (bool): Whether create camera view path following spherical type or spiral type.
        path_z_flat (bool): TODO

    Returns:
        imgs (ndarray | NxHxWxC): Images
        poses (ndarray | Nx3x5
        boundaries (ndarray | Nx2)
        render_poses (ndarray | N_views x3x5)
    """
    imgs, poses, boundaries = load_data(base_dir, factor=factor)
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    print(f'Boundary --> nearest: {boundaries.min()} | farthest: {boundaries.max()}')
    print('Pose shape: ', poses.shape)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    boundaries = np.moveaxis(boundaries, -1, 0).astype(np.float32)
    scale = 1. if bd_factor is None else 1. / (boundaries.min() * bd_factor)
    poses[:, :3, 3] *= scale
    boundaries *= scale
    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, boundaries = spherify_poses(poses, boundaries)
    else:
        c2w = average_poses(poses)
        up = normalize(poses[:, :3, 1].sum(0))
        close_depth, inf_depth = boundaries.min() * 0.9, boundaries.max() * 5.
        dt = 0.75  # TODO
        mean_dz = 1. / ((1. - dt) / close_depth + dt / inf_depth)
        focal = mean_dz
        z_delta = close_depth * 0.2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_z_flat:
            z_loc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w[:3, 3] + z_loc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        render_poses = render_path_spiral(c2w_path, up, rads, focal, z_delta, z_rate=0.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)
    c2w = average_poses(poses)
    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    imgs = imgs.astype(np.float32)
    poses = poses.astype(np.float32)
    print(f'Data: poses: {poses.shape} | images: {imgs.shape} | boundaries: {boundaries.shape}')
    print('Holdout view is: ', i_test)
    return imgs, poses, boundaries, render_poses, i_test


def load_data(base_dir, factor=4):
    """Load scaled dataset.

    Args:
        base_dir (str): Directory contains dataset
        factor (int): Scale factor

    Returns:
        poses (3x5xN): Pose matrix contains both intrinsic and extrinsic parameters
        boundaries (2xN): Nearest and farthest boundaries of scenes
        imgs (HxWxCxN): Image array
    """
    pose_annot = np.load(os.path.join(base_dir, 'poses_bounds.npy'))
    poses = pose_annot[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    boundaries = pose_annot[:, -2:].transpose([1, 0])
    minify(base_dir, factor)
    sfx = '' if factor <= 1 else f'_{factor}'
    imgs = [imageio.imread(f)[..., :3] / 255. for f in get_imgs_path(os.path.join(base_dir, f'images{sfx}'))]
    poses[:2, 4, :] = np.array(imgs[0].shape[:2]).reshape([2, 1])
    poses[2, 4, :] /= float(factor)
    imgs = np.stack(imgs, -1)
    print(f'Loaded images: shape: {imgs.shape} | HxWxFocal: {poses[:, -1, 0]}')
    return imgs, poses, boundaries


def minify(base_dir, factor):
    """Minify images (Scale images down)"""
    if factor == 1 or os.path.isdir(os.path.join(base_dir, f'images_{factor}')):
        print('Already minified!')
        return
    if not os.path.isdir(os.path.join(base_dir, f'images')):
        raise ValueError('The image folder is not existing')

    minify_dir = os.path.join(base_dir, f'images_{factor}')
    os.makedirs(minify_dir, exist_ok=True)
    img_paths = get_imgs_path(os.path.join(base_dir, 'images'))
    for path in img_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), cv2.INTER_AREA)
        cv2.imwrite(path.replace('/images/', f'/images_{factor}/'), img)

    print('Minifying done!')


def get_imgs_path(base_dir, sort=True):
    """Get images' path from the directory"""
    img_names = os.listdir(base_dir)
    img_paths = [os.path.join(base_dir, name) for name in img_names]
    img_paths = [path for path in img_paths if any([path.endswith(ex) for ex
                                                    in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    if sort:
        return sorted(img_paths)
    return img_paths


def recenter_poses(poses):
    """Recenter camera poses."""
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = average_poses(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_


def view_matrix(z, up, pos):
    """Create view matrix from z axis, average y matrix and camera position"""
    z = normalize(z)
    y_ = up
    x = normalize(np.cross(y_, z))
    y = normalize(np.cross(z, x))
    return np.stack([x, y, z, pos], 1)


def calc_min_line_dist(rays_o, rays_d):
    """TODO"""
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_min_dist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ b_i.mean(0))
    return pt_min_dist


def spherify_poses(poses, boundaries):
    """TODO"""
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]
    pt_min_dist = calc_min_line_dist(rays_o, rays_d)
    center = pt_min_dist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    scale = 1./rad
    poses_reset[:, :3, 3] *= scale
    boundaries *= scale
    rad *= scale
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    rad_circle = np.sqrt(rad**2 - zh**2)
    new_poses = []
    for th in np.linspace(0., 2. * np.pi, 120):
        cam_origin = np.array([rad_circle * np.cos(th), rad_circle * np.sin(th), zh])
        up = np.array([0, 0, -1.])
        vec2 = normalize(cam_origin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = cam_origin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)],
        -1
    )
    return poses_reset, new_poses, boundaries


def render_path_spiral(c2w, up, rads, focal, z_delta, z_rate, rots, N):
    """TODO"""
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([view_matrix(z, up, c), hwf], 1))

    return render_poses
