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
    # 1. Compute the center of camera
    center = poses[:, :3, 3].mean(0)
    # 2. Compute the z axis
    z = normalize(poses[:, :3, 2].sum(0))
    # 3. Compute the y' axis
    y_ = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([create_view_matrix(z, y_, center), hwf], 1)
    print(poses.shape,c2w.shape)
    exit()
    return c2w


def create_view_matrix(z, y_, pos):
    """Create view matrix from z axis, average y axis and position.

    Args:
        z (ndarray | 3): z axis
        y_ (ndarray | 3): average y axis
        pos (ndarray | 3): camera position

    Returns:
        view matrix (ndarray | 3x4) : camera coord to world coord
    """
    z = normalize(z)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))
    # 5. Compute the y axis (z and x are normalized, so y is already normalized)
    y = normalize(np.cross(z, x))
    return np.stack([x, y, z, pos], 1)


def center_poses(poses):
    """Center the poses so that we can use NDC.

    Args:
        poses (ndarray | Nx4x4)

    Returns:
        centered_poses (ndarray | Nx3x4)
        avg_pose (ndarray | 3x4)
    """
    avg_pose = average_poses(poses)
    homo_avg_pose = np.eye(4)
    homo_avg_pose[:3] = avg_pose  # Homogeneous coordinate for faster computation
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # Nx1x4
    homo_poses = np.concatenate([poses, last_row], 1)  # Nx4x4
    centered_poses = np.linalg.inv(homo_avg_pose) @ homo_poses  # Nx4x4
    centered_poses = centered_poses[:, :3]

    last_row = np.reshape([0, 0, 0, 1], [1, 4])
    avg_pose = average_poses(poses)

    return centered_poses, np.linalg.inv(homo_avg_pose)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """Computes poses that follow a spiral path for rendering purpose.
    Refer to: https://tinyurl.com/ybgtfns3

    Args:
        radii (ndarray | 3): the radii of the spiral for each axis
        focus_depth (float): the depth that the spiral poses look at ?TODO
        n_poses (int): number of poses create along the path

    Returns:
        spiral_poses (n_poses x3x4): the poses in the spiral path
    """
    spiral_poses = []
    for t in np.linspace(0, 2*np.pi, n_poses + 1)[:-1]:
        # The parametric function
        camera_center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii
        # The viewing z axis is the vector pointing from the *focus_depth plane* to *center*
        z = normalize(camera_center - np.array([0, 0, -focus_depth]))
        # Compute other axes as in *avg_poses*
        y_ = np.array(0, 1, 0)
        x = normalize(np.cross(y_, z))
        y = np.cross(z, x)
        spiral_poses += [np.stack([x, y, z, camera_center], 1)]  # 3x4

    return np.stack(spiral_poses, 0)  # n_poses x3x4


def create_spheric_poses(radius, theta, phi=np.pi/5, n_poses=120):
    """Create circular poses around z axis.

    Args:
        radius (float): near / far controller
        theta (degrees): horizontal angle. Ex: 30
        phi (degrees |): vertical angle
        n_poses (int): number of poses

    Returns:
        spherical poses (n_poses x3x4)
    """
    spherical_poses = []
    for theta in np.linspace(-theta, theta, n_poses + 1)[:-1]:
        spherical_poses += [spheric_pose(theta, phi, radius)]

    return np.stack(spherical_poses, 0)


def load_llff_data(base_dir, res_factor=1):
    """Load llff-type dataset

    Args:
        base_dir (str): Directory which contains dataset
        res_factor (int): Image scaling factor

    Returns:
        imgs (ndarray | NxHxWx3)
        poses (ndarray | Nx3x4): Camera extrinsic parameters (transform matrix)
        [H, W, focal, bounds]: Camera intrinsic parameters
    """
    sfx = f'_{res_factor}' if res_factor != 1 else ''
    poses_bounds = np.load(os.path.join(base_dir, 'poses_bounds.npy'))  # Nx17
    img_paths = sorted(glob(os.path.join(base_dir, f'images{sfx}/*')))
    assert len(poses_bounds) == len(img_paths), "Mismatch between num. imgs and num. poses"
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # Nx3x5
    bounds = poses_bounds[:, -2:]  # Nx2
    # Step 1: rescale focal length according to resolution
    H, W, focal = poses[0, :, -1]  # Intrinsics, same for all images
    focal /= res_factor
    H = H // res_factor
    W = W // res_factor
    # Step 2: Correct poses | From "down right back" to "right up back"
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., 0:1], poses[..., 2:4]],
        -1
    )  # Nx3x4
    poses, avg_poses = center_poses(poses)
    # Step 3: Correct scale so that the nearest depth is at a little more than 1.0
    near_original = bounds.min()
    scale_factor = near_original * 0.75
    bounds /= scale_factor
    poses[..., 3] /= scale_factor
    poses = np.array(poses).astype(np.float32)
    return img_paths, poses, [int(H), int(W), focal, bounds]
