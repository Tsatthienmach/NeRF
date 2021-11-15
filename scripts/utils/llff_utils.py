import numpy as np


def normalize(v):
    """Normalize a vector"""
    return v / np.linalg.norm(v)


def average_pose(poses):
    """Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
        1. Compute the center: the average of pose centers.
        2. Compute the z axis: the normalized average z axis.
        3. Compute axis y': the average y axis.
        4. Compute x' = y' cross product z, then normalize it as the x axis.
        5. Compute the y axis: z cross product x.

        Note that at step 3, we cannot directly use y' as y axis since it's
        not necessarily orthogonal to z axis. We need to pass from x to y.

    Args:
        poses: (N_images, 3, 4)

    Returns:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg


def center_poses(poses):
    """Center the poses so that we can use NDC

    Args:
        poses (N_images, 3, 4)

    Returns:
        poses_centered (N_images, 3, 4): the centered poses
        pose_avg (3, 4): the average pose
    """
    pose_avg = average_pose(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    # Convert to homo coord for faster computation
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4)
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """Compute poses that follow a spiral path for rendering purpose.
    https://tinyurl.com/ybgtfns3

    Args:
        radii (3): radii of the spiral for each axis
        focus_depth (float): the depth that the spiral poses look at
        n_poses (int): number of poses to create along the path

    Returns:
        poses_spiral (n_poses, 3, 4): the poses in the spiral path
    """
    poses_spiral = []
    for t in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        # The parametric function of spiral
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
        # The viewing z axis is the vector pointing from @focus_depth plane to
        # @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        # Compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])
        x = normalize(np.cross(y_, z))
        y = np.cross(z, x)
        poses_spiral.append(np.stack([x, y, z, center], 1))

    return np.stack(poses_spiral, 0)


def spheric_pose(theta, phi, radius):
    trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -0.9 * t],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ])

    rot_phi = lambda ph: np.array([
        [1, 0, 0, 0],
        [0, np.cos(ph), -np.sin(ph), 0],
        [0, np.sin(ph), np.cos(ph), 0],
        [0, 0, 0, 1],
    ])

    rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ])

    c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    c2w = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]) @ c2w
    return c2w[:3]


def create_spheric_poses(radius, phi=30, theta=(0, 180), n_poses=120):
    """Create circular poses around z axis

    Args:
        radius: the (negative) height and the radius of the circle
        phi (angle): Vertically circle
        theta (tuple | (starting angle, finishing angle)): Horizontally circle
        n_poses (int): number of poses to create along the path

    Returns:
        spheric_poses (n_poses, 3, 4): the poses in the circular path
    """
    poses_spheric = []
    for th in np.linspace(theta[0], theta[1]/180 * np.pi, n_poses + 1)[:-1]:
        poses_spheric.append(spheric_pose(th, phi/180 * np.pi, radius))

    return np.stack(poses_spheric, 0)
