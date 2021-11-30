import numpy as np


def normalize(v):
    """Normalize a vector"""
    return v / np.linalg.norm(v)


def min_line_dist(rays_o, rays_d):
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(
        -np.linalg.inv(
            (np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)
        ) @ (b_i).mean(0)
    )
    return pt_mindist


def spherify_pose(poses, bds):
    """Spherify poses that captured in inward-forward condition

    Args:
        poses: camera poses
        bds: pose boundaries
    """
    p34_to_44 = lambda p : np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])],
        1
    )
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]
    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = normalize(up)
    vec1 = normalize(np.cross([1., 2., 3.], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])
    ) @ p34_to_44(poses[:, :3, :4])
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1./rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    return poses_reset, bds, radcircle, zh


def create_spherical_pose(theta, radcircle, zh):
    cam_origin = np.array([radcircle * np.cos(theta),
                           radcircle * np.sin(theta),
                           zh])
    up = np.array([0, 0, -1.])
    vec2 = normalize(cam_origin)
    vec0 = normalize(np.cross(vec2, up))
    vec1 = normalize(np.cross(vec2, vec0))
    pos = cam_origin
    p = np.stack([vec0, vec1, vec2, pos], 1)
    return p


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


def create_spheric_poses(radcircle, zh, theta=(0, 180), n_poses=120):
    """Create circular poses around z axis

    Args:
        theta (tuple | (starting angle, finishing angle)): Horizontally circle
        n_poses (int): number of poses to create along the path

    Returns:
        spheric_poses (n_poses, 3, 4): the poses in the circular path
    """
    poses_spheric = []
    for th in np.linspace(theta[0]/180 * np.pi, theta[1]/180 * np.pi,
                          n_poses + 1)[:-1]:
        poses_spheric.append(create_spherical_pose(th, radcircle, zh))

    return np.stack(poses_spheric, 0)
