import os
import cv2
import numpy as np


class VideoWriter:
    """Video writer module.

    Args:
        save_dir (str): saving directory
        exp_name (str): experiment name
        sfx (str): experiment suffix
        v_sfx (str): video suffix
        fps (int): frame per second
        res (tuple | WxH): video resolution
    """
    def __init__(self, exp_name, sfx='', v_sfx='', save_dir='logs', fps=24,
                 res=(500, 500)):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        sfx = f'_{sfx}' if sfx else ""
        v_sfx = f'_{v_sfx}' if v_sfx else ""
        self.rgb_writer = cv2.VideoWriter(
            os.path.join(
                save_dir, f'{exp_name}{sfx}/rgb_{res[0]}x{res[1]}{v_sfx}.mp4'
            ), fourcc, fps, res
        )
        self.depth_writer = cv2.VideoWriter(
            os.path.join(
                save_dir, f'{exp_name}{sfx}/depth_{res[0]}x{res[1]}{v_sfx}.mp4'
            ), fourcc, fps, res
        )

    def add_rgb_frames(self, rgb_frames):
        """Add rgb frames into video.

        Args:
            rgb_frames (tensor | BxHxWx3)
        """
        rgb_frames = (rgb_frames.numpy() * 255).astype(np.uint8)
        for frame in rgb_frames:
            self.rgb_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def add_depth_frames(self, depth_frames):
        """Add rgb frames into video.

        Args:
            depth_frames (tensor | BxHxWx1)
        """
        depth_frames = np.clip(depth_frames.numpy() * 255, 0, 255)
        depth_frames = np.tile(depth_frames, (1, 1, 1, 3)).astype(np.uint8)
        for frame in depth_frames:
            self.depth_writer.write(frame)

    def close(self):
        self.rgb_writer.release()
        self.depth_writer.release()
