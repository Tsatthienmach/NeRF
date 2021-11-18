import os
import cv2
import numpy as np


class VideoWriter:
    """Video writer module.

    Args:
        save_dir (str): saving directory
        exp_name (str): experiment name
        sfx (str): experiment suffix
        fps (int): frame per second
        res (tuple | WxH): video resolution
    """
    def __init__(self, exp_name, sfx='', save_dir='logs', fps=24,
                 res=(500, 500)):
        self.exp_name = exp_name
        self.save_dir = save_dir
        self.fps = fps
        self.res = res
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.sfx = f'_{sfx}' if sfx else ""
        self.rgb_writer = None
        self.depth_writer = None

    def create_videos(self, v_sfx):
        """Initialize videos"""
        W, H = self.res
        v_sfx = f'_{v_sfx}' if v_sfx else ""
        path = os.path.join(
            self.save_dir,
            f'{self.exp_name}{self.sfx}/rgb_{W}x{H}{v_sfx}.mp4'
        )
        self.rgb_writer = cv2.VideoWriter(path, self.fourcc, self.fps,
                                          self.res)
        self.depth_writer = cv2.VideoWriter(path.replace('rgb', 'depth'),
                                            self.fourcc, self.fps, self.res)

    def add_rgb_frames(self, rgb_frames):
        """Add rgb frames into video.

        Args:
            rgb_frames (tensor | BxHxWx3)
        """
        if not self.ready():
            raise ValueError('No writers exist!')
        rgb_frames = (rgb_frames.numpy() * 255).astype(np.uint8)
        for frame in rgb_frames:
            self.rgb_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def add_depth_frames(self, depth_frames):
        """Add rgb frames into video.

        Args:
            depth_frames (tensor | BxHxWx1)
        """
        if not self.ready():
            raise ValueError('No writers exist!')
        depth_frames = np.clip(depth_frames.numpy() * 255, 0, 255)
        depth_frames = np.tile(depth_frames, (1, 1, 1, 3)).astype(np.uint8)
        for frame in depth_frames:
            self.depth_writer.write(frame)

    def ready(self):
        return self.rgb_writer is not None and self.depth_writer is not None

    def close(self):
        self.rgb_writer.release()
        self.depth_writer.release()
