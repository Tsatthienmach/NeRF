"""NeRF data loader
Developer: Dong Quoc Tranh
Created at: 29/10/2021
"""
import os
import json
from torch.utils.data import Dataset


class NeRFDataset(Dataset):
    """NeRF dataset.
    """

    def __init__(self, pos_emd, view_dir_emb, data_dir, transform_path=None, data_type='synthetic'):
        """
        Args:
            pos_emd (class): Position embedding class
            view_dir_emb (class): View direction embedding class
            data_dir (str): Dataset base directory
            data_type (str | 'synthetic'/'llff'): Two types of dataset.
        """
        super(NeRFDataset, self).__init__()
        self.pos_embed = pos_emd
        self.view_dir_emb = view_dir_emb
        self.data_dir = data_dir
        self.data_type = data_type

        if data_type == 'synthetic':
            if not transform_path:
                raise ValueError(f'The synthetic dataset needs transform information, got {transform_path} path.')
            with open(os.path.join(data_dir, transform_path), 'r') as f:
                frame_info, camera_angle_x = json.load(f)['frames'], json.load(f)['camera_angle_x']

            self.transforms =
            pass
        elif data_type == 'llff':
            pass
        else:
            raise ValueError('Inappropriate data type')

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
