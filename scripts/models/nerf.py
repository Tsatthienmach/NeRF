import torch
from torch import nn


class NeRF(nn.Module):
    """Neural Radiance Field module

    Args:
        D (int): number of layers for density (sigma) encoder
        W (int): number of hidden units in each layer
        in_channels_xyz (int): number of input channels for xyz
            (3 + 3*10*2 = 63), default 10 is referred to paper
        in_channels_dir (int): number of input channels for direction
            (3 + 3*4*2) = 27, default 4 is referred to paper
        skips (list): add skip connection in the D-th layer
    """
    def __init__(self,
                 D=8,
                 W=256,
                 in_channels_xyz=63,
                 in_channels_dir=27,
                 skips=[4]):
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)

            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f'xyz_encoding_{i+1}', layer)

        self.xyz_encoding_final = nn.Linear(W, W)
        # direction encoding layers
        self.dir_encoding = nn.Sequential(nn.Linear(W + in_channels_dir, W//2),
                                          nn.ReLU(True))
        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(nn.Linear(W//2, 3),
                                 nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """Encode input (xyz + dir) to rgb+sigma

        Args:
            x (B, self.in_channels_xyz [+ self.in_channels_dir])
            sigma_only (bool): whether to infer sigma only

        Returns:
            if sigma_only:
                sigma (B, 1): sigma
            else:
                out (B, 4): rgb + sigma
        """
        if not sigma_only:
            input_xyz, input_dir = torch.split(
                x, [self.in_channels_xyz, self.in_channels_dir], dim=-1
            )
        else:
            input_xyz = x

        xyz = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz = torch.cat([input_xyz, xyz], dim=-1)
            xyz = getattr(self, f'xyz_encoding_{i+1}')(xyz)

        sigma = self.sigma(xyz)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], dim=-1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        out = torch.cat([rgb, sigma], dim=-1)
        return out
