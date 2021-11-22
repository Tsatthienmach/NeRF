import torch
from torch import nn


class Embedder(nn.Module):
    """Embed nerf input to higher dimension (R --> R ** (2L))

    Args:
        N_freqs (int): The frequency of embedding (L)
        in_channels (int): number of input channels (3 for both xyz and direction)
        log_scale (bool): If True.
    """
    def __init__(self, N_freqs, in_channels, log_scale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        if log_scale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """Embed x to (x, sin(2^k * x), sin(2^k * x), ...)

        Args:
            x (B, self.in_channels): input (xyz/dir)

        Returns:
            encoded embedding (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, -1)
