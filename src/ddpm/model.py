import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the (integer) timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock(nn.Module):
    """Residual block conditioned on the time embedding."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """A small U-Net that predicts the noise added to a 32x32 image."""

    def __init__(self, in_ch=3, base=64, ch_mults=(1, 2, 2), time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        chs = [base * m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # Downsampling path.
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base
        for ch in chs:
            self.downs.append(ResBlock(prev, ch, time_dim))
            self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))
            prev = ch

        # Bottleneck.
        self.mid = ResBlock(prev, prev, time_dim)

        # Upsampling path (mirrors the downs, with skip connections).
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for ch in reversed(chs):
            self.upsamples.append(
                nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1))
            self.ups.append(ResBlock(ch * 2, ch, time_dim))
            prev = ch

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))

        h = self.in_conv(x)
        skips = []
        for block, down in zip(self.downs, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid(h, t_emb)

        for up, block, skip in zip(self.upsamples, self.ups, reversed(skips)):
            h = up(h)
            h = block(torch.cat([h, skip], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
