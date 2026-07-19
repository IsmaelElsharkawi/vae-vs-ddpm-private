"""Minimal Gaussian diffusion (DDPM) with a linear beta schedule.

Implements the two things a DDPM needs:
  * a training loss: predict the noise added to a clean image, and
  * ancestral sampling: iteratively denoise pure noise into an image.
"""

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def _gather(values, t, shape):
    """Pick per-sample values at timesteps ``t`` and broadcast to ``shape``."""
    out = values.gather(0, t)
    return out.view(-1, *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Posterior variance used during sampling.
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        """Add ``t`` steps of noise to a clean image ``x0`` (forward process)."""
        sqrt_ac = _gather(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = _gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_losses(self, model, x0):
        """Training loss: MSE between true and predicted noise."""
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        predicted = model(x_noisy, t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t):
        """One reverse (denoising) step at timestep ``t``."""
        betas_t = _gather(self.betas, t, x.shape)
        sqrt_om = _gather(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas = _gather(
            torch.rsqrt(self.alphas), t, x.shape)

        mean = sqrt_recip_alphas * (x - betas_t * model(x, t) / sqrt_om)

        if t[0].item() == 0:
            return mean
        noise = torch.randn_like(x)
        var = _gather(self.posterior_variance, t, x.shape)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, n, device, image_size=32, channels=3):
        """Generate ``n`` images by denoising from pure Gaussian noise."""
        x = torch.randn(n, channels, image_size, image_size, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x
