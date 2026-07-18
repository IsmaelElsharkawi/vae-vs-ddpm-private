import torch.nn.functional as F


def vae_loss(x_hat, x, mu, logvar):
    """ELBO loss (beta=1). Returns total, reconstruction, and KL terms,
    each averaged per image (summed over pixels/latent dims)."""
    batch_size = x.size(0)

    recon = F.mse_loss(x_hat, x, reduction="sum") / batch_size
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / batch_size

    total = recon + kl
    return total, recon, kl
