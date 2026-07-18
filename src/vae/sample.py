import argparse
import os

import torch

from src.common.utils import denormalize, get_device, save_grid
from src.vae.model import VAE


def sample(args):
    device = get_device()

    model = VAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Grid preview.
    with torch.no_grad():
        grid = model.sample(64, device)
    save_grid(grid, os.path.join(args.output_dir, "sample_grid.png"))

    # Bulk export for FID / IS.
    from torchvision.utils import save_image
    os.makedirs(args.samples_dir, exist_ok=True)
    remaining = args.num_samples
    idx = 0
    while remaining > 0:
        n = min(args.batch_size, remaining)
        with torch.no_grad():
            imgs = denormalize(model.sample(n, device))
        for i in range(n):
            save_image(imgs[i],
                       os.path.join(args.samples_dir, f"{idx:05d}.png"))
            idx += 1
        remaining -= n
    print(f"Saved {idx} samples to {args.samples_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./outputs/vae/vae.pt")
    p.add_argument("--output-dir", default="./outputs/vae")
    p.add_argument("--samples-dir", default="./outputs/vae/samples")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    sample(parse_args())
