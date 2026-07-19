import argparse
import os

import torch
from torchvision.utils import save_image

from src.common.utils import denormalize, get_device, save_grid
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.model import UNet


def sample(args):
    device = get_device()

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    # Grid preview.
    grid = diffusion.sample(model, 64, device)
    save_grid(grid, os.path.join(args.output_dir, "sample_grid.png"))

    # Bulk export for FID / IS.
    os.makedirs(args.samples_dir, exist_ok=True)
    remaining = args.num_samples
    idx = 0
    while remaining > 0:
        n = min(args.batch_size, remaining)
        imgs = denormalize(diffusion.sample(model, n, device))
        for i in range(n):
            save_image(imgs[i],
                       os.path.join(args.samples_dir, f"{idx:05d}.png"))
            idx += 1
        remaining -= n
    print(f"Saved {idx} samples to {args.samples_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./outputs/ddpm/ddpm.pt")
    p.add_argument("--output-dir", default="./outputs/ddpm")
    p.add_argument("--samples-dir", default="./outputs/ddpm/samples")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    sample(parse_args())
