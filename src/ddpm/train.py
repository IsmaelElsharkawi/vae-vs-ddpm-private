import argparse
import os

import torch
from tqdm import tqdm

from src.common.data import get_cifar10_loaders
from src.common.utils import get_device, save_grid, set_seed
from src.ddpm.diffusion import GaussianDiffusion
from src.ddpm.model import UNet


def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[train] device = {device}")

    train_loader, _ = get_cifar10_loaders(
        args.data_root, args.batch_size, args.num_workers)

    model = UNet().to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, _ in pbar:
            x = x.to(device)

            loss = diffusion.p_losses(model, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch}: loss={running / len(train_loader):.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            samples = diffusion.sample(model, 64, device)
            save_grid(samples, os.path.join(
                args.output_dir, f"samples_epoch{epoch}.png"))

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "ddpm.pt"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./outputs/ddpm")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
