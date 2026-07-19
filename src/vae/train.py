import argparse
import os

import torch
from tqdm import tqdm

from src.common.data import get_mnist_loaders
from src.common.utils import get_device, save_grid, set_seed
from src.vae.loss import vae_loss
from src.vae.model import VAE


def train(args):
    set_seed(args.seed)
    device = get_device()

    train_loader, _ = get_mnist_loaders(
        args.data_root, args.batch_size, args.num_workers)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "recon": 0.0, "kl": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, _ in pbar:
            x = x.to(device)

            x_hat, mu, logvar = model(x)
            total, recon, kl = vae_loss(x_hat, x, mu, logvar)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running["total"] += total.item()
            running["recon"] += recon.item()
            running["kl"] += kl.item()
            pbar.set_postfix(total=total.item(), recon=recon.item(),
                             kl=kl.item())

        n = len(train_loader)
        print(f"Epoch {epoch}: total={running['total'] / n:.2f} "
              f"recon={running['recon'] / n:.2f} kl={running['kl'] / n:.2f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                samples = model.sample(64, device)
            save_grid(samples, os.path.join(
                args.output_dir, f"samples_epoch{epoch}.png"))

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "vae.pt"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./outputs/vae")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
