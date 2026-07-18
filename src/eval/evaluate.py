"""Generic evaluation entry point for image generators (VAE, DDPM, ...).

Computes FID and Inception Score for a directory of generated images using
``torchmetrics``. The FID reference can be either another image directory or
the CIFAR-10 training split.

Examples:
    # FID against CIFAR-10 train split + Inception Score.
    python -m src.eval.evaluate --generated-dir outputs/vae/samples

    # FID against a custom reference directory.
    python -m src.eval.evaluate \\
        --generated-dir outputs/ddpm/samples \\
        --reference-dir data/real_images
"""

import argparse
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from tqdm import tqdm

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class ImageFolderDataset(Dataset):
    """Loads every image in a directory as a ``[0, 1]`` tensor."""

    def __init__(self, root):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(_IMAGE_EXTS))
        if not self.paths:
            raise FileNotFoundError(f"No images found in '{root}'.")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.to_tensor(img)


class _CIFAR10Images(Dataset):
    """CIFAR-10 images in ``[0, 1]`` (no normalization) for reference stats."""

    def __init__(self, data_root, train=True):
        self.dataset = datasets.CIFAR10(
            data_root, train=train, download=True,
            transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


def _loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # normalize=True lets us feed images in [0, 1]; torchmetrics handles the
    # Inception preprocessing internally.
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception = InceptionScore(splits=args.is_splits, normalize=True).to(device)

    # Generated images feed both metrics (FID fake side + Inception Score).
    gen_ds = ImageFolderDataset(args.generated_dir)
    print(f"[eval] Scoring {len(gen_ds)} generated images "
          f"from '{args.generated_dir}'.")
    for batch in tqdm(_loader(gen_ds, args.batch_size, args.num_workers),
                      desc="generated"):
        batch = batch.to(device)
        fid.update(batch, real=False)
        inception.update(batch)

    # Reference images for the FID real side.
    if args.reference_dir:
        ref_ds = ImageFolderDataset(args.reference_dir)
        ref_desc = f"reference '{args.reference_dir}'"
    else:
        ref_ds = _CIFAR10Images(args.data_root, train=True)
        ref_desc = "reference CIFAR-10 train"
    print(f"[eval] Computing reference statistics over "
          f"{len(ref_ds)} images ({ref_desc}).")
    for batch in tqdm(_loader(ref_ds, args.batch_size, args.num_workers),
                      desc="reference"):
        fid.update(batch.to(device), real=True)

    fid_value = float(fid.compute())
    is_mean, is_std = inception.compute()
    is_mean, is_std = float(is_mean), float(is_std)

    print("\n===== Evaluation =====")
    print(f"FID:             {fid_value:.4f}")
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    return {"fid": fid_value, "is_mean": is_mean, "is_std": is_std}


def parse_args():
    p = argparse.ArgumentParser(description="FID / Inception Score evaluation.")
    p.add_argument("--generated-dir", required=True,
                   help="Directory of generated images to score.")
    p.add_argument("--reference-dir", default=None,
                   help="Reference image directory for FID. If omitted, the "
                        "CIFAR-10 train split is used.")
    p.add_argument("--data-root", default="./data",
                   help="Where CIFAR-10 is stored/downloaded.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--is-splits", type=int, default=10,
                   help="Number of splits for Inception Score variance.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
