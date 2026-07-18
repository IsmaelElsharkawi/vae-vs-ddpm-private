import os
import random

import numpy as np
import torch
from torchvision.utils import save_image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x):
    """Map images from [-1, 1] back to [0, 1] for saving."""
    return (x.clamp(-1, 1) + 1) / 2


def save_grid(images, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(denormalize(images), path, nrow=nrow)
