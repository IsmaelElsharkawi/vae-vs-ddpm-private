import torch
from torchvision import datasets, transforms

# MNIST is grayscale (1 channel) and resized to 32x32 so the models operate on
# a convenient power-of-two spatial size.
CHANNELS = 1
IMAGE_SIZE = 32


def get_mnist_loaders(data_root="./data", batch_size=128, num_workers=4):
    """Return (train_loader, test_loader) for MNIST, normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    print(f"[data] Preparing MNIST in '{data_root}' "
          "(downloading if not already present)...")
    train_set = datasets.MNIST(data_root, train=True, download=True,
                               transform=transform)
    print(f"[data] Training split ready: {len(train_set)} images.")
    test_set = datasets.MNIST(data_root, train=False, download=True,
                              transform=transform)
    print(f"[data] Test split ready: {len(test_set)} images.")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
