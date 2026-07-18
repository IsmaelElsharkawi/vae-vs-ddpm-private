import torch
from torchvision import datasets, transforms


def get_cifar10_loaders(data_root="./data", batch_size=128, num_workers=4):
    """CIFAR-10 loaders normalized to [-1, 1]."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print(f"[data] Preparing CIFAR-10 in '{data_root}' "
          "(downloading if not already present)...")
    print("[data] Loading training split...")
    train_set = datasets.CIFAR10(data_root, train=True, download=True,
                                 transform=train_transform)
    print(f"[data] Training split ready: {len(train_set)} images.")
    print("[data] Loading test split...")
    test_set = datasets.CIFAR10(data_root, train=False, download=True,
                                transform=test_transform)
    print(f"[data] Test split ready: {len(test_set)} images.")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
