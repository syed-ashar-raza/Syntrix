from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_fashion_mnist(
    root: Path,
    batch_size: int = 128,
    num_workers: int = 0,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    train = datasets.FashionMNIST(
        root=str(root),
        train=True,
        download=True,
        transform=transform,
    )

    test = datasets.FashionMNIST(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )

    use_pin_memory = torch.cuda.is_available()

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": num_workers > 0,
    }

    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        ),
        DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        ),
    )
