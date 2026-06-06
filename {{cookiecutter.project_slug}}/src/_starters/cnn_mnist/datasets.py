from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def get_mnist_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=".data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=".data", train=False, download=True, transform=transform)
    train_ds, val_ds = random_split(train_full, [50000, 10000])
    return train_ds, val_ds, test_ds


_DATASET_REGISTRY: dict[str, Callable[[dict], tuple[Dataset, Dataset, Dataset]]] = {
    "mnist": get_mnist_datasets,
}


def get_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    ds_type = config.get("type", "mnist")
    if ds_type not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {ds_type!r}. Available: {list(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[ds_type](config)


def split_dataset(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[Dataset, Dataset, Dataset]:
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test])


def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
