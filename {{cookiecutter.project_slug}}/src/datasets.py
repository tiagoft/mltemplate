import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TemplateDataset(Dataset):
    """Replace with your actual dataset implementation.

    Expected: __getitem__ returns (input_tensor, label_tensor).
    """

    def __init__(self, size: int = 1000, input_size: int = 128, num_classes: int = 10):
        self.size = size
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def split_dataset(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test subsets.

    Remainder from rounding is absorbed into the test split.
    """
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
    """Return (train_loader, val_loader, test_loader)."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
