from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class _BERTIMDBDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = list(zip(texts, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }, label


def get_bert_imdb_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    from datasets import load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    bert_model_name = config.get("bert_model_name", "bert-base-uncased")
    max_seq_len = config.get("max_seq_len", 128)

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    raw = load_dataset("imdb")

    train_full = _BERTIMDBDataset(raw["train"]["text"], raw["train"]["label"], tokenizer, max_seq_len)
    test_ds = _BERTIMDBDataset(raw["test"]["text"], raw["test"]["label"], tokenizer, max_seq_len)
    train_ds, val_ds = random_split(train_full, [22500, 2500])
    return train_ds, val_ds, test_ds


_DATASET_REGISTRY: dict[str, Callable[[dict], tuple[Dataset, Dataset, Dataset]]] = {
    "bert_imdb": get_bert_imdb_datasets,
}


def get_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    ds_type = config.get("type", "bert_imdb")
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
