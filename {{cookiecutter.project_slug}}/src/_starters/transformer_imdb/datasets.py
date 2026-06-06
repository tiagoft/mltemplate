import re
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split

_PAD_IDX = 0
_UNK_IDX = 1


class _IMDBDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_seq_len: int):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.samples = list(zip(texts, labels))

    def _encode(self, text: str) -> torch.Tensor:
        tokens = re.findall(r"\w+", text.lower())[:self.max_seq_len]
        ids = [self.vocab.get(t, _UNK_IDX) for t in tokens]
        ids += [_PAD_IDX] * (self.max_seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        text, label = self.samples[idx]
        return {"input_ids": self._encode(text)}, label


def _build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(re.findall(r"\w+", text.lower()))
    vocab = {"<pad>": _PAD_IDX, "<unk>": _UNK_IDX}
    for token, _ in counter.most_common(vocab_size - 2):
        vocab[token] = len(vocab)
    return vocab


def get_imdb_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    from datasets import load_dataset  # noqa: PLC0415

    max_seq_len = config.get("max_seq_len", 256)
    vocab_size = config.get("vocab_size", 20000)

    raw = load_dataset("imdb")
    train_texts = raw["train"]["text"]
    train_labels = raw["train"]["label"]
    test_texts = raw["test"]["text"]
    test_labels = raw["test"]["label"]

    vocab = _build_vocab(train_texts, vocab_size)

    train_full = _IMDBDataset(train_texts, train_labels, vocab, max_seq_len)
    test_ds = _IMDBDataset(test_texts, test_labels, vocab, max_seq_len)
    train_ds, val_ds = random_split(train_full, [22500, 2500])
    return train_ds, val_ds, test_ds


_DATASET_REGISTRY: dict[str, Callable[[dict], tuple[Dataset, Dataset, Dataset]]] = {
    "imdb": get_imdb_datasets,
}


def get_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    ds_type = config.get("type", "imdb")
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
