from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class _ChunkIMDBDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_chunks: int,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.samples = list(zip(texts, labels))

    def _tokenize_and_chunk(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(text, truncation=False, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)

        stride = self.chunk_size - self.chunk_overlap
        chunks_ids: list[torch.Tensor] = []
        chunks_mask: list[torch.Tensor] = []

        pos = 0
        while len(chunks_ids) < self.max_chunks and pos < ids.size(0):
            c_ids = ids[pos : pos + self.chunk_size]
            c_mask = mask[pos : pos + self.chunk_size]
            if c_ids.size(0) < self.chunk_size:
                pad = self.chunk_size - c_ids.size(0)
                c_ids = torch.cat([c_ids, torch.zeros(pad, dtype=torch.long)])
                c_mask = torch.cat([c_mask, torch.zeros(pad, dtype=torch.long)])
            chunks_ids.append(c_ids)
            chunks_mask.append(c_mask)
            pos += stride

        # Pad with empty chunks if document is shorter than max_chunks
        while len(chunks_ids) < self.max_chunks:
            chunks_ids.append(torch.zeros(self.chunk_size, dtype=torch.long))
            chunks_mask.append(torch.zeros(self.chunk_size, dtype=torch.long))

        return torch.stack(chunks_ids), torch.stack(chunks_mask)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        text, label = self.samples[idx]
        ids, mask = self._tokenize_and_chunk(text)
        return {"input_ids": ids, "attention_mask": mask}, label


def get_bert_chunks_imdb_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    from datasets import load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    bert_model_name = config.get("bert_model_name", "bert-base-uncased")
    max_chunks = config.get("max_chunks", 4)
    chunk_size = config.get("chunk_size", 128)
    chunk_overlap = config.get("chunk_overlap", 0)

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    raw = load_dataset("imdb")

    train_full = _ChunkIMDBDataset(
        raw["train"]["text"], raw["train"]["label"], tokenizer, max_chunks, chunk_size, chunk_overlap
    )
    test_ds = _ChunkIMDBDataset(
        raw["test"]["text"], raw["test"]["label"], tokenizer, max_chunks, chunk_size, chunk_overlap
    )
    train_ds, val_ds = random_split(train_full, [22500, 2500])
    return train_ds, val_ds, test_ds


_DATASET_REGISTRY: dict[str, Callable[[dict], tuple[Dataset, Dataset, Dataset]]] = {
    "bert_chunks_imdb": get_bert_chunks_imdb_datasets,
}


def get_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    ds_type = config.get("type", "bert_chunks_imdb")
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
