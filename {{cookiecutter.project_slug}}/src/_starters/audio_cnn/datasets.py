from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split

_DIGIT_LABELS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class _SpeechCommandsDigitsDataset(Dataset):
    """Wraps a torchaudio SPEECHCOMMANDS subset filtered to digit labels only."""

    def __init__(self, raw_dataset, mel_transform, time_frames: int):
        self.mel_transform = mel_transform
        self.time_frames = time_frames
        self.samples = [
            (waveform, sample_rate, label)
            for waveform, sample_rate, label, *_ in raw_dataset
            if label in _DIGIT_LABELS
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        waveform, sample_rate, label = self.samples[idx]
        mel = self.mel_transform(waveform)  # (1, n_mels, time)

        # Pad or trim the time axis to a fixed length
        t = mel.size(-1)
        if t < self.time_frames:
            mel = torch.nn.functional.pad(mel, (0, self.time_frames - t))
        else:
            mel = mel[..., : self.time_frames]

        return mel, _DIGIT_LABELS.index(label)


def get_audio_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    import torchaudio  # noqa: PLC0415
    import torchaudio.transforms as T  # noqa: PLC0415

    n_mels = config.get("n_mels", 64)
    time_frames = config.get("time_frames", 128)
    sample_rate = config.get("sample_rate", 16000)

    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)

    train_raw = torchaudio.datasets.SPEECHCOMMANDS(root=".data", subset="training", download=True)
    val_raw = torchaudio.datasets.SPEECHCOMMANDS(root=".data", subset="validation", download=True)
    test_raw = torchaudio.datasets.SPEECHCOMMANDS(root=".data", subset="testing", download=True)

    train_ds = _SpeechCommandsDigitsDataset(train_raw, mel_transform, time_frames)
    val_ds = _SpeechCommandsDigitsDataset(val_raw, mel_transform, time_frames)
    test_ds = _SpeechCommandsDigitsDataset(test_raw, mel_transform, time_frames)
    return train_ds, val_ds, test_ds


_DATASET_REGISTRY: dict[str, Callable[[dict], tuple[Dataset, Dataset, Dataset]]] = {
    "audio": get_audio_datasets,
}


def get_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    ds_type = config.get("type", "audio")
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
