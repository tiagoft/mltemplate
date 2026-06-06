# Starter: `audio_cnn` — CNN Classifier for Spoken Digits

Generate with:

```bash
cookiecutter https://github.com/tiagoft/mltemplate
# starting_point [template]: audio_cnn
```

Extra dependency added automatically: `torchaudio`.

---

## What you get

A convolutional classifier for the SPEECHCOMMANDS dataset, filtered to the ten spoken digit words ("zero" through "nine"). Raw audio waveforms are converted to mel spectrograms on-the-fly, then classified with a CNN identical in structure to the `cnn_mnist` starter. The dataset is downloaded automatically on first run (~2.3 GB).

```bash
cd my_project
uv pip install -e .
my_project train
```

---

## Architecture — `AudioCNN`

Input shape: `(B, 1, n_mels, time_frames)` — a single-channel image of the spectrogram.

```
Conv2d(1 → channels[0], 3×3, pad=1) → ReLU → MaxPool2d(2) → Dropout2d
Conv2d(channels[0] → channels[1], 3×3, pad=1) → ReLU → MaxPool2d(2) → Dropout2d
...
AdaptiveAvgPool2d(1, 1)    # works for any (n_mels, time_frames) size
Flatten
Linear(channels[-1], output_size)
```

`AdaptiveAvgPool2d` makes the model resolution-agnostic — you can change `n_mels` or `time_frames` in the config and the model code requires no update.

---

## Configuration

```toml
[training]
batch_size = 64
num_epochs = 20
learning_rate = 1e-3
checkpoint_every_n_epochs = 5
log_directory = "logs"

[[model]]
type = "audio_cnn"
channels = [32, 64]    # convolutional channels (one block per entry)
output_size = 10       # ten digit classes
dropout = 0.2

[[dataset]]
name = "spoken_digits"
type = "audio"
n_mels = 64            # mel filterbank bins (frequency axis)
time_frames = 128      # fixed time axis length (frames padded/trimmed)
sample_rate = 16000    # SPEECHCOMMANDS native rate
```

---

## Dataset

Loaded via `torchaudio.datasets.SPEECHCOMMANDS` using the built-in `subset` argument (`"training"`, `"validation"`, `"testing"`), so train/val/test splits are determined by the dataset itself (no manual splitting). Samples are filtered to the ten digit label words.

Each waveform is passed through `torchaudio.transforms.MelSpectrogram(sample_rate, n_mels)`. The time axis is then padded with zeros or trimmed to `time_frames`, giving a fixed-shape tensor. Items returned:

```python
(FloatTensor(1, n_mels, time_frames), digit_index)   # digit_index in 0–9
```

---

## Extending

**Deeper network:**

```toml
[[model]]
type = "audio_cnn"
channels = [32, 64, 128]
output_size = 10
dropout = 0.3
```

**Compare spectrogram resolutions:**

```toml
[[dataset]]
name = "coarse"
type = "audio"
n_mels = 32
time_frames = 64
sample_rate = 16000

[[dataset]]
name = "fine"
type = "audio"
n_mels = 64
time_frames = 128
sample_rate = 16000
```

```bash
my_project sweep
```

**Classify all 35 SPEECHCOMMANDS labels** (not just digits): remove the `_DIGIT_LABELS` filter in `src/datasets.py` and set `output_size = 35` in `[[model]]`.

**Add audio augmentation:** apply `torchaudio.transforms.FrequencyMasking` or `TimeMasking` inside `__getitem__` for data augmentation during training. No model changes needed.
