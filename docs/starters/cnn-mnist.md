# Starter: `cnn_mnist` — CNN Classifier for MNIST

Generate with:

```bash
cookiecutter https://github.com/tiagoft/mltemplate
# starting_point [template]: cnn_mnist
```

Extra dependency added to `pyproject.toml` automatically: `torchvision`.

---

## What you get

A convolutional classifier that trains on the MNIST handwritten digits dataset (28×28 grayscale images, 10 classes). The dataset is downloaded automatically on first run.

```bash
cd my_project
uv pip install -e .
my_project train
```

After ~10 epochs you should see validation loss dropping from ~2.3 toward ~0.3.

---

## Architecture — `MNISTClassifier`

Input shape: `(B, 1, 28, 28)`.

```
Conv2d(1 → channels[0], 3×3, pad=1) → ReLU → MaxPool2d(2) → Dropout2d
Conv2d(channels[0] → channels[1], 3×3, pad=1) → ReLU → MaxPool2d(2) → Dropout2d
...
AdaptiveAvgPool2d(1, 1)    # works for any spatial size
Flatten
Linear(channels[-1], output_size)
```

`AdaptiveAvgPool2d` makes the classifier resolution-agnostic — you can change `n_mels`/`time_frames` in the config without touching the model code.

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
output_size = 10       # number of classes
dropout = 0.2

[[dataset]]
name = "mnist"
type = "mnist"         # loads torchvision MNIST
```

`channels` controls depth and width. Adding more entries (e.g. `[32, 64, 128]`) adds a third conv block.

---

## Dataset

Loaded via `torchvision.datasets.MNIST` with `ToTensor()`. The 60 000-sample training set is split 50 000 / 10 000 (train / val). The standard 10 000-sample test set is held out.

Items returned: `(FloatTensor(1, 28, 28), int_label)` — no flattening, the CNN expects 2-D spatial input.

---

## Extending

**Add a deeper network:**

```toml
[[model]]
type = "mnist_cnn"
channels = [32, 64, 128]
output_size = 10
dropout = 0.3
```

**Sweep over channel configurations:**

```toml
[[model]]
type = "mnist_cnn"
channels = [16, 32]
output_size = 10
dropout = 0.1

[[model]]
type = "mnist_cnn"
channels = [32, 64]
output_size = 10
dropout = 0.2
```

```bash
my_project sweep
```

**Switch to Fashion-MNIST:** replace `MNIST` with `FashionMNIST` in `src/datasets.py` — same shape, same 10 classes, drop-in replacement.
