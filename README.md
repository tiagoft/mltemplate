# mltemplate

A [cookiecutter](https://cookiecutter.readthedocs.io) template for PyTorch machine learning projects. Generates a project with a training loop, CLI tools, checkpointing, and optional Weights & Biases logging — ready to run without boilerplate.

---

## Quickstart

**Install cookiecutter** (once):
```bash
uv tool install cookiecutter
```

**Generate a new project:**
```bash
cookiecutter https://github.com/your-org/mltemplate
```

Or, from a local clone:
```bash
git clone https://github.com/your-org/mltemplate
cookiecutter mltemplate/
```

You will be prompted for a few values:
```
project_name [My ML Project]: MNIST Classifier
project_slug [my_ml_project]: mnist_classifier
author_name [Your Name]: Ada Lovelace
author_email [you@example.com]: ada@example.com
python_version [3.13]:
```

This creates the directory `mnist_classifier/` with the following structure:
```
mnist_classifier/
├── pyproject.toml
└── src/
    ├── __init__.py
    ├── configuration.toml
    ├── datasets.py
    ├── models.py
    ├── eval.py
    ├── train.py
    └── main.py
```

**Install and explore the CLI:**
```bash
cd mnist_classifier
uv pip install -e .
mnist_classifier --help
```
```
Usage: mnist_classifier [OPTIONS] COMMAND [ARGS]...

  MNIST Classifier CLI

Commands:
  train      Train the model and save checkpoints + metrics.
  viewer     Plot training curves, or list all logged runs.
  inference  Run inference on a single input using a trained checkpoint.
```

---

## Run the template project

The generated project ships with a `TemplateDataset` that produces random tensors — useful for verifying the full pipeline works before you swap in real data.

**Train:**
```bash
mnist_classifier train
```
```
Run directory: logs/mnist_classifier_log/20240601_143022
Epochs: 100%|████████████| 10/10 [00:12<00:00]
  Saved checkpoint: checkpoint_epoch_2.pth
  Saved checkpoint: checkpoint_epoch_4.pth
  ...
Training complete.
```

Checkpoints (`.pth`) and a `metrics.jsonl` file are written to the timestamped run directory. A `config.json` snapshot is also saved there so you can always reproduce the run.

**Plot the training curve:**
```bash
mnist_classifier viewer logs/mnist_classifier_log/20240601_143022/
# Saved: logs/mnist_classifier_log/20240601_143022/training_curve.png
```

**List all logged runs:**
```bash
mnist_classifier viewer --list
```
```
              Runs in logs/mnist_classifier_log
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Run                 ┃ Epochs ┃ Final val_loss ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 20240601_143022     │     10 │         2.3021 │
└─────────────────────┴────────┴────────────────┘
```

---

## Change dataset and models: MNIST 1-layer classifier

This section walks through replacing the placeholder code with a real task: classifying handwritten digits with a single linear layer, end to end.

### Step 1 — Add torchvision

In `pyproject.toml`, add `"torchvision"` to `dependencies`:
```toml
dependencies = [
    "torch",
    "torchvision",   # ← add this
    ...
]
```

Re-install: `uv pip install -e .`

### Step 2 — Update `src/configuration.toml`

```toml
[training]
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
checkpoint_every_n_epochs = 2
log_directory = "logs"

[model]
type = "mlp"
input_size = 784        # 28 × 28 pixels, flattened
hidden_sizes = []       # empty = single linear layer (784 → 10)
output_size = 10        # digits 0–9
dropout = 0.0

[data]
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
```

### Step 3 — Replace `src/datasets.py`

Replace the `TemplateDataset` class with one that loads MNIST:

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """MNIST digits, flattened to 784-dimensional vectors."""

    def __init__(self, root: str = ".", train: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),  # (1, 28, 28) → (784,)
        ])
        self.data = datasets.MNIST(root, train=train, download=True, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, label = self.data[idx]
        return img, torch.tensor(label)


def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    from torch.utils.data import random_split
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test])


def get_dataloaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
```

### Step 4 — Update the dataset line in `src/train.py`

Near the top of `train()`, replace the dataset instantiation:

```python
# Before:
from src.datasets import TemplateDataset, get_dataloaders, split_dataset
...
dataset = TemplateDataset(
    input_size=m_cfg["input_size"],
    num_classes=m_cfg["output_size"],
)

# After:
from src.datasets import MNISTDataset, get_dataloaders, split_dataset
...
dataset = MNISTDataset()
```

### Step 5 — Train

```bash
mnist_classifier train
```

After 10 epochs on MNIST you should see validation loss dropping steadily from ~2.3 toward ~0.3–0.4.

### Step 6 — Classify an arbitrary digit

Prepare an input string from any MNIST test image:

```python
# prepare_input.py
from torchvision import datasets, transforms

mnist = datasets.MNIST(
    ".", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
    ]),
)

idx = 42
img, true_label = mnist[idx]
input_str = ",".join(f"{v:.6f}" for v in img.tolist())
print(f"True label: {true_label}")
print(f"Input string (first 20 values): {','.join(input_str.split(',')[:20])}...")
```

Run it:
```bash
python prepare_input.py
# True label: 9
# Input string (first 20 values): 0.000000,0.000000,...
```

Then run inference:
```bash
INPUT=$(python -c "
from torchvision import datasets, transforms
import torch
mnist = datasets.MNIST('.', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())]))
img, _ = mnist[42]
print(','.join(f'{v:.6f}' for v in img.tolist()))
")

mnist_classifier inference logs/mnist_classifier_log/20240601_143022/checkpoint_epoch_10.pth "$INPUT"
# → [-3.12, -1.45, 0.23, -2.1, -0.87, -1.32, -3.54, -0.45, -0.98, 4.21]
```

The output is 10 raw logits (one per digit class). The predicted digit is the index of the largest value:

```python
import torch
logits = torch.tensor([-3.12, -1.45, 0.23, -2.1, -0.87, -1.32, -3.54, -0.45, -0.98, 4.21])
print(f"Predicted: {logits.argmax().item()}")  # → 9
```

---

## Sweep

`sweep()` in `src/train.py` accepts a list of config dicts and trains one model per config, returning `(config, model, history)` for each. All runs share a W&B group (if W&B is enabled) for side-by-side comparison.

Create a `sweep_script.py` in your project root:

```python
# sweep_script.py
import tomllib
from src.train import sweep

with open("src/configuration.toml", "rb") as f:
    base = tomllib.load(f)

configs = [
    # Small MLP
    {**base, "model": {
        "type": "mlp", "input_size": 784,
        "hidden_sizes": [128], "output_size": 10, "dropout": 0.1,
    }},
    # Large MLP
    {**base, "model": {
        "type": "mlp", "input_size": 784,
        "hidden_sizes": [256, 128], "output_size": 10, "dropout": 0.2,
    }},
    # Small CNN  (expects input shape (B, 1, 28, 28) — see note below)
    {**base, "model": {
        "type": "cnn", "channels": [16, 32], "output_size": 10, "dropout": 0.1,
    }},
    # Large CNN
    {**base, "model": {
        "type": "cnn", "channels": [32, 64], "output_size": 10, "dropout": 0.2,
    }},
]

results = sweep(configs)

print(f"\n{'Model':<45} {'Final val_loss':>14}")
print("-" * 60)
for config, model, history in results:
    m = config["model"]
    label = f"{m['type']} {m.get('hidden_sizes') or m.get('channels')}"
    print(f"{label:<45} {history[-1]['val_loss']:>14.4f}")
```

Run it:
```bash
python sweep_script.py
```
```
Model                                         Final val_loss
------------------------------------------------------------
mlp [128]                                             0.3821
mlp [256, 128]                                        0.3104
cnn [16, 32]                                          0.1892
cnn [32, 64]                                          0.1543
```

> **Note for CNN input shape:** `SimpleCNN` expects `(B, 1, H, W)` tensors.
> Update `MNISTDataset` to omit the flatten transform when training CNNs:
> ```python
> transform = transforms.ToTensor()  # keeps (1, 28, 28) shape
> ```

For richer visualization (per-epoch loss curves, side-by-side comparison across all runs), enable Weights & Biases — see the next section. The sweep function automatically groups all runs under a shared W&B group.

---

## Enable Weights & Biases

W&B logging is off by default. To enable it, uncomment one line in `src/configuration.toml`:

```toml
[training]
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
checkpoint_every_n_epochs = 2
log_directory = "logs"
wandb_project = "mnist_classifier_experiments"  # ← uncomment this line
```

Make sure you are logged in (`wandb login`), then run training as usual:

```bash
mnist_classifier train
```

W&B will print a run URL:
```
wandb: Logging to https://wandb.ai/your-name/mnist_classifier_experiments/runs/abc123
```

**For sweeps:** no extra configuration needed. `sweep()` automatically groups all configs under a shared run group, which appears as a single experiment in the W&B UI where you can compare learning curves, hyperparameters, and final metrics side by side.
