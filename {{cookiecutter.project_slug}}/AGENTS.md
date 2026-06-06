# Agent Guide — {{ cookiecutter.project_name }}

This file is written for LLMs and code-generating tools. It describes the project structure, the exact interfaces that must be satisfied when extending it, and a complete worked example.

---

## Project layout

| File | Role |
| --- | --- |
| `src/configuration.toml` | All hyperparameters. `[[model]]` blocks define models to train. |
| `src/datasets.py` | Dataset class + `split_dataset` + `get_dataloaders`. Replace the dataset class here. |
| `src/models.py` | Model classes. Add new architectures here. |
| `src/eval.py` | `evaluate(model, dataloader, criterion, device) -> dict`. Runs validation. |
| `src/train.py` | `train(config, callbacks)`, `sweep(configs)`, `_MODEL_REGISTRY`. |
| `src/main.py` | CLI: `train`, `sweep`, `viewer`, `inference`. Do not modify unless adding a new command. |
| `pyproject.toml` | Dependencies. Add libraries here (e.g. `torchvision`). |

---

## Extension contracts

### 1. Replace the dataset

Edit `src/datasets.py`. Replace the `TemplateDataset` class. The new class must satisfy:

- Inherits from `torch.utils.data.Dataset`
- `__len__() -> int`
- `__getitem__(idx) -> tuple[Tensor, Tensor]` — always a two-element tuple

The second element is used as `targets` in the loss. For reconstruction tasks (autoencoder, VAE), return the same tensor twice: `return img, img`.

Then update **one line** in `src/train.py` inside `train()`:

```python
# Before (around line 61):
dataset = TemplateDataset(
    input_size=m_cfg["input_size"],
    num_classes=m_cfg["output_size"],
)

# After:
dataset = YourDataset()
```

Also update the import at the top of `src/train.py` to import `YourDataset` instead of `TemplateDataset`.

Do not touch `split_dataset`, `get_dataloaders`, or anything below the dataset line in `train()`.

---

### 2. Add a standard model (classifier, regressor)

A standard model uses `criterion(outputs, targets)` as its loss (`nn.CrossEntropyLoss` by default).

**Step A** — Add the class to `src/models.py`:

```python
class YourModel(nn.Module):
    def __init__(self, param1, param2, ...):
        super().__init__()
        # build layers

    def forward(self, x: Tensor) -> Tensor:
        # return predictions — shape must be compatible with criterion
```

The `__init__` parameter names must exactly match the keys in the `[[model]]` config block (minus `type`).

**Step B** — Register in `src/train.py`:

```python
from src.models import SimpleCNN, TemplateModel, YourModel

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": TemplateModel,
    "cnn": SimpleCNN,
    "your_type": YourModel,   # ← add this line
}
```

**Step C** — Add a `[[model]]` block to `src/configuration.toml`:

```toml
[[model]]
type = "your_type"
param1 = value1
param2 = value2
```

---

### 3. Add a model with custom loss (VAE, autoencoder, contrastive, etc.)

Follow the same three steps as above, but also implement a `loss` method on the model class:

```python
def loss(self, inputs: Tensor, outputs: Any, targets: Tensor) -> Tensor:
    """
    inputs  — the raw batch (same as what was passed to forward)
    outputs — whatever forward() returned (may be a tuple)
    targets — second element of dataset's __getitem__
    Returns a scalar tensor; .backward() is called on it.
    """
```

When `loss` is defined on the model, the training and evaluation loops call it instead of `criterion`. No other files need to change.

---

## What not to change

- `split_dataset` and `get_dataloaders` signatures in `src/datasets.py`
- `train()` and `sweep()` signatures in `src/train.py`
- `_make_callbacks` in `src/main.py`
- CLI command names: `train`, `sweep`, `viewer`, `inference`
- Keys in `metrics.jsonl`: `epoch`, `train_loss`, `val_loss`, `timestamp`

---

## Worked example: VAE on MNIST

**Prompt:** "Make the model a Variational AutoEncoder that operates on MNIST."

### Step 1 — Generate and install

```bash
cookiecutter https://github.com/your-org/mltemplate
# project_name: MNIST VAE
# project_slug: mnist_vae
cd mnist_vae
uv pip install -e .
```

### Step 2 — `pyproject.toml`

Add `"torchvision"` to `dependencies`:

```toml
dependencies = [
    "torch",
    "torchvision",
    "tqdm",
    "wandb",
    "typer[all]",
    "rich",
    "matplotlib",
]
```

Re-install: `uv pip install -e .`

### Step 3 — `src/configuration.toml`

Replace the `[[model]]` block (remove all existing `[[model]]` entries and add):

```toml
[[model]]
type = "vae"
input_size = 784
latent_dim = 20
```

### Step 4 — `src/datasets.py`

Replace the `TemplateDataset` class with:

```python
class MNISTDataset(Dataset):
    def __init__(self, root: str = ".", train: bool = True):
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten()),
        ])
        self.data = datasets.MNIST(root, train=train, download=True, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.data[idx]
        return img, img  # reconstruction target = input
```

Keep `split_dataset` and `get_dataloaders` unchanged.

### Step 5 — `src/train.py`

Two changes:

**Change 1** — update the import at the top:

```python
# Before:
from src.datasets import TemplateDataset, get_dataloaders, split_dataset

# After:
from src.datasets import MNISTDataset, get_dataloaders, split_dataset
```

**Change 2** — update the dataset line inside `train()`:

```python
# Before:
dataset = TemplateDataset(
    input_size=m_cfg["input_size"],
    num_classes=m_cfg["output_size"],
)

# After:
dataset = MNISTDataset()
```

**Change 3** — add `VAE` to `_MODEL_REGISTRY`:

```python
from src.models import SimpleCNN, TemplateModel, VAE

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": TemplateModel,
    "cnn": SimpleCNN,
    "vae": VAE,
}
```

### Step 6 — `src/models.py`

Add the `VAE` class at the end of the file:

```python
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational AutoEncoder for flat inputs.

    Config keys: input_size (int), latent_dim (int).
    Implements loss() so the training loop uses ELBO instead of CrossEntropyLoss.
    """

    def __init__(self, input_size: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 400), nn.ReLU())
        self.mu_head = nn.Linear(400, latent_dim)
        self.logvar_head = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ReLU(),
            nn.Linear(400, input_size), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar

    def loss(
        self,
        inputs: torch.Tensor,
        outputs: tuple,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        recon, mu, logvar = outputs
        recon_loss = F.binary_cross_entropy(recon, targets, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_loss) / inputs.size(0)
```

### Step 7 — Train and view

```bash
mnist_vae train
mnist_vae viewer logs/mnist_vae_log/<timestamp>/
```

Expected: ELBO loss decreases over epochs. The training curve is saved to `training_curve.png`.

### Inference note

For a VAE, `forward()` returns `(reconstruction, mu, logvar)`. The `inference` command prints the raw model output, so the reconstruction is the first element of the printed list. To decode a digit:

```bash
mnist_vae inference logs/mnist_vae_log/<timestamp>/checkpoint_epoch_10.pth "$INPUT"
# Output: [[0.02, 0.91, 0.03, ...], [...], [...]]
# First element is the reconstruction (784 floats, values in [0, 1])
```
