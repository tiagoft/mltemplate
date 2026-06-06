# Agent Guide — {{ cookiecutter.project_name }}

This file is written for LLMs and code-generating tools. It describes the project structure, the exact interfaces that must be satisfied when extending it, and a complete worked example.

---

## Project layout

| File | Role |
| --- | --- |
| `src/configuration.toml` | All hyperparameters. `[[model]]` blocks define models; `[[dataset]]` blocks define datasets. |
| `src/datasets.py` | Dataset classes + `_DATASET_REGISTRY` + `get_dataset` + `split_dataset` + `get_dataloaders`. |
| `src/models.py` | Model classes. Add new architectures here. |
| `src/eval.py` | `evaluate(model, dataloader, criterion, device) -> dict`. Runs validation. |
| `src/train.py` | `train(config, callbacks)`, `sweep(model_configs, dataset_configs, base_config)`, `_MODEL_REGISTRY`. |
| `src/main.py` | CLI: `train`, `sweep`, `viewer`, `inference`. Do not modify unless adding a new command. |
| `pyproject.toml` | Dependencies. Add libraries here (e.g. `torchvision`). |

---

## Extension contracts

### 1. Replace the dataset

Add a factory function to `src/datasets.py` and register it in `_DATASET_REGISTRY`. The factory signature is:

```python
def get_your_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train_ds, val_ds, test_ds)."""
```

The `config` argument is the matching `[[dataset]]` block from `configuration.toml`.

Each `Dataset.__getitem__` must return a two-element tuple:

```python
# For plain-tensor models (MLP, CNN):
return input_tensor, label_tensor

# For models requiring multiple named inputs (BERT, Transformer):
return {"input_ids": ids_tensor, "attention_mask": mask_tensor}, label_tensor

# For reconstruction tasks (autoencoder, VAE):
return img_tensor, img_tensor
```

When `__getitem__` returns a dict as the first element, the training and evaluation loops automatically call `model(**inputs)` instead of `model(inputs)`. When `loss` is defined on the model, `inputs` passed to `model.loss(inputs, outputs, targets)` is also the dict (or tensor, respectively).

**Step A** — Add the factory to `src/datasets.py`:

```python
def get_your_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    train_ds = ...
    val_ds = ...
    test_ds = ...
    return train_ds, val_ds, test_ds

_DATASET_REGISTRY["your_type"] = get_your_datasets
```

**Step B** — Add a `[[dataset]]` block to `src/configuration.toml`:

```toml
[[dataset]]
name = "my_dataset"
type = "your_type"
# any extra keys are passed through to get_your_datasets via config
```

Do not touch `split_dataset`, `get_dataloaders`, or any file in `src/train.py` to add a dataset.

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

If the model requires multiple named inputs (e.g. `input_ids` and `attention_mask`), define `forward` with those exact keyword argument names — they are unpacked from the dict returned by `__getitem__`:

```python
def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    ...
```

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
def loss(self, inputs, outputs, targets) -> Tensor:
    """
    inputs  — raw batch input (Tensor or dict of Tensors, same as forward's input)
    outputs — whatever forward() returned (may be a tuple)
    targets — second element of dataset's __getitem__
    Returns a scalar tensor; .backward() is called on it.
    """
```

When `loss` is defined on the model, the training and evaluation loops call it instead of `criterion`. No other files need to change.

---

## What not to change

- `split_dataset` and `get_dataloaders` signatures in `src/datasets.py`
- `get_dataset` dispatcher in `src/datasets.py` (add to `_DATASET_REGISTRY` instead)
- `train()` signature in `src/train.py`
- `_make_callbacks` in `src/main.py`
- CLI command names: `train`, `sweep`, `viewer`, `inference`
- Keys in `metrics.jsonl`: `epoch`, `train_loss`, `val_loss`, `dataset`, `timestamp`

---

## Worked example: VAE on MNIST

**Prompt:** "Make the model a Variational AutoEncoder that operates on MNIST."

### Step 1 — Generate and install

```bash
cookiecutter https://github.com/your-org/mltemplate
# project_name: MNIST VAE
# project_slug: mnist_vae
# starting_point: template
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

Replace the `[[model]]` block and the `[[dataset]]` blocks:

```toml
[[model]]
type = "vae"
input_size = 784
latent_dim = 20

[[dataset]]
name = "mnist"
type = "mnist_flat"
```

### Step 4 — `src/datasets.py`

Add a factory function after the existing code:

```python
def get_mnist_flat_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
    ])

    class _MNISTWrap(Dataset):
        def __init__(self, tv_dataset):
            self.ds = tv_dataset
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, _ = self.ds[idx]
            return img, img  # reconstruction target = input

    train_raw = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_raw  = datasets.MNIST(".", train=False, download=True, transform=transform)
    train_full = _MNISTWrap(train_raw)
    train_ds, val_ds = random_split(train_full, [50000, 10000])
    return train_ds, val_ds, _MNISTWrap(test_raw)

_DATASET_REGISTRY["mnist_flat"] = get_mnist_flat_datasets
```

### Step 5 — `src/train.py`

Add `VAE` to `_MODEL_REGISTRY`:

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

    def loss(self, inputs, outputs, targets) -> torch.Tensor:
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

Expected: ELBO loss decreases over epochs. The training curve is saved to `training_curve.png` with the title "Training curve — mnist".

### Inference note

For a VAE, `forward()` returns `(reconstruction, mu, logvar)`. The `inference` command prints the raw model output, so the reconstruction is the first element of the printed list. To decode a digit:

```bash
mnist_vae inference logs/mnist_vae_log/<timestamp>/checkpoint_epoch_10.pth "$INPUT"
# Output: [[0.02, 0.91, 0.03, ...], [...], [...]]
# First element is the reconstruction (784 floats, values in [0, 1])
```
