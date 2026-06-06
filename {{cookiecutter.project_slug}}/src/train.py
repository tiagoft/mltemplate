from datetime import datetime
from typing import Callable

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm


class _NoOpRun:
    """Stands in for a wandb run when W&B is not configured."""
    def log(self, *args, **kwargs): pass
    def finish(self): pass

from src.datasets import get_dataset, get_dataloaders
from src.eval import evaluate
from src.models import SimpleCNN, TemplateModel

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": TemplateModel,
    "cnn": SimpleCNN,
}


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(dataloader, desc="Batches", leave=False):
        if isinstance(batch[0], dict):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
        else:
            inputs = batch[0].to(device)
        targets = batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        if hasattr(model, "loss"):
            loss = model.loss(inputs, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else float("nan")


def train(
    config: dict,
    callbacks: list[Callable] | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Instantiate everything from config and run the full training loop.

    Each callback is called as cb(epoch, train_loss, val_metrics, model)
    after every epoch. Returns (trained_model, history).
    """
    if callbacks is None:
        callbacks = []

    t_cfg = config["training"]
    m_cfg = config["model"]
    ds_cfg = config["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = get_dataset(ds_cfg)
    train_loader, val_loader, _ = get_dataloaders(
        train_ds, val_ds, test_ds, t_cfg["batch_size"]
    )

    model_type = m_cfg.get("type", "mlp")
    model_cls = _MODEL_REGISTRY[model_type]
    model_kwargs = {k: v for k, v in m_cfg.items() if k != "type"}
    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    wandb_project = t_cfg.get("wandb_project") or config.get("wandb_project")
    if wandb_project:
        run = wandb.init(
            project=wandb_project,
            config=config,
            group=config.get("wandb_group"),
        )
    else:
        run = _NoOpRun()

    history: list[dict] = []
    try:
        for epoch in tqdm(range(1, t_cfg["num_epochs"] + 1), desc="Epochs"):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            run.log(
                {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )

            record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(record)

            for cb in callbacks:
                cb(epoch, train_loss, val_metrics, model)
    finally:
        run.finish()

    return model, history


def sweep(
    model_configs: list[dict],
    dataset_configs: list[dict],
    base_config: dict,
) -> list[tuple[dict, nn.Module, list[dict]]]:
    """Train one model per (model_config × dataset_config), grouped as a single W&B sweep."""
    sweep_group = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []
    for ds_cfg in dataset_configs:
        for m_cfg in model_configs:
            config = {**base_config, "model": m_cfg, "dataset": ds_cfg, "wandb_group": sweep_group}
            model, history = train(config)
            results.append((config, model, history))
    return results
