import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model over a dataloader. Returns {"loss": float}."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if hasattr(model, "loss"):
                loss = model.loss(inputs, outputs, targets)
            else:
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
    return {"loss": total_loss / n_batches if n_batches > 0 else float("nan")}


def evaluate_all(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate on val and test sets. Keys prefixed with val_ and test_."""
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    return {f"val_{k}": v for k, v in val_metrics.items()} | {
        f"test_{k}": v for k, v in test_metrics.items()
    }
