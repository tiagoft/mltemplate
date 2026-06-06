import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model over a dataloader. Returns loss, macro-F1, and accuracy."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], dict):
                inputs = {k: v.to(device) for k, v in batch[0].items()}
            else:
                inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            if hasattr(model, "loss"):
                loss = model.loss(inputs, outputs, targets)
            else:
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(outputs.argmax(dim=-1).cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")

    preds = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    accuracy = (preds == targets_cat).float().mean().item()

    # Macro-averaged F1 (no external deps)
    f1_per_class = []
    for c in targets_cat.unique():
        tp = ((preds == c) & (targets_cat == c)).sum().item()
        fp = ((preds == c) & (targets_cat != c)).sum().item()
        fn = ((preds != c) & (targets_cat == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1_per_class.append(2 * p * r / (p + r) if p + r else 0.0)
    macro_f1 = sum(f1_per_class) / len(f1_per_class) if f1_per_class else float("nan")

    return {"loss": avg_loss, "f1": macro_f1, "accuracy": accuracy}


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
