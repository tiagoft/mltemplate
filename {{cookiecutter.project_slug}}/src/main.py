import json
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import torch
import typer
from rich.console import Console
from rich.table import Table

from src.models import TemplateModel
from src.train import train as train_model

app = typer.Typer(help="{{ cookiecutter.project_name }} CLI")
console = Console()

_DEFAULT_CONFIG = Path("src/configuration.toml")
_LOG_SUBDIR = "{{ cookiecutter.project_slug }}_log"


def _load_config(config_file: Path) -> dict:
    with open(config_file, "rb") as f:
        return tomllib.load(f)


@app.command()
def train(
    config_file: Annotated[Path, typer.Option(help="Path to configuration.toml")] = _DEFAULT_CONFIG,
) -> None:
    """Train the model and save checkpoints + metrics to a timestamped run directory."""
    config = _load_config(config_file)
    t_cfg = config["training"]

    run_dir = (
        Path(t_cfg["log_directory"])
        / _LOG_SUBDIR
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Run directory: [bold]{run_dir}[/bold]")

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_every = t_cfg["checkpoint_every_n_epochs"]

    def metrics_callback(epoch: int, train_loss: float, val_metrics: dict, model) -> None:
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "timestamp": datetime.now().isoformat(),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def checkpoint_callback(epoch: int, train_loss: float, val_metrics: dict, model) -> None:
        if epoch % checkpoint_every == 0:
            path = run_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), path)
            console.print(f"  Saved checkpoint: {path.name}")

    train_model(config, callbacks=[metrics_callback, checkpoint_callback])
    console.print("[green]Training complete.[/green]")


@app.command()
def viewer(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing metrics.jsonl")] = None,
    list_runs: Annotated[bool, typer.Option("--list", help="List all logged runs instead of plotting")] = False,
    config_file: Annotated[Path, typer.Option(help="Config file (used with --list)")] = _DEFAULT_CONFIG,
) -> None:
    """Plot training curves from a run directory, or list all runs."""
    if list_runs:
        config = _load_config(config_file)
        log_dir = Path(config["training"]["log_directory"]) / _LOG_SUBDIR
        if not log_dir.exists():
            console.print(f"[yellow]No runs found in {log_dir}[/yellow]")
            raise typer.Exit()

        table = Table(title=f"Runs in {log_dir}")
        table.add_column("Run", style="cyan")
        table.add_column("Epochs", justify="right")
        table.add_column("Final val_loss", justify="right")

        for metrics_file in sorted(log_dir.glob("*/metrics.jsonl")):
            run_name = metrics_file.parent.name
            lines = metrics_file.read_text().strip().splitlines()
            if not lines:
                table.add_row(run_name, "0", "—")
                continue
            last = json.loads(lines[-1])
            val_loss = last.get("val_loss", "—")
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
            table.add_row(run_name, str(len(lines)), val_loss_str)

        console.print(table)
        return

    if run_dir is None:
        console.print("[red]Provide a run directory or use --list.[/red]")
        raise typer.Exit(1)

    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        console.print(f"[red]{metrics_file} not found.[/red]")
        raise typer.Exit(1)

    records = [json.loads(line) for line in metrics_file.read_text().strip().splitlines()]
    epochs = [r["epoch"] for r in records]
    train_losses = [r["train_loss"] for r in records]
    val_losses = [r.get("val_loss") for r in records]

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train loss")
    if any(v is not None for v in val_losses):
        ax.plot(epochs, val_losses, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve")
    ax.legend()

    out_path = run_dir / "training_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"Saved: [bold]{out_path}[/bold]")


@app.command()
def inference(
    checkpoint: Annotated[Path, typer.Argument(help="Path to a .pth checkpoint file")],
    input_data: Annotated[str, typer.Argument(help="Comma-separated float values, e.g. '1.0,2.0,...'")],
    config_file: Annotated[Path, typer.Option(help="Path to configuration.toml")] = _DEFAULT_CONFIG,
) -> None:
    """Run inference on a single input using a trained checkpoint."""
    config = _load_config(config_file)
    model = TemplateModel(**config["model"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    # Replace this parsing with your actual input preprocessing.
    values = [float(x.strip()) for x in input_data.split(",")]
    x = torch.tensor(values).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    console.print(output.squeeze(0).tolist())
