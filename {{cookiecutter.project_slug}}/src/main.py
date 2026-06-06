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

from src.train import _MODEL_REGISTRY, train as train_model

app = typer.Typer(help="{{ cookiecutter.project_name }} CLI")
console = Console()

_DEFAULT_CONFIG = Path("src/configuration.toml")
_LOG_SUBDIR = "{{ cookiecutter.project_slug }}_log"


def _load_config(config_file: Path) -> dict:
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def _make_callbacks(run_dir: Path, checkpoint_every: int, dataset_name: str = ""):
    metrics_path = run_dir / "metrics.jsonl"

    def metrics_callback(epoch: int, train_loss: float, val_metrics: dict, model) -> None:
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "dataset": dataset_name,
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

    return [metrics_callback, checkpoint_callback]


@app.command()
def train(
    config_file: Annotated[Path, typer.Option(help="Path to configuration.toml")] = _DEFAULT_CONFIG,
) -> None:
    """Train the model and save checkpoints + metrics to a timestamped run directory."""
    config = _load_config(config_file)
    config = {**config, "model": config["model"][0], "dataset": config["dataset"][0]}
    t_cfg = config["training"]

    run_dir = (
        Path(t_cfg["log_directory"])
        / _LOG_SUBDIR
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Run directory: [bold]{run_dir}[/bold]")

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    dataset_name = config["dataset"].get("name", "")
    train_model(config, callbacks=_make_callbacks(run_dir, t_cfg["checkpoint_every_n_epochs"], dataset_name))
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
        table.add_column("Dataset", style="yellow")
        table.add_column("Epochs", justify="right")
        table.add_column("Final val_loss", justify="right")

        for metrics_file in sorted(log_dir.rglob("metrics.jsonl")):
            run_name = str(metrics_file.parent.relative_to(log_dir))
            lines = metrics_file.read_text().strip().splitlines()

            dataset_name = "—"
            config_json = metrics_file.parent / "config.json"
            if config_json.exists():
                run_cfg = json.loads(config_json.read_text())
                dataset_name = run_cfg.get("dataset", {}).get("name", "—")

            if not lines:
                table.add_row(run_name, dataset_name, "0", "—")
                continue
            last = json.loads(lines[-1])
            val_loss = last.get("val_loss", "—")
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
            table.add_row(run_name, dataset_name, str(len(lines)), val_loss_str)

        console.print(table)
        return

    if run_dir is None:
        console.print("[red]Provide a run directory or use --list.[/red]")
        raise typer.Exit(1)

    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        console.print(f"[red]{metrics_file} not found.[/red]")
        raise typer.Exit(1)

    dataset_name = ""
    config_json = run_dir / "config.json"
    if config_json.exists():
        run_cfg = json.loads(config_json.read_text())
        dataset_name = run_cfg.get("dataset", {}).get("name", "")

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
    title = "Training curve" + (f" — {dataset_name}" if dataset_name else "")
    ax.set_title(title)
    ax.legend()

    out_path = run_dir / "training_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"Saved: [bold]{out_path}[/bold]")


@app.command()
def sweep(
    config_file: Annotated[Path, typer.Option(help="Path to configuration.toml")] = _DEFAULT_CONFIG,
) -> None:
    """Train all [[model]] × [[dataset]] combinations defined in configuration.toml."""
    config = _load_config(config_file)
    model_list = config["model"]
    dataset_list = config["dataset"]
    combos = [(ds, m) for ds in dataset_list for m in model_list]

    if len(combos) < 2:
        console.print(
            "[yellow]Only one combination found — use [bold]train[/bold] for a single run.[/yellow]"
        )
        console.print("Add more [[model]] or [[dataset]] blocks to configuration.toml to define sweep variants.")
        raise typer.Exit(1)

    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_group = f"sweep_{sweep_ts}"
    sweep_dir = Path(config["training"]["log_directory"]) / _LOG_SUBDIR / sweep_group
    console.print(f"Sweep directory: [bold]{sweep_dir}[/bold]  ({len(combos)} variants)\n")

    results = []
    for i, (ds_cfg, m_cfg) in enumerate(combos):
        run_cfg = {**config, "model": m_cfg, "dataset": ds_cfg, "wandb_group": sweep_group}

        run_dir = sweep_dir / f"v{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

        label = f"{ds_cfg.get('name', '?')} / {m_cfg.get('type', 'mlp')}"
        console.print(f"[bold cyan]Variant v{i:02d}[/bold cyan]: {label}")

        dataset_name = ds_cfg.get("name", "")
        model, history = train_model(
            run_cfg,
            callbacks=_make_callbacks(run_dir, run_cfg["training"]["checkpoint_every_n_epochs"], dataset_name),
        )
        results.append((label, ds_cfg.get("name", ""), m_cfg.get("type", "mlp"), history))
        console.print()

    table = Table(title=f"Sweep results: {sweep_group}")
    table.add_column("Variant", style="cyan")
    table.add_column("Dataset", style="yellow")
    table.add_column("Model", style="magenta")
    table.add_column("Final val_loss", justify="right")
    for i, (label, ds_name, model_type, history) in enumerate(results):
        val_loss = history[-1].get("val_loss", float("nan"))
        table.add_row(f"v{i:02d}", ds_name, model_type, f"{val_loss:.4f}")
    console.print(table)
    console.print(f"\nView curves: [bold]{sweep_group}/v<N>/[/bold] or run [bold]viewer --list[/bold]")


@app.command()
def inference(
    checkpoint: Annotated[Path, typer.Argument(help="Path to a .pth checkpoint file")],
    input_data: Annotated[str, typer.Argument(help="Comma-separated float values, e.g. '1.0,2.0,...'")],
    config_file: Annotated[Path, typer.Option(help="Path to configuration.toml")] = _DEFAULT_CONFIG,
) -> None:
    """Run inference on a single input using a trained checkpoint."""
    config = _load_config(config_file)
    m_cfg = config["model"][0]
    model_cls = _MODEL_REGISTRY[m_cfg.get("type", "mlp")]
    model = model_cls(**{k: v for k, v in m_cfg.items() if k != "type"})
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    # Replace this parsing with your actual input preprocessing.
    values = [float(x.strip()) for x in input_data.split(",")]
    x = torch.tensor(values).unsqueeze(0)

    with torch.no_grad():
        output = model(x)

    console.print(output.squeeze(0).tolist())
