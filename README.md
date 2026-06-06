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
cookiecutter https://github.com/tiagoft/mltemplate
```

You will be prompted for a few values:

```text
project_name [My ML Project]: My Classifier
project_slug [my_classifier]:
author_name [Your Name]: Ada Lovelace
author_email [you@example.com]: ada@example.com
python_version [3.13]:
starting_point [template]: cnn_mnist
```

**Install and run:**

```bash
cd my_classifier
uv pip install -e .
my_classifier train
```

---

## Starting points

Choose a fully-implemented architecture at generation time:

| Option | Task | Extra dep | Doc |
| --- | --- | --- | --- |
| `template` | Random tensors (blank scaffold) | — | [below](#template-scaffold) |
| `cnn_mnist` | Image classification — MNIST | `torchvision` | [cnn-mnist.md](docs/starters/cnn-mnist.md) |
| `transformer_imdb` | Sentiment — IMDB, from scratch | `datasets` | [transformer-imdb.md](docs/starters/transformer-imdb.md) |
| `bert_imdb` | Sentiment — IMDB, BERT fine-tune | `transformers`, `datasets` | [bert-imdb.md](docs/starters/bert-imdb.md) |
| `bert_chunks_imdb` | Long-doc — IMDB, BERT + Transformer | `transformers`, `datasets` | [bert-chunks-imdb.md](docs/starters/bert-chunks-imdb.md) |
| `audio_cnn` | Spoken digit recognition | `torchaudio` | [audio-cnn.md](docs/starters/audio-cnn.md) |

Each starter ships with working `models.py`, `datasets.py`, and `configuration.toml`. The CLI, training loop, checkpointing, and sweep logic are identical across all starters.

---

## Template scaffold

The `template` starting point ships with a `TemplateDataset` that produces random tensors — useful for verifying the full pipeline before you swap in real data, or as a blank canvas for a custom task.

Generated project layout:

```text
my_classifier/
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

**Train:**

```bash
my_classifier train
```

**Plot the training curve:**

```bash
my_classifier viewer logs/my_classifier_log/20240601_143022/
```

**List all logged runs:**

```bash
my_classifier viewer --list
```

```text
           Runs in logs/my_classifier_log
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Run                 ┃ Dataset ┃ Epochs ┃ Final val_loss ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 20240601_143022     │ small   │     10 │         2.3021 │
└─────────────────────┴─────────┴────────┴────────────────┘
```

---

## Multiple datasets

`configuration.toml` supports multiple `[[dataset]]` blocks. `train` uses only the first. `sweep` evaluates all dataset × model combinations:

```toml
[[dataset]]
name = "small"
type = "template"
size = 500

[[dataset]]
name = "large"
type = "template"
size = 2000

[[model]]
type = "mlp"
hidden_sizes = [128]
output_size = 10

[[model]]
type = "mlp"
hidden_sizes = [256, 128]
output_size = 10
```

```bash
my_classifier sweep
```

```text
         Sweep results: sweep_20240601_150312
┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Variant ┃ Dataset ┃ Model          ┃ Final val_loss ┃
┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ v00     │ small   │ mlp [128]      │         0.8312 │
│ v01     │ small   │ mlp [256, 128] │         0.7904 │
│ v02     │ large   │ mlp [128]      │         0.5221 │
│ v03     │ large   │ mlp [256, 128] │         0.4873 │
└─────────┴─────────┴────────────────┴────────────────┘
```

Each variant saves its own checkpoints and `metrics.jsonl` under `sweep_<timestamp>/v0N/`.

---

## Customize with an LLM

Every generated project includes `AGENTS.md` — a machine-readable guide that describes the exact files, interfaces, and contracts an LLM needs to extend the project correctly.

**With Claude Code** (auto-loads `AGENTS.md` from the project root):

```bash
cd my_classifier
claude
```

Then describe what you want in plain language:

```text
Make the model a Variational AutoEncoder.
```

**With any other LLM** (ChatGPT, Gemini, Copilot Chat, etc.) — paste `AGENTS.md` first:

```text
<paste the contents of AGENTS.md here>

Now add a ResNet-18 backbone using torchvision.models.resnet18.
```

The LLM will make the minimum set of changes needed — typically 3–5 targeted edits across `datasets.py`, `models.py`, and `configuration.toml`. Everything else (training loop, CLI, checkpointing, sweep logic) stays untouched.

---

## Enable Weights & Biases

W&B logging is off by default. To enable it, uncomment one line in `src/configuration.toml`:

```toml
[training]
...
wandb_project = "my_classifier_experiments"  # ← uncomment this line
```

Log in (`wandb login`), then run training as usual. The `sweep` command automatically groups all variants under a shared run group, which appears as a single experiment in the W&B UI where you can compare learning curves and final metrics side by side.
