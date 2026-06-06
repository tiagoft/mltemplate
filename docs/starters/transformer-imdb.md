# Starter: `transformer_imdb` — Transformer Classifier for IMDB

Generate with:

```bash
cookiecutter https://github.com/tiagoft/mltemplate
# starting_point [template]: transformer_imdb
```

Extra dependency added automatically: `datasets` (HuggingFace).

---

## What you get

A Transformer-from-scratch sentiment classifier trained on the IMDB movie review dataset (25 000 training samples, binary: positive / negative). The dataset is downloaded automatically via HuggingFace `datasets`.

```bash
cd my_project
uv pip install -e .
my_project train
```

---

## Architecture — `TransformerClassifier`

Input: `{"input_ids": LongTensor(B, max_seq_len)}`.

```
Embedding(vocab_size, d_model)
SinusoidalPositionalEncoding(d_model, max_seq_len)   # fixed, not learnable
TransformerEncoder(
    TransformerEncoderLayer(d_model, nhead, batch_first=True),
    num_layers=num_layers
)
mean-pool over sequence dimension   # (B, max_seq_len, d_model) → (B, d_model)
Linear(d_model, num_classes)
```

Positional encoding uses fixed sinusoids registered as a buffer (not a parameter), so it works without any gradient updates and handles any sequence length up to `max_seq_len`.

---

## Configuration

```toml
[training]
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
checkpoint_every_n_epochs = 2
log_directory = "logs"

[[model]]
type = "transformer"
vocab_size = 20000      # top-N word-level tokens from training set
d_model = 128           # embedding / hidden dimension
nhead = 4               # attention heads (must divide d_model)
num_layers = 2          # stacked encoder layers
num_classes = 2         # positive / negative
dropout = 0.1

[[dataset]]
name = "imdb"
type = "imdb"
max_seq_len = 256       # reviews truncated/padded to this length
vocab_size = 20000
```

---

## Dataset

Loaded via `datasets.load_dataset("imdb")`. A word-level vocabulary is built from the training texts using the top `vocab_size` tokens (`<pad>=0`, `<unk>=1`). Reviews are tokenized by whitespace, encoded to `max_seq_len` (truncated or zero-padded), and returned as `({"input_ids": LongTensor(max_seq_len)}, int_label)`.

The 25 000-sample training split is divided 22 500 / 2 500 (train / val). The standard test set is held out.

---

## Extending

**Increase capacity:**

```toml
[[model]]
type = "transformer"
vocab_size = 30000
d_model = 256
nhead = 8
num_layers = 4
dropout = 0.2
```

**Compare sequence lengths:**

```toml
[[dataset]]
name = "short"
type = "imdb"
max_seq_len = 128
vocab_size = 20000

[[dataset]]
name = "long"
type = "imdb"
max_seq_len = 512
vocab_size = 20000
```

```bash
my_project sweep
```

**Switch to subword tokenization:** replace the word-level vocab builder in `src/datasets.py` with a HuggingFace `AutoTokenizer` — the model architecture requires no changes, only the vocabulary size and tokenization logic change.
