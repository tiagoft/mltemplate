# Starter: `bert_chunks_imdb` — BERT + Chunking for Long Documents (IMDB)

Generate with:

```bash
cookiecutter https://github.com/tiagoft/mltemplate
# starting_point [template]: bert_chunks_imdb
```

Extra dependencies added automatically: `transformers`, `datasets` (HuggingFace).

---

## What you get

A classifier for long documents. BERT has a 512-token limit, which truncates many IMDB reviews. This starter splits each review into fixed-size chunks, encodes each chunk independently with BERT to get a CLS embedding, then uses a Transformer-from-scratch to aggregate the chunk-level representations into a document-level prediction.

```bash
cd my_project
uv pip install -e .
my_project train
```

> **GPU required for practical use.** Each forward pass runs BERT once per chunk (default: 4 chunks). batch_size is set to 8 by default to fit in GPU memory.

---

## Architecture — `ChunkBERTClassifier`

Input: `{"input_ids": LongTensor(B, max_chunks, chunk_size), "attention_mask": LongTensor(B, max_chunks, chunk_size)}`.

```text
# 1. Encode all chunks in parallel
reshape (B, num_chunks, chunk_size) → (B * num_chunks, chunk_size)
BertModel → CLS embeddings: (B * num_chunks, hidden_size)
reshape → (B, num_chunks, hidden_size)

# 2. Optional projection (if hidden_size ≠ d_model)
Linear(hidden_size, d_model)   or   Identity()

# 3. Aggregate chunks with a Transformer
SinusoidalPositionalEncoding(d_model, num_chunks)
TransformerEncoder(TransformerEncoderLayer(d_model, nhead), num_layers)
mean-pool over chunk dimension → (B, d_model)

# 4. Classify
Linear(d_model, num_classes)
```

The batch and chunk dimensions are merged before BERT (`B*C, L`) and restored after, so all chunks are processed in one forward pass — no Python loop over chunks.

---

## Configuration

```toml
[training]
batch_size = 8              # small: each sample expands to max_chunks BERT passes
num_epochs = 5
learning_rate = 2e-5
checkpoint_every_n_epochs = 1
log_directory = "logs"

[[model]]
type = "bert_chunks"
bert_model_name = "bert-base-uncased"
max_chunks = 4              # Transformer sequence length (positions for chunk-level PE)
chunk_size = 128            # tokens per chunk (≤ 512)
d_model = 768               # must match bert hidden_size unless projection is used
nhead = 8                   # attention heads for chunk-level Transformer
num_layers = 2
num_classes = 2
dropout = 0.1

[[dataset]]
name = "imdb_chunked"
type = "bert_chunks_imdb"
bert_model_name = "bert-base-uncased"
chunk_size = 128            # tokens per chunk
chunk_overlap = 32          # tokens shared between consecutive chunks
max_chunks = 4              # cap on number of chunks per document
```

`max_chunks` and `chunk_size` must match between `[[model]]` and `[[dataset]]`. The stride between chunk starts is `chunk_size - chunk_overlap`. With `chunk_overlap = 0` the chunks are non-overlapping (original behaviour).

---

## Dataset

Loaded via `datasets.load_dataset("imdb")`. Each text is tokenized without truncation and split into overlapping chunks with stride `chunk_size - chunk_overlap`. Documents shorter than `max_chunks` chunks are zero-padded. Items returned:

```python
(
    {"input_ids": LongTensor(max_chunks, chunk_size),
     "attention_mask": LongTensor(max_chunks, chunk_size)},
    int_label
)
```

PyTorch's default collate handles the nested dict correctly.

---

## Extending

**Cover more of each document** (higher memory cost):

```toml
max_chunks = 8
chunk_size = 128
chunk_overlap = 32
```

**Use DistilBERT** for lower memory / faster training:

In `src/models.py`, replace `BertModel.from_pretrained(...)` with `DistilBertModel.from_pretrained(...)` — the `last_hidden_state` attribute and CLS extraction are identical, so only the import changes.

**Sweep overlap amounts:**

```toml
[[dataset]]
name = "no_overlap"
type = "bert_chunks_imdb"
bert_model_name = "bert-base-uncased"
chunk_size = 128
chunk_overlap = 0
max_chunks = 4

[[dataset]]
name = "overlap_32"
type = "bert_chunks_imdb"
bert_model_name = "bert-base-uncased"
chunk_size = 128
chunk_overlap = 32
max_chunks = 4
```

```bash
my_project sweep
```
