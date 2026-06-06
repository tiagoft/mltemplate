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

**Switch to RoBERTa:**

RoBERTa's `pooler_output` is not meaningful for classification (it was not trained with NSP). Our code already uses `last_hidden_state[:, 0, :]` — the `<s>` token at position 0, RoBERTa's equivalent of CLS — so only the model class and name need to change.

In `src/models.py`, replace the import and instantiation inside `__init__`:

```python
from transformers import RobertaModel          # was: BertModel
self.bert = RobertaModel.from_pretrained(bert_model_name)
```

`forward()` is unchanged — `last_hidden_state[:, 0, :]` extracts the same position. `AutoTokenizer` handles RoBERTa's tokenizer automatically, so no dataset changes are needed.

In `configuration.toml`, update the model name in both sections:

```toml
[[model]]
type = "bert_chunks"
bert_model_name = "roberta-base"
max_chunks = 4
d_model = 768

[[dataset]]
name = "imdb_chunks"
type = "bert_chunks_imdb"
bert_model_name = "roberta-base"
chunk_size = 128
chunk_overlap = 32
max_chunks = 4
```

**Switch to Sentence-BERT (sBERT):**

sBERT models produce better chunk-level representations by using attention-mask-weighted mean pooling over all token embeddings instead of relying solely on the CLS token. This is especially beneficial here, since each chunk is a short passage and mean pooling captures the full chunk content.

In `src/models.py`, replace the CLS extraction line inside `forward()`:

```python
def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, C, L = input_ids.shape
    ids_flat = input_ids.view(B * C, L)
    mask_flat = attention_mask.view(B * C, L)
    bert_out = self.bert(input_ids=ids_flat, attention_mask=mask_flat)

    # Attention-mask-weighted mean pool per chunk (sBERT strategy)
    token_emb = bert_out.last_hidden_state                              # (B*C, L, hidden)
    mask = mask_flat.unsqueeze(-1).float()                              # (B*C, L, 1)
    chunk_emb_flat = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B*C, hidden)

    chunk_emb = self.proj(chunk_emb_flat).view(B, C, -1)
    chunk_emb = self.pos_enc(chunk_emb)
    aggregated = self.chunk_encoder(chunk_emb).mean(dim=1)
    return self.classifier(self.dropout(aggregated))
```

In `configuration.toml`, use an sBERT model name. Note that `all-MiniLM-L6-v2` has `hidden_size=384`, which is smaller than `bert-base` (768). Set `d_model` accordingly (or leave it at 384 and let the `Identity` projection pass through):

```toml
[[model]]
type = "bert_chunks"
bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
max_chunks = 4
d_model = 384      # matches all-MiniLM-L6-v2 hidden_size; no projection needed
nhead = 8
num_layers = 2
num_classes = 2

[[dataset]]
name = "imdb_chunks"
type = "bert_chunks_imdb"
bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 128
chunk_overlap = 32
max_chunks = 4
```
