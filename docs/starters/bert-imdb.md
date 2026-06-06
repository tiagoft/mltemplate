# Starter: `bert_imdb` — BERT Classifier for IMDB

Generate with:

```bash
cookiecutter https://github.com/tiagoft/mltemplate
# starting_point [template]: bert_imdb
```

Extra dependencies added automatically: `transformers`, `datasets` (HuggingFace).

---

## What you get

A fine-tuned BERT sentiment classifier for the IMDB movie review dataset. Only a small classification head is trained from scratch — BERT weights are initialized from `bert-base-uncased` and fine-tuned end-to-end via a low learning rate.

```bash
cd my_project
uv pip install -e .
my_project train
```

> **GPU recommended.** Fine-tuning BERT on CPU is very slow. With a GPU, training converges in 2–3 epochs (~30 min).

---

## Architecture — `BERTClassifier`

Input: `{"input_ids": LongTensor(B, max_seq_len), "attention_mask": LongTensor(B, max_seq_len)}`.

```
BertModel.from_pretrained(bert_model_name)   # frozen or fine-tuned
CLS token → bert_out.last_hidden_state[:, 0, :]   # (B, hidden_size)
Dropout(dropout)
Linear(hidden_size, num_classes)
```

Only the classification head is added from scratch. All BERT parameters receive gradients and are fine-tuned end-to-end.

---

## Configuration

```toml
[training]
batch_size = 16
num_epochs = 3
learning_rate = 2e-5          # standard BERT fine-tuning range
checkpoint_every_n_epochs = 1
log_directory = "logs"

[[model]]
type = "bert"
bert_model_name = "bert-base-uncased"
num_classes = 2
dropout = 0.1

[[dataset]]
name = "imdb"
type = "bert_imdb"
bert_model_name = "bert-base-uncased"
max_seq_len = 128             # BERT input length (reviews truncated here)
```

The same `bert_model_name` must appear in both `[[model]]` and `[[dataset]]` so the tokenizer and model architecture match.

---

## Dataset

Loaded via `datasets.load_dataset("imdb")`. Texts are tokenized with `AutoTokenizer.from_pretrained(bert_model_name)` and padded/truncated to `max_seq_len`. Items returned:

```python
({"input_ids": LongTensor(max_seq_len), "attention_mask": LongTensor(max_seq_len)}, int_label)
```

The 25 000-sample training split is divided 22 500 / 2 500 (train / val). The standard test set is held out.

---

## Extending

**Freeze BERT, train only the head** (much faster, slightly lower accuracy):

In `src/models.py`, add inside `__init__`:

```python
for param in self.bert.parameters():
    param.requires_grad = False
```

**Try a different pretrained model:**

```toml
[[model]]
type = "bert"
bert_model_name = "distilbert-base-uncased"   # 40% smaller, 60% faster

[[dataset]]
name = "imdb"
type = "bert_imdb"
bert_model_name = "distilbert-base-uncased"
max_seq_len = 128
```

Then update `src/models.py` to import `DistilBertModel` instead of `BertModel` (the forward signature and CLS extraction are identical).

**Compare sequence lengths:**

```toml
[[dataset]]
name = "short"
type = "bert_imdb"
bert_model_name = "bert-base-uncased"
max_seq_len = 64

[[dataset]]
name = "standard"
type = "bert_imdb"
bert_model_name = "bert-base-uncased"
max_seq_len = 128
```

```bash
my_project sweep
```

**Switch to RoBERTa:**

RoBERTa uses byte-pair encoding and was not trained with the Next Sentence Prediction objective, so its `pooler_output` is not meaningful for classification. Always use `last_hidden_state[:, 0, :]` (the `<s>` token, RoBERTa's equivalent of CLS). Our code already does this, so only the model class and name change.

In `src/models.py`, replace the import and instantiation inside `__init__`:

```python
from transformers import RobertaModel          # was: BertModel
self.bert = RobertaModel.from_pretrained(bert_model_name)
```

`forward()` is unchanged — `last_hidden_state[:, 0, :]` extracts the same position.

In `configuration.toml`, update both sections to the same model name:

```toml
[[model]]
type = "bert"
bert_model_name = "roberta-base"   # or "roberta-large"

[[dataset]]
name = "imdb"
type = "bert_imdb"
bert_model_name = "roberta-base"
max_seq_len = 128
```

`AutoTokenizer` handles RoBERTa's tokenizer automatically — no dataset changes needed.

**Switch to Sentence-BERT (sBERT):**

sBERT models are trained to produce semantically meaningful sentence embeddings via mean pooling over all token representations (weighted by the attention mask), rather than relying on the CLS token. The pretrained weights are available on HuggingFace under the `sentence-transformers` namespace.

In `src/models.py`, replace the `forward` method:

```python
def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    # Attention-mask-weighted mean pool
    token_emb = out.last_hidden_state                                   # (B, L, hidden)
    mask = attention_mask.unsqueeze(-1).float()                         # (B, L, 1)
    pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # (B, hidden)
    return self.classifier(self.dropout(pooled))
```

In `configuration.toml`, use an sBERT model name. Note that these models typically have a smaller hidden size than `bert-base` (e.g. `all-MiniLM-L6-v2` has `hidden_size=384`), so `d_model` should be updated accordingly if you carry the hidden size anywhere:

```toml
[[model]]
type = "bert"
bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
num_classes = 2
dropout = 0.1

[[dataset]]
name = "imdb"
type = "bert_imdb"
bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
max_seq_len = 128
```
