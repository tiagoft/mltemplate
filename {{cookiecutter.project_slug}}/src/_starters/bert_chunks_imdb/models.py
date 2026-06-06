import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ChunkBERTClassifier(nn.Module):
    """BERT-based long-document classifier using chunk-level Transformer aggregation.

    Each document is split into overlapping chunks. BERT encodes each chunk independently
    to produce a CLS embedding. A Transformer-from-scratch then attends over chunk
    embeddings to produce the final document representation.

    Expects dict input:
        input_ids:      LongTensor(B, max_chunks, chunk_size)
        attention_mask: LongTensor(B, max_chunks, chunk_size)
    """

    def __init__(
        self,
        bert_model_name: str,
        max_chunks: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import BertModel  # noqa: PLC0415

        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # Project BERT hidden size → d_model if they differ
        self.proj = (
            nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        )
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_chunks)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.chunk_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, C, L = input_ids.shape

        # Encode all chunks in parallel by merging batch and chunk dims
        ids_flat = input_ids.view(B * C, L)
        mask_flat = attention_mask.view(B * C, L)
        bert_out = self.bert(input_ids=ids_flat, attention_mask=mask_flat)
        cls_emb = bert_out.last_hidden_state[:, 0, :]   # (B*C, hidden_size)

        # Reshape → (B, C, d_model) and add positional encoding
        chunk_emb = self.proj(cls_emb).view(B, C, -1)
        chunk_emb = self.pos_enc(chunk_emb)

        # Attend over chunks, mean-pool, classify
        aggregated = self.chunk_encoder(chunk_emb).mean(dim=1)  # (B, d_model)
        return self.classifier(self.dropout(aggregated))


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "bert_chunks": ChunkBERTClassifier,
}
