import torch
import torch.nn as nn


class BERTClassifier(nn.Module):
    """BERT-based text classifier with a custom classification head.

    Loads BERT weights from HuggingFace; the classification head is trained from scratch.
    Expects dict input: {"input_ids": LongTensor(B, seq), "attention_mask": LongTensor(B, seq)}.
    """

    def __init__(self, bert_model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        from transformers import BertModel  # noqa: PLC0415

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "bert": BERTClassifier,
}
