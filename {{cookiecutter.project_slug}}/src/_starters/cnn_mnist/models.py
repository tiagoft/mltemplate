import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """CNN classifier for 28×28 greyscale images (MNIST). Input shape: (B, 1, 28, 28)."""

    def __init__(self, channels: list[int], output_size: int, dropout: float = 0.0):
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mnist_cnn": MNISTClassifier,
}
