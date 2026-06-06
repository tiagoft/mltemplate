import torch
import torch.nn as nn


class TemplateModel(nn.Module):
    """MLP with configurable hidden layers. Replace or extend as needed.

    Signature matches configuration.toml [model] section directly:
        TemplateModel(**config["model"])
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(dropout)]
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Convolutional classifier for 2D inputs (B, 1, H, W). Replace or extend as needed.

    Signature matches a configuration.toml [model] section with type = "cnn":
        SimpleCNN(**{k: v for k, v in config["model"].items() if k != "type"})
    """

    def __init__(
        self,
        channels: list[int],
        output_size: int,
        dropout: float = 0.0,
    ):
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
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        x = self.pool(self.conv(x))
        return self.classifier(x.flatten(1))
