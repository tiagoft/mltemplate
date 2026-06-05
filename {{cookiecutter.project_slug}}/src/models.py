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
