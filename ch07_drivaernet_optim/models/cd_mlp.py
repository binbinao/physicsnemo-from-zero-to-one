"""
Chapter 7: MLP Model for Drag Coefficient Prediction
======================================================
Simple multi-layer perceptron: car_params (7) → Cd (1)

This serves as the surrogate model for design optimization.
"""

import torch
import torch.nn as nn


class CdMLP(nn.Module):
    """
    MLP for predicting drag coefficient from car design parameters.
    
    Architecture: input(7) → hidden layers with ReLU → output(1)
    
    Args:
        in_features: number of input features (default: 7 car params)
        hidden_dim: hidden layer width
        n_layers: number of hidden layers
        dropout: dropout rate
    """

    def __init__(self, in_features: int = 7, hidden_dim: int = 128,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = in_features
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Input normalization stats (set during training)
        self.register_buffer('input_mean', torch.zeros(in_features))
        self.register_buffer('input_std', torch.ones(in_features))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set input normalization statistics."""
        self.input_mean = mean
        self.input_std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / (self.input_std + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) raw car parameters
        Returns:
            (batch, 1) predicted Cd
        """
        x = self.normalize(x)
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forward + squeeze."""
        return self.forward(x).squeeze(-1)


if __name__ == "__main__":
    model = CdMLP(in_features=7, hidden_dim=128, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CdMLP parameters: {n_params:,}")

    x = torch.randn(16, 7)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
