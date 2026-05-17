"""
Chapter 4: Fourier Neural Operator (FNO2D) from Scratch
=========================================================
A minimal FNO implementation in pure PyTorch.

Key components:
  - SpectralConv2d: Fourier-space convolution layer
  - FNOBlock: SpectralConv + bypass + activation
  - FNO2D: Lifting → FNO blocks → Projection

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: multiply in Fourier space then transform back.
    
    Learns complex-valued weights R of shape (in_channels, out_channels, modes_x, modes_y)
    and applies: output = iFFT(R · FFT(input)) for the retained modes.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int = 12, modes_y: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)
        # Two sets of weights for the two corners of the 2D frequency domain
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))

    def compl_mul2d(self, input_tensor, weights):
        """Complex multiplication: (batch, in, x, y) × (in, out, x, y) → (batch, out, x, y)"""
        # Convert real tensor with last dim=2 to complex
        inp = torch.view_as_complex(input_tensor)
        w = torch.view_as_complex(weights)
        return torch.view_as_real(torch.einsum("bixy,ioxy->boxy", inp, w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # FFT
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.view_as_real(x_ft)

        # Allocate output in Fourier space
        out_ft = torch.zeros(batch_size, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1, 2,
                             device=x.device, dtype=x.dtype)

        # Multiply with retained modes
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y],
                            self.weights1)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y],
                            self.weights2)

        # Inverse FFT
        out_ft = torch.view_as_complex(out_ft)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock(nn.Module):
    """Single FNO block: spectral convolution + 1x1 conv bypass + GELU."""

    def __init__(self, width: int, modes_x: int = 12, modes_y: int = 12):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes_x, modes_y)
        self.bypass = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral_conv(x) + self.bypass(x)))


class FNO2D(nn.Module):
    """
    Fourier Neural Operator for 2D problems.
    
    Architecture:
        Input (in_channels) → Lifting (1x1 conv) → N FNO blocks → Projection → Output (out_channels)
    
    Args:
        in_channels: number of input channels (e.g., 1 for permeability field)
        out_channels: number of output channels (e.g., 1 for solution field)
        width: hidden channel width in FNO blocks
        modes_x, modes_y: number of Fourier modes to retain
        n_layers: number of FNO blocks
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 width: int = 32, modes_x: int = 12, modes_y: int = 12,
                 n_layers: int = 4):
        super().__init__()
        self.lifting = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes_x, modes_y) for _ in range(n_layers)
        ])
        self.projection = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
        Returns:
            (batch, out_channels, H, W)
        """
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        return self.projection(x)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing FNO2D...")
    model = FNO2D(in_channels=1, out_channels=1, width=32,
                  modes_x=12, modes_y=12, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (4, 1, 64, 64), "Shape mismatch!"
    print("OK!")
