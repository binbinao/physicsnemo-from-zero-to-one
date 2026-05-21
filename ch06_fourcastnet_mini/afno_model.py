"""
Chapter 6: Simplified AFNO (Adaptive Fourier Neural Operator)
================================================================
A minimal version of the AFNO block used in FourCastNet.

Key idea: Replace standard attention with Fourier-space token mixing.
Each "token" is a spatial location; mixing happens via learned weights
in the frequency domain.

Reference: Guibas et al., "Adaptive Fourier Neural Operators" (2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator block.
    
    1. Layer norm
    2. Fourier token mixing (global, in frequency domain)
    3. Residual connection
    4. Layer norm + MLP + Residual
    
    Args:
        dim: embedding dimension (channels)
        mlp_ratio: MLP hidden dim = dim * mlp_ratio
        n_blocks: number of frequency-domain blocks (for weight sharing)
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, n_blocks: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Fourier mixing weights (complex-valued, stored as real pairs)
        self.n_blocks = n_blocks
        block_size = dim // n_blocks
        self.scale = 0.002
        # Weights for frequency-domain MLP: two layers
        self.w1 = nn.Parameter(self.scale * torch.randn(2, n_blocks, block_size, block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, n_blocks, block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, n_blocks, block_size, block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, n_blocks, block_size))

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def complex_mul(self, x_real, x_imag, w_real, w_imag, b_real, b_imag):
        """Complex matrix multiply + bias: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        x_real/x_imag: (B, H, W2, n_blocks, block_size)
        w_real/w_imag: (n_blocks, block_size, block_size)
        b_real/b_imag: (n_blocks, block_size)
        """
        out_real = torch.einsum("...bi,bij->...bj", x_real, w_real) - \
                   torch.einsum("...bi,bij->...bj", x_imag, w_imag) + b_real
        out_imag = torch.einsum("...bi,bij->...bj", x_real, w_imag) + \
                   torch.einsum("...bi,bij->...bj", x_imag, w_real) + b_imag
        return out_real, out_imag

    def fourier_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C)
        Apply mixing in Fourier domain along spatial dimensions.
        """
        B, H, W, C = x.shape
        block_size = C // self.n_blocks

        # 2D FFT along spatial dims
        x_ft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        # Reshape for block-wise processing
        x_ft_real = x_ft.real.reshape(B, H, W // 2 + 1, self.n_blocks, block_size)
        x_ft_imag = x_ft.imag.reshape(B, H, W // 2 + 1, self.n_blocks, block_size)

        # Apply learned frequency-domain MLP (2 layers with soft thresholding)
        # Layer 1
        o_real, o_imag = self.complex_mul(
            x_ft_real, x_ft_imag,
            self.w1[0], self.w1[1], self.b1[0], self.b1[1])
        # Soft thresholding (AFNO-style sparsity)
        magnitude = torch.sqrt(o_real**2 + o_imag**2 + 1e-8)
        # NOTE: Original AFNO uses a learnable threshold parameter; fixed here for simplicity
        shrink = F.relu(magnitude - 0.01) / (magnitude + 1e-8)
        o_real = o_real * shrink
        o_imag = o_imag * shrink
        # Layer 2
        o_real, o_imag = self.complex_mul(
            o_real, o_imag,
            self.w2[0], self.w2[1], self.b2[0], self.b2[1])

        # Reshape back
        x_ft_out = torch.complex(
            o_real.reshape(B, H, W // 2 + 1, C),
            o_imag.reshape(B, H, W // 2 + 1, C))

        # Inverse FFT
        return torch.fft.irfft2(x_ft_out, s=(H, W), dim=(1, 2), norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        # Fourier mixing branch
        residual = x
        x = self.norm1(x)
        x = self.fourier_mixing(x)
        x = x + residual

        # MLP branch
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class AFNOMini(nn.Module):
    """
    Mini FourCastNet-style model.
    
    Architecture:
        Input (C_in, H, W) → Patch embed → N AFNO blocks → Head → Output (C_out, H, W)
    
    For simplicity, we use patch_size=1 (pixel-level tokens).
    
    Args:
        in_channels: input weather variables
        out_channels: output weather variables (usually same as in)
        embed_dim: internal embedding dimension
        depth: number of AFNO blocks
        mlp_ratio: MLP expansion ratio
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 embed_dim: int = 64, depth: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        # NOTE: Original FourCastNet uses patch_size=4/8 for efficiency; pixel-level used here for simplicity
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            AFNOBlock(embed_dim, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.head = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H, W)
        """
        # Embed: (B, C, H, W) -> (B, D, H, W)
        x = self.embed(x)
        # Reshape to (B, H, W, D) for AFNO blocks
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        # Back to (B, D, H, W)
        x = x.permute(0, 3, 1, 2)
        return self.head(x)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing AFNO Mini...")
    model = AFNOMini(in_channels=4, out_channels=4, embed_dim=64, depth=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(2, 4, 32, 32)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch!"
    print("OK!")
