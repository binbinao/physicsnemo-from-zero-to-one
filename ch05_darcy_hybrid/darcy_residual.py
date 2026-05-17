"""
Chapter 5: Darcy PDE Residual via Finite Differences
======================================================
Compute the residual of the Darcy equation:
    -∇·(a(x) ∇u) = f

on a uniform grid using finite difference operators.
This is used as a physics loss for training physics-informed FNO.
"""

import torch
import torch.nn.functional as F


def darcy_residual(a: torch.Tensor, u: torch.Tensor,
                   f: float = 1.0, dx: float = None) -> torch.Tensor:
    """
    Compute Darcy PDE residual via central finite differences.
    
    PDE: -∇·(a(x) ∇u(x)) - f = 0
    
    Discretized:
        -[ (a_{i+1,j} + a_{i,j})/2 * (u_{i+1,j} - u_{i,j})
         - (a_{i,j} + a_{i-1,j})/2 * (u_{i,j} - u_{i-1,j}) ] / dx²
        -[ similar for j direction ] / dx²
        - f = 0

    Args:
        a: permeability field, shape (B, 1, H, W)
        u: solution field, shape (B, 1, H, W)
        f: forcing term (scalar)
        dx: grid spacing (default: 1/(H-1))

    Returns:
        Residual tensor of shape (B, 1, H-2, W-2) (interior points only)
    """
    if dx is None:
        dx = 1.0 / (u.shape[-1] - 1)

    # Extract shifted arrays for finite differences
    # u[i+1,j], u[i-1,j], u[i,j+1], u[i,j-1], u[i,j]
    u_ip = u[:, :, 2:, 1:-1]   # i+1
    u_im = u[:, :, :-2, 1:-1]  # i-1
    u_jp = u[:, :, 1:-1, 2:]   # j+1
    u_jm = u[:, :, 1:-1, :-2]  # j-1
    u_c  = u[:, :, 1:-1, 1:-1] # center

    # Permeability at half-points (averaged)
    a_ip_half = (a[:, :, 2:, 1:-1] + a[:, :, 1:-1, 1:-1]) / 2  # (i+1/2, j)
    a_im_half = (a[:, :, :-2, 1:-1] + a[:, :, 1:-1, 1:-1]) / 2  # (i-1/2, j)
    a_jp_half = (a[:, :, 1:-1, 2:] + a[:, :, 1:-1, 1:-1]) / 2  # (i, j+1/2)
    a_jm_half = (a[:, :, 1:-1, :-2] + a[:, :, 1:-1, 1:-1]) / 2  # (i, j-1/2)

    # -∇·(a ∇u) via finite differences
    flux_x = (a_ip_half * (u_ip - u_c) - a_im_half * (u_c - u_im)) / dx**2
    flux_y = (a_jp_half * (u_jp - u_c) - a_jm_half * (u_c - u_jm)) / dx**2

    residual = -(flux_x + flux_y) - f
    return residual


def darcy_residual_simple(u: torch.Tensor, f: float = 1.0,
                           dx: float = None) -> torch.Tensor:
    """
    Simplified Darcy residual for constant a=1 (Poisson equation):
        -∇²u = f  →  -(∂²u/∂x² + ∂²u/∂y²) - f = 0
    
    Using 5-point Laplacian stencil.
    """
    if dx is None:
        dx = 1.0 / (u.shape[-1] - 1)

    # Laplacian via conv2d with Laplacian kernel
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=u.dtype, device=u.device).view(1, 1, 3, 3)
    laplacian = F.conv2d(u, kernel, padding=0) / dx**2
    return -laplacian - f


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Darcy residual computation...")

    B, H, W = 4, 64, 64
    a = torch.ones(B, 1, H, W) * 5.0  # constant permeability
    # Exact solution for -5∇²u = 1 with u=0 on boundary:
    # Not trivial analytically, so just test shapes
    u = torch.randn(B, 1, H, W)

    res = darcy_residual(a, u, f=1.0)
    print(f"Input shape:    a={list(a.shape)}, u={list(u.shape)}")
    print(f"Residual shape: {list(res.shape)}")
    print(f"Residual stats: mean={res.mean().item():.4f}, "
          f"std={res.std().item():.4f}")

    res_simple = darcy_residual_simple(u, f=1.0)
    print(f"Simple residual shape: {list(res_simple.shape)}")
    print("OK!")
