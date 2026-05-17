"""
Chapter 3: Heat Conduction Equations (Raw PyTorch autograd)
============================================================
PDE:  ∇·(k ∇T) + Q = 0   →  k (∂²T/∂x² + ∂²T/∂y²) + Q = 0

For steady-state 2D heat conduction with constant conductivity k.

Robin BC:  -k ∂T/∂n = h (T - T_inf)
    where h = convection coefficient, T_inf = ambient temperature.
"""

import torch


def heat_conduction_2d(T: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                       k: float = 1.0, Q: float = 0.0) -> torch.Tensor:
    """
    Compute the PDE residual:  k(∂²T/∂x² + ∂²T/∂y²) + Q = 0

    Args:
        T: predicted temperature field, shape (N, 1)
        x, y: spatial coordinates with requires_grad=True
        k: thermal conductivity
        Q: volumetric heat source

    Returns:
        Residual tensor of shape (N, 1). Should be zero if PDE is satisfied.
    """
    # First derivatives
    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    dT_dy = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]

    # Second derivatives
    d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx),
                                   create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y, grad_outputs=torch.ones_like(dT_dy),
                                   create_graph=True)[0]

    return k * (d2T_dx2 + d2T_dy2) + Q


def robin_boundary(T: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                   nx: torch.Tensor, ny: torch.Tensor,
                   k: float = 1.0, h_conv: float = 10.0,
                   T_inf: float = 0.0) -> torch.Tensor:
    """
    Robin boundary residual:  -k ∂T/∂n - h(T - T_inf) = 0

    ∂T/∂n = (∂T/∂x) nx + (∂T/∂y) ny

    Args:
        T: predicted temperature, shape (N, 1)
        x, y: boundary coordinates
        nx, ny: outward unit normals
        k: thermal conductivity
        h_conv: convective heat transfer coefficient
        T_inf: ambient temperature

    Returns:
        Robin residual, shape (N, 1)
    """
    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    dT_dy = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]

    dT_dn = dT_dx * nx + dT_dy * ny
    return -k * dT_dn - h_conv * (T - T_inf)


def neumann_boundary(T: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                     nx: torch.Tensor, ny: torch.Tensor) -> torch.Tensor:
    """
    Neumann (insulated) boundary: ∂T/∂n = 0

    Returns:
        Normal derivative residual, shape (N, 1)
    """
    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    dT_dy = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    return dT_dx * nx + dT_dy * ny
