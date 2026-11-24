"""
Chapter 3: Heat Conduction Equations (Raw PyTorch autograd)
============================================================
PDE:  ∇·(k ∇T) + Q = 0   →  k (∂²T/∂x² + ∂²T/∂y²) + Q = 0

For steady-state 2D heat conduction with constant conductivity k.

Robin BC (physical, SI):  -k ∂T/∂n = h (T - T_inf)
    where h = convection coefficient, T_inf = ambient temperature,
    and n is measured in metres (see "Units" below).

Units / coordinate normalisation
--------------------------------
This module works in two coordinate systems and they must not be mixed:

* ``k``, ``h_conv``, ``T`` are in **SI**-consistent units, so ``∂T/∂n`` in the
  Robin condition is a derivative w.r.t. **metres**.
* The training/validation scripts feed the network coordinates normalised by
  ``scale = COORD_SCALE_MM = 30.0`` (mm → ~unit box), i.e. ``x̃ = x_mm / 30``.
  Hence

      ∂T/∂n_SI = (1 / L0) · ∂T/∂ñ ,      L0 = 0.03 m   (= 30 mm)

  and the physical Robin residual is ``-(k/L0)·∂T/∂ñ - h(T - T_inf)``.

Pass ``scale=L0_SI`` (0.03, **metres**, not 30) to ``robin_boundary`` when the
coordinates handed to autograd are normalised. Leaving ``scale=None``
reproduces the historical behaviour bit-for-bit (the ``1/L0`` factor is
missing there, which is equivalent to using ``h_eff = h / L0 = 333 W/(m²·K)``).
"""

import warnings

import torch


#: Coordinate normalisation constant used by the ch03 scripts, in **millimetres**
#: (network input is ``x_mm / COORD_SCALE_MM``). Single source of truth: the
#: scripts import this instead of hard-coding ``30.0``.
COORD_SCALE_MM = 30.0

#: Physical length corresponding to **one normalised coordinate unit**, in SI
#: metres: ``L0_SI = COORD_SCALE_MM / 1000 = 0.03``. This is the "restore
#: length" of the coordinate system (30 mm expressed in metres), *not* the fin
#: height and *not* the number 30. It is derived from ``COORD_SCALE_MM`` so the
#: two can never drift apart: change the normalisation and this follows.
L0_SI = COORD_SCALE_MM / 1000.0


def resolve_robin_scale(value) -> "float | None":
    """Map a config/CLI value onto the ``scale`` argument of :func:`robin_boundary`.

    ``value > 0``    → ``float(value)``  (physical characteristic length, SI metres)
    ``value <= 0`` / ``None`` → ``None`` (legacy: reproduce the pre-fix numerics)

    ``0`` (or negative) is therefore the documented "legacy off" sentinel and
    must not be read as "use the default": ``None`` as a *config default* is not
    allowed because ``heat_sink_train.main_argparse`` derives ``type=type(v)``
    from the default value.
    """
    if value is None:
        return None
    v = float(value)
    return v if v > 0.0 else None


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
                   T_inf: float = 0.0,
                   scale: "float | None" = None) -> torch.Tensor:
    """
    Robin boundary residual:  -(k / L0) ∂T/∂ñ - h(T - T_inf) = 0

    ∂T/∂ñ = (∂T/∂x̃) nx + (∂T/∂ỹ) ny

    Args:
        T: predicted temperature, shape (N, 1)
        x, y: boundary coordinates **as fed to the network** (normalised)
        nx, ny: outward unit normals
        k: thermal conductivity (SI with h_conv)
        h_conv: convective heat transfer coefficient (SI)
        T_inf: ambient temperature
        scale: **physical length of one normalised coordinate unit, in the same
            unit system as ``k`` and ``h_conv``** (``L0 = COORD_SCALE_MM/1000 =
            0.03 m``). Converts the normalised-coordinate normal derivative back
            to a physical one (``∂T/∂n = (1/L0)·∂T/∂ñ``). It follows the
            *coordinate normalisation*, not the geometry: a taller fin does not
            change it, a different ``COORD_SCALE_MM`` does.

            - ``None`` (default): legacy behaviour, ``-k ∂T/∂ñ - h(T-T_inf)``.
              This is *bit-for-bit* what the pre-fix implementation computed
              (it silently misses the ``1/L0`` factor, i.e. it solves the same
              problem with ``h_eff = h/L0``); kept for replaying old
              checkpoints and old experiment numbers.
            - SI units (``k=1 W/(m·K)``, ``h=10 W/(m²·K)``): must be ``0.03``
              (= 30 mm expressed in **metres**), *not* ``30``. Passing ``30``
              makes the conduction term 1000x too small, which is the same as
              solving the physical problem with ``h_eff = h·s/L0 = 1000·h``
              (convection 1000x too strong) — a different, even colder, but
              equally well-converging wrong solution.

    Returns:
        Robin residual, shape (N, 1). Units: W/m².
    """
    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    dT_dy = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]

    dT_dn = dT_dx * nx + dT_dy * ny
    if scale is None:
        # Legacy path: no extra arithmetic, identical to the pre-fix expression.
        return -k * dT_dn - h_conv * (T - T_inf)

    L0 = float(scale)
    if L0 <= 0.0:
        raise ValueError(
            f"robin_boundary: scale must be > 0, got {L0}. scale is the "
            f"physical characteristic length in SI metres (30 mm -> 0.03); "
            f"use 0/None only to select the legacy (pre-fix) behaviour."
        )
    if L0 >= 1.0:
        warnings.warn(
            f"robin_boundary: scale={L0} looks like millimetres. L0 must use "
            f"the same unit system as k/h_conv (SI: 0.03 for this 30 mm model).",
            stacklevel=2,
        )
    return -(k / L0) * dT_dn - h_conv * (T - T_inf)


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
