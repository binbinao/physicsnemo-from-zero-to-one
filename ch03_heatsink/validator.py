"""
Chapter 3: Pointwise validation report (CAE-style sanity checks).
================================================================
Not a sign-off tool — compares PINN against:
  - PDE / BC residual RMS on dense samples
  - Order-of-magnitude 1D fin tip temperature bound
  - Physical fin-temperature probes + 1D fin analytic cross-check

Residual conventions (see docs/CAE_UNITS_AND_NONDIM.md, "ch03 单位规则")
----------------------------------------------------------------------
Two families of numbers are reported, both from the *same* point set:

* **raw** (legacy fields, unchanged meaning): ``pde_residual_rms`` and
  ``robin_residual_rms`` are w.r.t. the **normalised** coordinates the network
  sees. ``robin_residual_rms`` uses the checkpoint's own convention
  (``robin_scale_used``): 0.03 → physical ``-(k/L0)∂T/∂ñ - hΔT``; 0 → the
  pre-fix expression ``-k∂T/∂ñ - hΔT`` (equivalent to ``h_eff = h/L0``).
* **relative** (new, dimensionless, comparable across checkpoints):
  ``pde_relative_rms = pde_residual_rms / (k·ΔT)`` and
  ``robin_relative_rms = robin_residual_rms_reference / (h·ΔT)`` with
  ``ΔT = T_source - T_inf``. The gate ``GATE = 0.1`` means "10 % of the
  characteristic flux", which is the only reading of ``0.1`` that is
  physically meaningful (raw residuals are in W/m², so a bare ``0.1`` there
  would mean 0.1 W/m², i.e. 1e-4 relative — unreachable and unit-less).

Usage:
    python validator.py --checkpoint outputs/model.pt   # if saved
    python validator.py --checkpoint outputs/model.pt --robin_scale 0.03 --seed 0
    # Or called automatically at end of heat_sink_train.py
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch

from heat_sink_geometry import HeatSinkGeometry
from equations import (L0_SI, heat_conduction_2d, resolve_robin_scale,
                       robin_boundary, neumann_boundary)

#: Dimensionless residual gate (fraction of the characteristic flux).
GATE = 1e-1

#: Fixed physical probes (mm) on the middle fin axis: root / mid height / tip.
#: The middle fin is centred at x = 0 (spacing = 60/(3+1) = 15 mm), so its axis
#: is x = 0 and its faces are x = ±2 mm. Fixed points keep the probes free of
#: sampling noise.
PROBES = {
    "T_fin_root": (0.0, 10.5),
    "T_fin_mid": (0.0, 20.0),
    "T_fin_tip": (0.0, 29.5),
}


def fin_tip_temperature_bound(T_source: float, T_inf: float, h: float, k: float,
                            fin_length: float) -> float:
    """1D fin with convection at tip: loose upper bound on |T_source - T_tip|."""
    m = np.sqrt(h / max(k * fin_length, 1e-6))
    eta = np.tanh(m * fin_length) / (m * fin_length + 1e-8)
    T_tip = T_inf + (T_source - T_inf) * (1.0 - min(eta, 1.0))
    return float(T_tip)


def fin_tip_1d_from_root(T_root: float, T_inf: float, h: float, k: float,
                         fin_height_mm: float, fin_width_mm: float) -> float:
    """1D rectangular-fin tip temperature driven by the *measured* root temp.

    Solves ``d²θ/dx² = m²θ`` with ``m = sqrt(2h/(k·t))`` (convection on both
    faces of a fin of thickness ``t``) and convection at the tip:

        θ_tip / θ_root = 1 / (cosh(mL) + (h/(m·k))·sinh(mL))

    **Unit trap (guarded here):** ``h`` and ``k`` are SI, so the geometry must
    be converted to **metres** before use. Feeding millimetres gives
    ``mL ≈ 44.7`` → ``cosh`` overflows → a bogus ``≈0 ℃`` tip, which would
    make a healthy fin look like the old "cold fin" artefact.

    Args:
        T_root: fin root temperature (same unit as ``T_inf``), ℃ here.
        T_inf: ambient temperature, ℃.
        h: convection coefficient, W/(m²·K).
        k: conductivity, W/(m·K).
        fin_height_mm: fin height in **millimetres** (converted internally).
        fin_width_mm: fin thickness in **millimetres** (converted internally).
    """
    L = float(fin_height_mm) / 1000.0
    t = float(fin_width_mm) / 1000.0
    if L <= 0.0 or t <= 0.0 or k <= 0.0 or h <= 0.0:
        return float(T_inf)
    m = math.sqrt(2.0 * h / (k * t))
    mL = m * L
    if mL > 50.0:                      # cosh overflow guard; θ_tip → 0 anyway
        return float(T_inf)
    r = h / (m * k)
    theta = (T_root - T_inf) / (math.cosh(mL) + r * math.sinh(mL))
    return float(T_inf + theta)


def _tensor(a, scale: float, device: str, grad: bool = False) -> torch.Tensor:
    t = torch.tensor(np.asarray(a) / scale, dtype=torch.float32, device=device).unsqueeze(1)
    if grad:
        t.requires_grad_(True)
    return t


def run_validation(
    model: torch.nn.Module,
    geo: HeatSinkGeometry,
    cfg: dict,
    device: str = "cpu",
    n_int: int = 2000,
    n_bc: int = 400,
) -> dict[str, Any]:
    model.eval()
    k = cfg.get("k", 1.0)
    Q = cfg.get("Q", 0.0)
    T_source = cfg.get("T_source", 100.0)
    h_conv = cfg.get("h_conv", 10.0)
    T_inf = cfg.get("T_inf", 0.0)
    scale = cfg.get("scale", 30.0)
    # Robin convention of *this* evaluation: 0.03 = physical, 0 = legacy
    # (pre-fix). Old checkpoints have no "robin_scale" key -> 0.0 -> legacy, so
    # their historical numbers are reproduced exactly.
    rs = resolve_robin_scale(cfg.get("robin_scale", 0.0))

    # Interior PDE residual
    x, y = geo.sample_interior(n_int)
    x_t = _tensor(x, scale, device, grad=True)
    y_t = _tensor(y, scale, device, grad=True)
    T = model(x_t, y_t)
    res_pde = heat_conduction_2d(T, x_t, y_t, k=k, Q=Q)
    pde_rms = float(torch.sqrt((res_pde**2).mean()).detach().cpu())

    # Robin on fin surfaces. The point set is sampled ONCE and both residuals
    # reuse it, so 口径 comparisons are point-wise identical.
    xr, yr, nx, ny = geo.sample_robin_bc(n_bc)
    xr_t = _tensor(xr, scale, device, grad=True)
    yr_t = _tensor(yr, scale, device, grad=True)
    nx_t = _tensor(nx, scale=1.0, device=device)
    ny_t = _tensor(ny, scale=1.0, device=device)
    Tr = model(xr_t, yr_t)
    res_robin = robin_boundary(Tr, xr_t, yr_t, nx_t, ny_t,
                               k=k, h_conv=h_conv, T_inf=T_inf, scale=rs)
    robin_rms = float(torch.sqrt((res_robin**2).mean()).detach().cpu())
    # Same point set, always the *physical* convention -> comparable across ckpts
    res_robin_ref = robin_boundary(Tr, xr_t, yr_t, nx_t, ny_t,
                                   k=k, h_conv=h_conv, T_inf=T_inf, scale=L0_SI)
    robin_rms_ref = float(torch.sqrt((res_robin_ref**2).mean()).detach().cpu())

    # Temperature range vs 1D bound
    with torch.no_grad():
        T_vals = T.detach().cpu().numpy()
    T_max = float(T_vals.max())
    T_min = float(T_vals.min())
    fin_h = cfg.get("fin_height", 20.0)
    fin_w = cfg.get("fin_width", 4.0)
    T_tip_bound = fin_tip_temperature_bound(T_source, T_inf, h_conv, k, fin_h / scale)

    # Fixed physical probes (no sampling noise)
    probes: dict[str, float] = {}
    with torch.no_grad():
        for name, (px, py) in PROBES.items():
            probes[name] = float(model(_tensor([px], scale, device),
                                       _tensor([py], scale, device)).item())
    tip_1d = fin_tip_1d_from_root(probes["T_fin_root"], T_inf, h_conv, k, fin_h, fin_w)

    # Dimensionless (relative) residuals — convention A, single definition
    dT = float(T_source) - float(T_inf)
    pde_den = abs(k * dT) if abs(k * dT) > 0 else 1.0
    robin_den = abs(h_conv * dT) if abs(h_conv * dT) > 0 else 1.0
    pde_rel = pde_rms / pde_den
    robin_rel = robin_rms_ref / robin_den

    report = {
        # --- legacy fields: names, types and meaning unchanged ---------------
        "pde_residual_rms": pde_rms,
        "robin_residual_rms": robin_rms,
        "T_min": T_min,
        "T_max": T_max,
        "T_source_config": T_source,
        "fin_tip_1d_bound": T_tip_bound,
        "pass_pde": pde_rms < GATE,
        "pass_robin": robin_rms < GATE,
        "pass_temperature_range": T_min >= T_inf - 5.0 and T_max <= T_source + 50.0,
        "note": "2D PINN micro-model; not comparable to Icepak sign-off without imported CFD reference.",
        "pass_all": None,   # filled below (order kept: legacy keys first)
        # --- new fields: doubly-reported residuals + physical probes --------
        "robin_scale_used": float(rs if rs is not None else 0.0),
        "robin_residual_rms_reference": robin_rms_ref,
        "pde_relative_rms": pde_rel,
        "robin_relative_rms": robin_rel,
        "T_fin_root": probes["T_fin_root"],
        "T_fin_mid": probes["T_fin_mid"],
        "T_fin_tip": probes["T_fin_tip"],
        "fin_tip_1d_from_root": tip_1d,
        "pass_pde_relative": pde_rel < GATE,
        "pass_robin_relative": robin_rel < GATE,
        "pass_all_relative": None,   # filled below
        "convention": ("robin_relative = robin_rms_ref/(h*dT); "
                       "pde_relative = pde_rms/(k*dT); dT = T_source - T_inf; "
                       f"L0 = {L0_SI} m"),
    }
    report["pass_all"] = (report["pass_pde"] and report["pass_robin"]
                          and report["pass_temperature_range"])
    report["pass_all_relative"] = (report["pass_pde_relative"]
                                   and report["pass_robin_relative"]
                                   and report["pass_temperature_range"])
    return report


def load_and_validate(ckpt_path: str, device: str = "cpu",
                      robin_scale: float | None = None,
                      seed: int | None = None) -> dict:
    """Rebuild the checkpoint's model and run :func:`run_validation` on it.

    Args:
        ckpt_path: checkpoint written by ``heat_sink_train.py``.
        device: torch device string.
        robin_scale: optional override of the Robin convention (``0.03`` =
            physical, ``0`` = legacy). ``None`` (default) uses the checkpoint's
            own ``robin_scale``; checkpoints without that key fall back to
            ``0.0`` = legacy, so their historical numbers do not drift.
        seed: optional ``np.random`` / ``torch`` seed for the sampling points.
            ``None`` (default) leaves the sampling exactly as before.
    """
    from heat_sink_train import HeatSinkNet
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config", {}) or {})
    cfg["scale"] = ckpt.get("scale", 30.0)
    if robin_scale is None:
        cfg["robin_scale"] = float(ckpt.get("robin_scale", 0.0))
    else:
        cfg["robin_scale"] = float(robin_scale)
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
    geo = HeatSinkGeometry(fin_height=cfg.get("fin_height", 20.0), n_fins=cfg.get("n_fins", 3))
    model = HeatSinkNet(
        hidden=int(cfg.get("hidden", 128)),
        depth=int(cfg.get("depth", 5)),
        hard_bottom=bool(cfg.get("hard_bottom", False)),
        T_source=float(cfg.get("T_source", 100.0)),
        activation=str(cfg.get("activation", "tanh")),
        fourier_maps=int(cfg.get("fourier_maps", 0)),
        fourier_scale=float(cfg.get("fourier_scale", 6.0)),
        seed=int(cfg.get("seed", 0)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return run_validation(model, geo, cfg, device=device)


def save_report(report: dict, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "validation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run ch03 PINN validation report")
    p.add_argument("--checkpoint", default="outputs/heat_sink.pt")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--robin_scale", type=float, default=None,
                   help="Override the Robin convention: 0.03 = physical (SI m), "
                        "0 = legacy/pre-fix. Default: use the checkpoint's own value.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the sampling points (default: unchanged behaviour).")
    args = p.parse_args()
    rep = load_and_validate(args.checkpoint, robin_scale=args.robin_scale, seed=args.seed)
    print(json.dumps(rep, indent=2))
    print("Saved:", save_report(rep, args.out_dir))
