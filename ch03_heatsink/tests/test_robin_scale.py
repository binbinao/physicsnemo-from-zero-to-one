#!/usr/bin/env python3
"""Regression tests for the ch03 Robin units fix (``robin_boundary(scale=...)``).

Plain ``assert`` + stdlib + torch only — the repo has no pytest dependency:

    cd ch03_heatsink && python3 tests/test_robin_scale.py

Cases
-----
C1  legacy path (``scale=None``) is bit-for-bit identical to the pre-fix code.
    The golden numbers below were produced by running this exact snippet on
    baseline commit ``07117d8`` (before the fix existed):

        import torch
        from equations import robin_boundary
        i = torch.arange(5, dtype=torch.float64)
        x = (i*0.37 + 0.11).unsqueeze(1).requires_grad_(True)
        y = (i*0.23 - 0.07).unsqueeze(1).requires_grad_(True)
        T = 0.5*x**2 + 0.3*y + 2.0
        nx = torch.full((5,1), 0.6, dtype=torch.float64)
        ny = torch.full((5,1), 0.8, dtype=torch.float64)
        print([repr(float(v)) for v in robin_boundary(T, x, y, nx, ny).flatten()])

C2  physical path (``scale=L0_SI``) matches the hand-computed
    ``-(k/L0)·∂T/∂ñ − h(T−T_inf)`` for an analytic field.
C3  guard rail: ``scale=30.0`` (millimetres — the classic mistake) differs from
    ``scale=0.03`` by exactly 1000× in the conduction term.
C4  1D fin analytic tip temperature (mm/m unit trap: with millimetres the
    probe returns ~0 ℃ instead of ~30 ℃).
C5  sentinel / validation behaviour of ``resolve_robin_scale`` and ``scale``.
C6  optional (``--with-fv``): 1 mm finite-volume reference solution, pure
    stdlib, ≈5–15 s; checks the fin tip lands in 26–38 ℃, i.e. the *column ①*
    of the fix report, not the ~0 ℃ of the buggy ``h_eff=333`` equation.
"""

import math
import os
import sys
import warnings

_CH03_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CH03_DIR not in sys.path:
    sys.path.insert(0, _CH03_DIR)

import torch  # noqa: E402

from equations import L0_SI, resolve_robin_scale, robin_boundary  # noqa: E402

# Golden output of the pre-fix implementation (baseline commit 07117d8).
GOLDEN_LEGACY = [
    -20.1565,
    -22.159999999999997,
    -25.5325,
    -30.274,
    -36.3845,
]


def _field():
    """Fixed analytic field T = 0.5x² + 0.3y + 2 on 5 points (float64)."""
    i = torch.arange(5, dtype=torch.float64)
    x = (i * 0.37 + 0.11).unsqueeze(1).requires_grad_(True)
    y = (i * 0.23 - 0.07).unsqueeze(1).requires_grad_(True)
    T = 0.5 * x ** 2 + 0.3 * y + 2.0
    nx = torch.full((5, 1), 0.6, dtype=torch.float64)
    ny = torch.full((5, 1), 0.8, dtype=torch.float64)
    return T, x, y, nx, ny


# ---------------------------------------------------------------------------
# C1 — legacy default path is bit-for-bit unchanged
# ---------------------------------------------------------------------------
def case1_legacy_bitwise_identical():
    T, x, y, nx, ny = _field()
    out = robin_boundary(T, x, y, nx, ny)  # scale defaults to None
    got = [float(v) for v in out.flatten().tolist()]
    max_delta = max(abs(a - b) for a, b in zip(got, GOLDEN_LEGACY))
    assert max_delta == 0.0, f"C1: legacy path drifted, max|d|={max_delta}, got={got}"

    # float32 as well (the dtype the training loop actually uses)
    x32 = x.detach().float().requires_grad_(True)
    y32 = y.detach().float().requires_grad_(True)
    T32 = 0.5 * x32 ** 2 + 0.3 * y32 + 2.0
    out32 = robin_boundary(T32, x32, y32, nx.float(), ny.float())
    expect32 = [
        -20.1564998626709,
        -22.15999984741211,
        -25.532501220703125,
        -30.274002075195312,
        -36.38450241088867,
    ]
    max_delta32 = max(abs(float(a) - b) for a, b in zip(out32.flatten().tolist(), expect32))
    assert max_delta32 == 0.0, f"C1: float32 legacy drifted, max|d|={max_delta32}"
    print("C1 ok  legacy scale=None bit-for-bit identical (float64 & float32)")


# ---------------------------------------------------------------------------
# C2 — physical formula on an analytic field
# ---------------------------------------------------------------------------
def case2_physical_formula():
    # T = a*y  ->  ∂T/∂x = 0, ∂T/∂y = a, n = (0, 1)
    a, k, h, T_inf = 2.0, 1.0, 10.0, 0.0
    n = 7
    x = torch.zeros(n, 1, dtype=torch.float64, requires_grad=True)
    y = (torch.arange(n, dtype=torch.float64) * 0.05).unsqueeze(1).requires_grad_(True)
    T = a * y + 0.0 * x          # x kept in the graph so dT/dx is well defined
    nx = torch.zeros(n, 1, dtype=torch.float64)
    ny = torch.ones(n, 1, dtype=torch.float64)
    out = robin_boundary(T, x, y, nx, ny, k=k, h_conv=h, T_inf=T_inf, scale=L0_SI)
    expect = -(k / L0_SI) * a * ny - h * (T.detach() - T_inf)
    rel = (out.detach() - expect).abs() / expect.abs().clamp_min(1e-12)
    assert float(rel.max()) <= 1e-6, f"C2: rel err {float(rel.max())} > 1e-6"
    # and the magnitude is ~ -(1/0.03)*2 = -66.7, i.e. 33x the legacy -2
    assert abs(float(out.detach()[0, 0]) - (-(k / L0_SI) * a - h * 0.0)) < 1e-9, out[0, 0]
    print("C2 ok  -(k/L0)·∂T/∂ñ − h(T−T∞) reproduced (rtol ≤ 1e-6)")


# ---------------------------------------------------------------------------
# C3 — 0.03 (metres) vs 30.0 (millimetres) differ by 1000x
# ---------------------------------------------------------------------------
def case3_mm_vs_m_guard():
    T, x, y, nx, ny = _field()
    out_m = robin_boundary(T, x, y, nx, ny, k=1.0, h_conv=10.0, T_inf=0.0, scale=L0_SI)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_mm = robin_boundary(T, x, y, nx, ny, k=1.0, h_conv=10.0, T_inf=0.0, scale=30.0)
    assert any(issubclass(w.category, UserWarning) for w in caught), \
        "C3: scale=30 must raise a UserWarning (looks like millimetres)"

    # strip the (identical) convection term, keep only the conduction term
    Td = T.detach()
    cond_m = out_m.detach() + 10.0 * (Td - 0.0)
    cond_mm = out_mm.detach() + 10.0 * (Td - 0.0)
    # scale=30 (mm) divides the conduction term by 30 instead of 0.03, i.e. the
    # term becomes 1000x weaker. Both branches "converge"; only 0.03 is physical.
    ratio = cond_m / cond_mm
    assert float((ratio / 1000.0 - 1.0).abs().max()) < 1e-9, float(ratio.max())
    print("C3 ok  scale=30 (mm) keeps the conduction term 1000x smaller than 0.03 "
          "(<=> h_eff = h*1000, still colder) + warning")


# ---------------------------------------------------------------------------
# C4 — 1D fin analytic solution, millimetre/metre trap
# ---------------------------------------------------------------------------
def case4_fin_tip_1d():
    from validator import fin_tip_1d_from_root

    tip = fin_tip_1d_from_root(T_root=74.9, T_inf=0.0, h=10.0, k=1.0,
                               fin_height_mm=20.0, fin_width_mm=4.0)
    assert abs(tip - 30.55) <= 0.05, f"C4: expected 30.55 ± 0.05 ℃, got {tip}"
    # m·L must be O(1), not O(45): millimetres would give cosh(44.7) -> tip ≈ 0
    L, t = 0.020, 0.004
    m = math.sqrt(2.0 * 10.0 / (1.0 * t))
    assert abs(m * L - 1.4142) < 1e-3, m * L
    print(f"C4 ok  fin_tip_1d_from_root(74.9) = {tip:.3f} ℃  (mL={m*L:.4f})")


# ---------------------------------------------------------------------------
# C5 — sentinel + validation of scale
# ---------------------------------------------------------------------------
def case5_sentinel_and_validation():
    assert resolve_robin_scale(None) is None
    assert resolve_robin_scale(0) is None
    assert resolve_robin_scale(0.0) is None
    assert resolve_robin_scale(-1.0) is None
    assert resolve_robin_scale(0.03) == 0.03
    assert isinstance(resolve_robin_scale("0.03"), float)

    T, x, y, nx, ny = _field()
    try:
        robin_boundary(T, x, y, nx, ny, scale=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("C5: scale=0 (after resolve) must raise ValueError")
    print("C5 ok  resolve_robin_scale sentinel + ValueError on non-positive scale")


# ---------------------------------------------------------------------------
# C6 — optional 1 mm finite-volume reference (pure stdlib, ~5–15 s)
# ---------------------------------------------------------------------------
def reference_fv(h_conv: float, d_mm: float = 1.0,
                 n_iter_max: int = 200000, tol: float = 1e-9) -> dict:
    """Steady 2D conduction on the ch03 geometry, ``d_mm`` finite-volume cells.

    Base 60x10 mm, 3 fins 4x20 mm (cx = -15, 0, 15 mm); bottom Dirichlet
    100 ℃; base side walls adiabatic; fin faces + exposed base top Robin
    (alternating-direction Gauss–Seidel, in-place).

    ``h_conv`` is the *physical* coefficient: pass 10 for the true physics, or
    ``10/L0_SI = 333.3`` to emulate the pre-fix equation (``h_eff = h/L0``).
    ``k = 1 W/(m·K)``, so the conduction conductance between two cells equals
    the Robin conductance of one exposed face, ``h·Δ``.
    """
    k, T_inf, T_src = 1.0, 0.0, 100.0
    base_w, base_h, fin_w, fin_h, n_fins = 60.0, 10.0, 4.0, 20.0, 3
    nx, ny = int(round(base_w / d_mm)), int(round((base_h + fin_h) / d_mm))
    spacing = base_w / (n_fins + 1)
    fins = [(-base_w / 2 + spacing * (i + 1) - fin_w / 2,
             -base_w / 2 + spacing * (i + 1) + fin_w / 2) for i in range(n_fins)]

    def solid(i, j):
        x = -base_w / 2 + (i + 0.5) * d_mm
        y = (j + 0.5) * d_mm
        if y <= base_h:
            return True
        if y <= base_h + fin_h:
            return any(a <= x <= b for a, b in fins)
        return False

    solid_grid = [[solid(i, j) for j in range(ny)] for i in range(nx)]
    T = [[T_src if j == 0 else 0.0 for j in range(ny)] for i in range(nx)]
    hc = h_conv * d_mm * 1e-3            # h·Δ  (k·Δ = 1 for unit conductivity)

    for _ in range(n_iter_max):
        max_delta = 0.0
        for j in range(1, ny):           # row 0 is the Dirichlet heat source
            for i in range(nx):
                if not solid_grid[i][j]:
                    continue
                acc = coef = 0.0
                base_wall = (j + 0.5) * d_mm <= base_h + 1e-12
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny and solid_grid[ii][jj]:
                        acc += T[ii][jj]
                        coef += 1.0
                    elif di != 0 and base_wall:
                        continue                 # insulated base side wall
                    else:
                        acc += hc * T_inf          # Robin face against air
                        coef += hc
                new = acc / coef
                max_delta = max(max_delta, abs(new - T[i][j]))
                T[i][j] = new
        if max_delta < tol:
            break

    def val(x_mm, y_mm):
        i = min(max(int((x_mm + base_w / 2) / d_mm), 0), nx - 1)
        j = min(max(int(y_mm / d_mm), 0), ny - 1)
        return T[i][j]

    return {
        "base_top": val(0.0, base_h - 0.5 * d_mm),
        "fin_root": val(0.0, base_h + 0.5 * d_mm),
        "fin_mid": val(0.0, 20.0),
        "fin_tip": val(0.0, base_h + fin_h - 0.5 * d_mm),
    }


def case6_fv_reference():
    ref = reference_fv(10.0, d_mm=1.0)
    assert 26.0 <= ref["fin_tip"] <= 38.0, f"C6: fin tip {ref['fin_tip']:.2f} ℃ outside [26,38]"
    assert ref["fin_root"] > ref["fin_mid"] > ref["fin_tip"], ref
    emul = reference_fv(10.0 / L0_SI, d_mm=1.0)
    assert emul["fin_tip"] < 1.0, f"C6: emulated pre-fix field should be cold, got {emul['fin_tip']}"
    print(f"C6 ok  FV(1 mm) h=10 → root/mid/tip = {ref['fin_root']:.2f}/{ref['fin_mid']:.2f}/"
          f"{ref['fin_tip']:.2f} ℃; pre-fix equivalent h_eff={10.0/L0_SI:.0f} → tip "
          f"{emul['fin_tip']:.2f} ℃")


def main() -> int:
    case1_legacy_bitwise_identical()
    case2_physical_formula()
    case3_mm_vs_m_guard()
    case4_fin_tip_1d()
    case5_sentinel_and_validation()
    if "--with-fv" in sys.argv:
        case6_fv_reference()
    else:
        print("C6 skip  (pass --with-fv to run the 1 mm finite-volume reference)")
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
