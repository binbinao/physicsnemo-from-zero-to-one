#!/usr/bin/env python3
"""Regression tests for the ch03 validator *gates* (physics vs residuals).

Plain ``assert`` + stdlib + torch only (no pytest dependency):

    cd ch03_heatsink && python3 tests/test_validator_gates.py

Cases
-----
V1  probe coordinates are derived from the geometry (root/mid/tip follow
    ``fin_height``; axis follows the middle fin) instead of being hard-coded.
V2  a constant field ``T ≡ T_inf`` has **zero** PDE and **zero** Robin residual
    but is not physical: residual gates pass, ``pass_physical`` is false, and
    ``pass_all_relative`` must therefore be false (residual gate is necessary,
    not sufficient).
V3  report key contract: legacy field names/order preserved, ``pass_all`` still
    immediately after ``note``, new fields appended after it.
V4  ``fin_tip_1d_from_root`` is driven by the *probe span*
    (``fin_height - 2·inset``), not the nominal fin height (bias ≤ 2.5 ℃).
V5  ``L0_SI`` is derived from the coordinate normalisation constant
    (``COORD_SCALE_MM/1000``) — the two can no longer drift apart.
"""

import os
import sys

_CH03_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CH03_DIR not in sys.path:
    sys.path.insert(0, _CH03_DIR)

import torch  # noqa: E402

from equations import COORD_SCALE_MM, L0_SI  # noqa: E402
from heat_sink_geometry import HeatSinkGeometry  # noqa: E402
from validator import fin_probe_points, run_validation, fin_tip_1d_from_root  # noqa: E402

CFG = {
    "k": 1.0, "Q": 0.0, "T_source": 100.0, "h_conv": 10.0, "T_inf": 0.0,
    "scale": COORD_SCALE_MM, "robin_scale": L0_SI,
    "fin_height": 20.0, "fin_width": 4.0, "n_fins": 3,
}


class ConstantField(torch.nn.Module):
    """Tricky non-solution: T ≡ T_inf (zero residual everywhere, not a fin)."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = float(value)

    def forward(self, x, y):  # noqa: D401 - nn.Module API
        # constant field, but second-order differentiable (validation takes
        # ∂²T/∂x²), so keep quadratic terms with zero coefficient in the graph
        return 0.0 * x ** 2 + 0.0 * y ** 2 + self.value


# ---------------------------------------------------------------------------
# V1 — probes follow the geometry
# ---------------------------------------------------------------------------
def case1_probes_from_geometry():
    p = fin_probe_points(HeatSinkGeometry())
    assert p["T_fin_root"] == (0.0, 10.5), p
    assert p["T_fin_mid"] == (0.0, 20.0), p
    assert p["T_fin_tip"] == (0.0, 29.5), p

    tall = fin_probe_points(HeatSinkGeometry(fin_height=40.0))
    assert tall["T_fin_mid"] == (0.0, 30.0), tall      # mid-fin, not 20
    assert tall["T_fin_tip"] == (0.0, 49.5), tall      # real tip, not 29.5
    assert tall["T_fin_root"] == (0.0, 10.5), tall

    two = fin_probe_points(HeatSinkGeometry(n_fins=2))
    assert abs(two["T_fin_tip"][0]) == 10.0, two       # fin centres at ±10 mm
    print("V1 ok  probes derived from geometry (fin_height=40 → mid 30 / tip 49.5)")


# ---------------------------------------------------------------------------
# V2 — residual gate is necessary but not sufficient
# ---------------------------------------------------------------------------
def case2_constant_field_fails_physical_gate():
    rep = run_validation(ConstantField(0.0), HeatSinkGeometry(), dict(CFG),
                         n_int=200, n_bc=100)
    assert rep["pde_residual_rms"] == 0.0, rep["pde_residual_rms"]
    assert rep["robin_residual_rms"] == 0.0, rep["robin_residual_rms"]
    assert rep["pass_pde_relative"] is True and rep["pass_robin_relative"] is True
    assert rep["pass_temperature_range"] is True
    assert rep["pass_tip_band"] is False, rep
    assert rep["pass_fin_profile"] is False, rep
    # NOTE: the 1D self-consistency check is *degenerate* at T_root = T_inf
    # (both sides are 0), which is exactly why the tip band and the monotonic
    # profile are needed as well.
    assert rep["pass_1d_consistency"] is True
    assert rep["pass_physical"] is False, rep
    assert rep["pass_all_relative"] is False, "a T≡T_inf field must not pass the gate"
    # The legacy gate (unchanged semantics) *is* satisfied by this trivial
    # field -- documented in EXPERIMENTS_PASS §10.2, deliberately not changed.
    assert rep["pass_all"] is True
    print("V2 ok  T≡T_inf: pde/robin residual 0 → pass_physical false, "
          "pass_all_relative false (legacy pass_all still true, semantics kept)")


# ---------------------------------------------------------------------------
# V3 — report key contract
# ---------------------------------------------------------------------------
def case3_key_contract():
    legacy = ["pde_residual_rms", "robin_residual_rms", "T_min", "T_max",
              "T_source_config", "fin_tip_1d_bound", "pass_pde", "pass_robin",
              "pass_temperature_range", "note", "pass_all"]
    rep = run_validation(ConstantField(0.0), HeatSinkGeometry(), dict(CFG),
                         n_int=200, n_bc=100)
    keys = list(rep)
    assert keys[:len(legacy)] == legacy, keys[:len(legacy)]
    for k in ("pass_physical", "pass_tip_band", "pass_fin_profile",
              "pass_1d_consistency", "fin_probe_points_mm", "fin_probe_span_mm"):
        assert k in rep, k
    assert isinstance(rep["pass_physical"], bool)
    assert isinstance(rep["fin_probe_span_mm"], float)
    print("V3 ok  legacy key order preserved (pass_all after note), new keys appended")


# ---------------------------------------------------------------------------
# V4 — 1D reference length = probe span
# ---------------------------------------------------------------------------
def case4_one_d_reference_uses_span():
    rep = run_validation(ConstantField(0.0), HeatSinkGeometry(), dict(CFG),
                         n_int=200, n_bc=100)
    span = rep["fin_probe_span_mm"]
    assert span == 19.0, span
    expect = fin_tip_1d_from_root(rep["T_fin_root"], 0.0, 10.0, 1.0, span, 4.0)
    assert abs(rep["fin_tip_1d_from_root"] - expect) < 1e-12
    # and the bias vs. the nominal fin height is the documented ~2.4 ℃
    nominal = fin_tip_1d_from_root(72.0, 0.0, 10.0, 1.0, 20.0, 4.0)
    span_v = fin_tip_1d_from_root(72.0, 0.0, 10.0, 1.0, span, 4.0)
    assert 0.0 < span_v - nominal < 3.0, (nominal, span_v)
    print(f"V4 ok  1D reference uses span {span} mm "
          f"(nominal-height bias {span_v - nominal:.2f} ℃)")


# ---------------------------------------------------------------------------
# V5 — L0_SI derives from the coordinate normalisation
# ---------------------------------------------------------------------------
def case5_l0_invariant():
    assert L0_SI == COORD_SCALE_MM / 1000.0
    assert L0_SI == 0.03, L0_SI          # exact float equality with the documented value
    print("V5 ok  L0_SI == COORD_SCALE_MM/1000 == 0.03 (single source of truth)")


def main() -> int:
    case1_probes_from_geometry()
    case2_constant_field_fails_physical_gate()
    case3_key_contract()
    case4_one_d_reference_uses_span()
    case5_l0_invariant()
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
