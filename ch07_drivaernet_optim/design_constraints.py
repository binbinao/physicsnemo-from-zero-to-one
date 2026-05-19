"""
Engineering design constraints for toy car parameters (demo-level).
Use in optimize / optimize_multi / hifi export to filter infeasible designs.
"""

from __future__ import annotations

PARAM_NAMES = [
    "body_length",
    "body_width",
    "body_height",
    "front_angle",
    "rear_angle",
    "ground_clearance",
    "wheel_diameter",
]


def check_engineering_constraints(params: dict) -> tuple[bool, list[str]]:
    """Return (feasible, violation_messages)."""
    v: list[str] = []

    gc = params.get("ground_clearance", 0.2)
    if gc < 0.12:
        v.append(f"ground_clearance={gc:.3f} < 0.12 m (underbody risk)")

    bl = params.get("body_length", 4.0)
    if bl < 3.8:
        v.append(f"body_length={bl:.2f} < 3.8 m (packaging)")

    bh = params.get("body_height", 1.5)
    if bh > 1.75:
        v.append(f"body_height={bh:.2f} > 1.75 m (garage height)")

    fa = params.get("front_angle", 25.0)
    if fa > 38:
        v.append(f"front_angle={fa:.1f}° > 38° (visibility / stall risk)")

    frontal = params.get("body_width", 1.8) * bh
    if frontal > 3.2:
        v.append(f"frontal_proxy={frontal:.2f} > 3.2 m²")

    return len(v) == 0, v


def penalty_score(params: dict) -> float:
    """Soft penalty for Optuna (0 = OK)."""
    ok, msgs = check_engineering_constraints(params)
    if ok:
        return 0.0
    return 10.0 * len(msgs)
