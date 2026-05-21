#!/usr/bin/env python3
"""Run a single OpenFOAM case from the template directory.

Copies the template, substitutes placeholders, executes blockMesh + simpleFoam,
and extracts force coefficients into a JSON results file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "openfoam"

PLACEHOLDERS = {
    "{{INLET_VELOCITY}}": "inlet_velocity",
    "{{KINEMATIC_VISCOSITY}}": "viscosity",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instantiate and run an OpenFOAM case.")
    parser.add_argument("--case_dir", type=Path, required=True, help="Target case directory")
    parser.add_argument("--inlet_velocity", type=float, required=True, help="Inlet velocity magnitude [m/s]")
    parser.add_argument("--viscosity", type=float, required=True, help="Kinematic viscosity [m^2/s]")
    parser.add_argument("--dry_run", action="store_true", help="Prepare case without running solver")
    return parser.parse_args(argv)


def copy_template(case_dir: Path) -> None:
    """Copy template tree to case_dir, renaming .template files."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(TEMPLATE_DIR, case_dir)
    # Remove template README from the instantiated case
    readme = case_dir / "README.md"
    if readme.exists():
        readme.unlink()


def substitute_placeholders(case_dir: Path, values: dict[str, str]) -> list[Path]:
    """Replace placeholders in .template files and rename them."""
    processed: list[Path] = []
    for template_file in case_dir.rglob("*.template"):
        content = template_file.read_text()
        for placeholder, key in PLACEHOLDERS.items():
            content = content.replace(placeholder, values[key])
        target = template_file.with_suffix("")
        target.write_text(content)
        template_file.unlink()
        processed.append(target)
    return processed


def run_solver(case_dir: Path) -> subprocess.CompletedProcess:
    """Execute blockMesh and simpleFoam in the case directory."""
    print(f"[run_openfoam_case] Running blockMesh in {case_dir}")
    result = subprocess.run(
        ["blockMesh", "-case", str(case_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"[run_openfoam_case] Running simpleFoam in {case_dir}")
    result = subprocess.run(
        ["simpleFoam", "-case", str(case_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result


def extract_force_coefficients(case_dir: Path) -> dict[str, float]:
    """Parse Cd and Cl from postProcessing output."""
    coeff_file = case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    if not coeff_file.exists():
        return {"Cd": float("nan"), "Cl": float("nan")}

    last_line = ""
    for line in coeff_file.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            last_line = stripped

    if not last_line:
        return {"Cd": float("nan"), "Cl": float("nan")}

    # Typical format: Time Cd Cl ...
    parts = last_line.split()
    cd = float(parts[1]) if len(parts) > 1 else float("nan")
    cl = float(parts[2]) if len(parts) > 2 else float("nan")
    return {"Cd": cd, "Cl": cl}


def mock_results(inlet_velocity: float, viscosity: float) -> dict[str, float]:
    """Generate synthetic force coefficients for dry-run mode."""
    re = inlet_velocity * 1.0 / viscosity  # assume characteristic length = 1 m
    cd_mock = 0.3 + 1.0 / (re / 1e4 + 1.0)
    cl_mock = 0.01 * (inlet_velocity / 10.0)
    return {"Cd": round(cd_mock, 6), "Cl": round(cl_mock, 6)}


def write_results(case_dir: Path, results: dict) -> Path:
    """Write results JSON to case directory."""
    output_path = case_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    case_dir = args.case_dir.resolve()

    values = {
        "inlet_velocity": str(args.inlet_velocity),
        "viscosity": str(args.viscosity),
    }

    # Step 1: Copy template
    print(f"[run_openfoam_case] Copying template to {case_dir}")
    copy_template(case_dir)

    # Step 2: Substitute placeholders
    processed = substitute_placeholders(case_dir, values)
    print(f"[run_openfoam_case] Substituted {len(processed)} template file(s):")
    for p in processed:
        print(f"  - {p.relative_to(case_dir)}")

    # Step 3: Run solver or mock
    if args.dry_run:
        print("[run_openfoam_case] Dry-run mode — skipping solver execution")
        coeffs = mock_results(args.inlet_velocity, args.viscosity)
        print(f"[run_openfoam_case] Mock results: Cd={coeffs['Cd']}, Cl={coeffs['Cl']}")
    else:
        run_solver(case_dir)
        coeffs = extract_force_coefficients(case_dir)
        print(f"[run_openfoam_case] Results: Cd={coeffs['Cd']}, Cl={coeffs['Cl']}")

    # Step 4: Write JSON output
    results = {
        "case_dir": str(case_dir),
        "inlet_velocity": args.inlet_velocity,
        "viscosity": args.viscosity,
        "dry_run": args.dry_run,
        "Cd": coeffs["Cd"],
        "Cl": coeffs["Cl"],
    }
    output_path = write_results(case_dir, results)
    print(f"[run_openfoam_case] Results written to {output_path}")


if __name__ == "__main__":
    main()
