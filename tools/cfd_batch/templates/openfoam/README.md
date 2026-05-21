# OpenFOAM Template for CFD Batch Workflow

## Structure

```
templates/openfoam/
├── 0/
│   ├── U.template          # Velocity BCs — {{INLET_VELOCITY}} placeholder
│   └── p.template          # Pressure BCs
├── constant/
│   └── transportProperties.template  # {{KINEMATIC_VISCOSITY}} placeholder
├── system/
│   ├── controlDict         # Solver control (endTime, writeInterval)
│   ├── fvSchemes           # Discretization schemes (steady-state)
│   └── fvSolution          # Linear solver & SIMPLE settings
└── Allrun.template         # Shell script skeleton
```

## Parameterized Fields

| Placeholder              | File                        | Example Value |
|:-------------------------|:----------------------------|:--------------|
| `{{INLET_VELOCITY}}`    | `0/U.template`              | `10.0`        |
| `{{KINEMATIC_VISCOSITY}}`| `constant/transportProperties.template` | `1e-5` |

## Usage

Use `run_openfoam_case.py` to instantiate a case from this template:

```bash
python tools/cfd_batch/run_openfoam_case.py \
  --case_dir /tmp/my_case \
  --inlet_velocity 10.0 \
  --viscosity 1e-5 \
  --dry_run
```

The script copies the template, substitutes placeholders, and (unless `--dry_run`)
runs `blockMesh` followed by `simpleFoam`. Results are written to a JSON file.
