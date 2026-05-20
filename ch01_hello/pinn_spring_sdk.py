#!/usr/bin/env python3
"""
Chapter 1 - SDK Reference: Declarative PhysicsNeMo-Sym API for Spring Oscillator

  python pinn_spring_sdk.py

Requires nvidia-physicsnemo and nvidia-physicsnemo.sym.
If not installed, use pinn_spring.py (raw PyTorch version) instead.

Demonstrates PhysicsNeMo-Sym declarative style:
  - Define PDE with SymPy
  - Define temporal domain with 1D geometry
  - Define loss terms with Constraints
  - Solver manages training loop
"""

try:
    import sympy as sp
    import hydra
    from omegaconf import DictConfig
    from physicsnemo.sym.eq.pde import PDE
    from physicsnemo.sym.solver import Solver
    from physicsnemo.sym.domain import Domain
    from physicsnemo.sym.domain.constraint import (
        PointwiseBoundaryConstraint,
        PointwiseInteriorConstraint,
    )
    from physicsnemo.sym.geometry.primitives_1d import Line1D
    from physicsnemo.sym.key import Key
    from physicsnemo.sym.hydra import instantiate_arch

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# ── Custom PDE: spring oscillator m*x'' + k*x = 0 ────────
class SpringPDE(PDE):
    """Simple harmonic oscillator PDE: m * d^2x/dt^2 + k * x = 0"""

    name = "SpringODE"

    def __init__(self, m=1.0, k=4.0):
        t = sp.Symbol("t")
        x = sp.Function("x")(t)
        self.equations = {"spring_ode": m * x.diff(t, 2) + k * x}


def run_sdk():
    """SDK training (requires Hydra config file conf/config.yaml)"""

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def _main(cfg: DictConfig) -> None:
        # 1) Geometry: 1D time domain t in [0, 10]
        geo = Line1D(0.0, 10.0)

        # 2) Network
        net = instantiate_arch(
            input_keys=[Key("t")],
            output_keys=[Key("x")],
            cfg=cfg.arch.fully_connected,
        )

        # 3) PDE nodes
        spring_eq = SpringPDE(m=1.0, k=4.0)
        nodes = spring_eq.make_nodes() + [net.make_node(name="spring_net")]

        # 4) Domain + constraints
        domain = Domain()

        # PDE interior constraint
        interior = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"spring_ode": 0},
            batch_size=cfg.batch_size.interior,
        )
        domain.add_constraint(interior, "interior")

        # Initial position: x(0) = 1.0
        ic_pos = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"x": 1.0},
            batch_size=1,
            criteria=sp.Eq(sp.Symbol("t"), 0),
            lambda_weighting={"x": 100.0},
        )
        domain.add_constraint(ic_pos, "ic_pos")

        # Initial velocity: dx/dt|_{t=0} = 0
        ic_vel = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"x__t": 0.0},
            batch_size=1,
            criteria=sp.Eq(sp.Symbol("t"), 0),
            lambda_weighting={"x__t": 100.0},
        )
        domain.add_constraint(ic_vel, "ic_vel")

        # 5) Solver
        solver = Solver(cfg, domain)
        solver.solve()

    _main()


def main():
    if not HAS_SDK:
        print("=" * 60)
        print("WARNING: PhysicsNeMo-Sym is not installed")
        print()
        print("This file requires nvidia-physicsnemo and nvidia-physicsnemo.sym")
        print("To run Chapter 1 without SDK, use the raw PyTorch version:")
        print()
        print("    python pinn_spring.py --epochs 5000")
        print()
        print("To install the SDK:")
        print("    pip install nvidia-physicsnemo nvidia-physicsnemo.sym")
        print("=" * 60)
        return

    run_sdk()


if __name__ == "__main__":
    main()
