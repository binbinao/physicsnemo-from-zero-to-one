#!/usr/bin/env python3
"""
第 1 章 · SDK 参考版：用 physicsnemo-sym 声明式 API 求解弹簧振子

  python pinn_spring_sdk.py

⚠️ 需要安装 nvidia-physicsnemo 和 nvidia-physicsnemo.sym
   如果没有安装，请先用 pinn_spring.py（裸 PyTorch 版）

本文件展示 PhysicsNeMo-Sym 的声明式风格：
  - 用 SymPy 定义 PDE
  - 用 1D 几何定义时间域
  - 用 Constraint 定义损失三件套
  - Solver 替你管训练循环
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


# ── 自定义 PDE：弹簧振子 m·ẍ + k·x = 0 ──────────────────
class SpringPDE(PDE):
    """简谐振动 PDE：m * d²x/dt² + k * x = 0"""

    name = "SpringODE"

    def __init__(self, m=1.0, k=4.0):
        t = sp.Symbol("t")
        x = sp.Function("x")(t)
        self.equations = {"spring_ode": m * x.diff(t, 2) + k * x}


def run_sdk():
    """SDK 版训练（需要 Hydra 配置文件 conf/config.yaml）"""

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def _main(cfg: DictConfig) -> None:
        # 1) 几何：1D 时间域 t ∈ [0, 10]
        geo = Line1D(0.0, 10.0)

        # 2) 网络
        net = instantiate_arch(
            input_keys=[Key("t")],
            output_keys=[Key("x")],
            cfg=cfg.arch.fully_connected,
        )

        # 3) PDE 节点
        spring_eq = SpringPDE(m=1.0, k=4.0)
        nodes = spring_eq.make_nodes() + [net.make_node(name="spring_net")]

        # 4) Domain + 约束
        domain = Domain()

        # PDE 内部约束
        interior = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"spring_ode": 0},
            batch_size=cfg.batch_size.interior,
        )
        domain.add_constraint(interior, "interior")

        # 初始位置：x(0) = 1.0
        ic_pos = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"x": 1.0},
            batch_size=1,
            criteria=sp.Eq(sp.Symbol("t"), 0),
            lambda_weighting={"x": 100.0},
        )
        domain.add_constraint(ic_pos, "ic_pos")

        # 初始速度：dx/dt|_{t=0} = 0
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
        print("⚠️  PhysicsNeMo-Sym 未安装")
        print()
        print("本文件需要 nvidia-physicsnemo 和 nvidia-physicsnemo.sym")
        print("如果你只想跑通第 1 章，请使用裸 PyTorch 版：")
        print()
        print("    python pinn_spring.py --epochs 5000")
        print()
        print("SDK 安装方式：")
        print("    pip install nvidia-physicsnemo nvidia-physicsnemo.sym")
        print("=" * 60)
        return

    run_sdk()


if __name__ == "__main__":
    main()
