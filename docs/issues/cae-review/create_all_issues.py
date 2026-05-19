#!/usr/bin/env python3
"""Create CAE review issues #C1–#C35. Run from repo root."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cae_issue_map.json"
BODY_DIR = Path(__file__).resolve().parent / "bodies"
LABELS = "cae-review,documentation,enhancement"

ISSUES = [
    ("P0", "C1", "ch03 落地 Validator（参考解/残差报告）", "正文要求 validator，代码缺失。交付 run_validator + 训练后报告。"),
    ("P0", "C2", "反问题：书—码对齐（非一次训练联合反演）", "书称一次训练反演；代码为 brute-force 重训。改文或改实现。"),
    ("P0", "C3", "ch04 行业节按 Darcy / 翼型分叉叙述", "默认 Darcy 与航空 RANS 叙事分离。"),
    ("P0", "C4", "ch07 高保真复核闭环（脚本/文档）", "optimize 后 Top-K CFD 复核工作流骨架。"),
    ("P0", "C5", "ch07 数据档位诚实标注（toy vs DrivAerNet）", "章首/README 明确 toy 合成数据。"),
    ("P0", "C6", "ch00 弱化「无网格/直接喂几何」表述", "强调离散表征与 CFD 网格数据来源。"),
    ("P1", "C7", "CAE 单位与无量纲化指南", "docs/CAE_UNITS_AND_NONDIM.md"),
    ("P1", "C8", "瞬态 vs 稳态：ch02–ch03 衔接说明", "何时需要瞬态 Icepak 模型。"),
    ("P1", "C9", "ch03 明确 2D 微缩 vs 3D Icepak", "几何维度与对标范围。"),
    ("P1", "C10", "ch05 FD 物理损失 vs FEM 通量形式说明", "守恒格式差异与局限。"),
    ("P1", "C11", "ch04 湍流/RANS 标签与训练数据口径", "AirfRANS/RANS 设置说明。"),
    ("P1", "C12", "CAE 守恒/残差检查清单", "docs/CAE_CONSERVATION_CHECKLIST.md"),
    ("P1", "C13", "ch01 行业表：FNO 扫参表述时机", "避免未学 FNO 即出现扫参结论。"),
    ("P1", "C14", "ch00 本书物理场覆盖范围说明", "NVH/热/CFD 边界。"),
    ("P1", "C15", "CAE 训练数据生成 SOP 骨架", "mesh/solver/deck/版本记录。"),
    ("P1", "C16", "网格收敛对代理模型影响（文档）", "写入 DATA SOP。"),
    ("P1", "C17", "多保真概念与本书边界（文档）", "无实现则标明。"),
    ("P1", "C18", "ch07 OOD/设计空间包络检查", "optimize 前参数范围校验。"),
    ("P1", "C19", "toy→真实数据迁移检查表", "域适应要点。"),
    ("P1", "C20", "ch07 checkpoint 数据/版本元数据", "git hash、solver、数据路径。"),
    ("P2", "C21", "ch07 优化约束扩展说明（工程约束）", "文档列真实约束类型。"),
    ("P2", "C22", "多目标气动优化（文档延伸）", "Cd 以外指标。"),
    ("P2", "C23", "部署生产化差距说明（Triton/API）", "SLA/多租户。"),
    ("P2", "C24", "ONNX 与 CAE 求解器集成说明", "消费侧集成路径。"),
    ("P2", "C25", "工业级/API 变动风险说明", "v2 维护策略。"),
    ("P2", "C26", "ch03 TensorBoard/Monitor 书码对齐", "有则链文档，无则改书。"),
    ("P2", "C27", "ch04 调参图「待替换」注释清理", "配图说明与实测状态。"),
    ("P2", "C28", "ch05 PhysicsInformer 与仓库代码对照注", "import 路径以代码为准。"),
    ("P2", "C29", "硬件表：3D 级 PINN 资源免责声明", "HARDWARE_EXPECTATIONS 补充。"),
    ("P2", "C30", "许可证合规说明", "docs/LICENSE_NOTES.md"),
    ("P3", "C31", "START_HERE_CAE 导读", "资深 CAE 阅读路径。"),
    ("P3", "C32", "V&V 报告模板（可填）", "docs/VV_REPORT_TEMPLATE.md"),
    ("P3", "C33", "参考解对照实验设计（文档）", "FEM/CFD vs 代理。"),
    ("P3", "C34", "PINN/FNO/传统 CFD 决策树", "docs/CAE_SURROGATE_DECISION_TREE.md"),
    ("P3", "C35", "CAE 延伸阅读（V&V40/NAFEMS）", "WHATS_NEXT 或 CAE 导读链接。"),
]


def main() -> None:
    BODY_DIR.mkdir(parents=True, exist_ok=True)
    mapping = []
    for prio, cid, title, desc in ISSUES:
        i = len(mapping) + 1
        body_path = BODY_DIR / f"cae-{i:02d}.md"
        body_path.write_text(
            f"## 背景\n\n资深 CAE 仿真工程师视角审阅（2026-05）。编号 **{cid}**。\n\n"
            f"## 问题\n\n{desc}\n\n## 优先级\n\n**{prio}**\n\n"
            f"## 验收\n\n- [ ] 书—码—文档一致\n- [ ] 不夸大签审能力\n",
            encoding="utf-8",
        )
        full = f"[CAE][{prio}] {cid} · {title}"
        out = subprocess.run(
            ["gh", "issue", "create", "--title", full, "--body-file", str(body_path), "--label", LABELS],
            capture_output=True,
            text=True,
            check=True,
        )
        url = out.stdout.strip()
        num = int(url.rsplit("/", 1)[-1])
        mapping.append({"id": cid, "priority": prio, "title": title, "number": num, "url": url})
        print(f"#{num} {prio} {cid} {title}")

    OUT.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
