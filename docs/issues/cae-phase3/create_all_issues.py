#!/usr/bin/env python3
"""CAE phase-3: items previously deferred. Creates GitHub issues."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "phase3_issue_map.json"
BODY_DIR = Path(__file__).resolve().parent / "bodies"
LABELS = "cae-review,enhancement,documentation"

ISSUES = [
    ("P0", "C36", "CFD 批跑工作流骨架（deck/清单/对接 hifi_queue）",
     "Icepak/Fluent 自动批跑模板、manifest、从 hifi_queue.csv 导入；dry-run 默认。"),
    ("P0", "C37", "ch03 联合反演 PINN（单训练学习 fin_height）",
     "新增 heat_sink_inverse_joint.py，与 brute-force 版并存。"),
    ("P1", "C38", "ch07 多目标 Optuna 示例（Cd + 迎风面积代理）",
     "optimize_multi.py + 文档与 COMMAND_REFERENCE。"),
    ("P1", "C39", "CFD 批跑结果回灌与 V&V 字段扩展",
     "results/cfd_runs 模板 + VV_REPORT 复核段 + ingest_results.py 骨架。"),
    ("P2", "C40", "教材与 START_HERE_CAE 更新 phase-3 能力",
     "ch03/ch07 正文链到新脚本；CAE_REVIEW_BACKLOG phase-3 表。"),
]

def main() -> None:
    BODY_DIR.mkdir(parents=True, exist_ok=True)
    mapping = []
    for i, (prio, cid, title, desc) in enumerate(ISSUES, start=1):
        bp = BODY_DIR / f"p3-{i:02d}.md"
        bp.write_text(
            f"## 背景\n\nCAE phase-3（原「刻意未做」项）。**{cid}**\n\n## 问题\n\n{desc}\n\n## 优先级\n\n{prio}\n",
            encoding="utf-8",
        )
        full = f"[CAE-P3][{prio}] {cid} · {title}"
        url = subprocess.run(
            ["gh", "issue", "create", "--title", full, "--body-file", str(bp), "--label", LABELS],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        num = int(url.rsplit("/", 1)[-1])
        mapping.append({"id": cid, "priority": prio, "number": num, "url": url, "title": title})
        print(f"#{num} {prio} {cid}")
    OUT.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
