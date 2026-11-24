#!/usr/bin/env python3
"""Round 3 (learner UX): create GitHub issues from bodies/."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "issue_map.json"
BODY_DIR = Path(__file__).resolve().parent / "bodies"
LABELS = "beginner,documentation,onboarding"

ISSUES = [
    ("P0", "R3-01", "check_env 补 scipy 自检（ch04/ch05）",
     "check_env --chapter 4 显示 ✅ 但 dataset.py 缺 scipy 会 ImportError；"
     "在 run_base_checks 与 ch04/ch05 required 中加入 scipy。"),
    ("P0", "R3-02", "Hydra/argparse 双语法提示（README + COMMAND_REFERENCE + 入口脚本）",
     "装 hydra-core 后 --epochs/--steps 对 ch02–ch06 多个入口失效；"
     "文档明确两种语法，Hydra 入口启动时打印一行提示。"),
    ("P0", "R3-03", "统一 ch01/ch07 README 与代码默认 epoch",
     "ch01 README 2000 vs 代码 5000；ch07 README 100 vs 代码 200；对齐为一致值。"),
    ("P0", "R3-04", "修复 BASELINE.md example_record 死链",
     "results/BASELINE.md 内链应为 results/ch01_hello/example_record/。"),
    ("P1", "R3-05", "补 ch02–ch07 example_record 与 BASELINE 数量级",
     "为各章补 results/chXX_*/example_record/final_losses.txt，并更新 BASELINE。"),
    ("P1", "R3-06", "对齐 Colab notebook 与 6 周计划无 GPU 覆盖",
     "colab_quickstart 仅 ch01/ch04；STUDY_PLAN W3 写 Colab 跑 ch03。"
     "补 notebook 或改文档口径。"),
    ("P1", "R3-07", "拆分 6 周计划第 6 周过载（ch06 / ch07）",
     "W6 硬塞 ch06+ch07+附录；拆成两周或标注零基础默认 8–10 周。"),
    ("P1", "R3-08", "三档变体命名对照表 + raw/sdk 不可直接对比说明",
     "保留文件名，各章 README 补三档对照；显式标注 ch02/ch03 raw 与 sdk 物理设定不同。"),
    ("P2", "R3-09", "ch04 翼型/Darcy 路径 A/B 与 §4.6–4.7 示例对齐标注",
     "统一路径 A=Darcy 默认 / 路径 B=翼型扩展；正文示例标明路径。"),
    ("P2", "R3-10", "ch03/ch06 可选深读标注 + DDP 无多卡可跳过",
     "章首路线图标可选深读；ch06 §6.7 前提醒无多卡可跳过。"),
    ("P2", "R3-11", "体例小错：ch07 编号、SDK 口径、第0章、图号",
     "修复 §7.6.1 重复、ch01「ch02起只用SDK」与 ch02 推荐 raw 冲突、"
     "「第 0 章」表述、页脚图数。"),
]


def main() -> None:
    BODY_DIR.mkdir(parents=True, exist_ok=True)
    mapping = []
    for i, (prio, cid, title, desc) in enumerate(ISSUES, start=1):
        bp = BODY_DIR / f"r3-{i:02d}.md"
        bp.write_text(
            f"## 背景\n\n第三轮 · 学习体验整改（初学者视角审阅）。**{cid}**\n\n"
            f"## 问题\n\n{desc}\n\n## 优先级\n\n{prio}\n\n"
            f"## 验收\n\n见 `docs/BEGINNER_ISSUE_BACKLOG.md` 第三轮条目。\n",
            encoding="utf-8",
        )
        full = f"[Beginner][R3][{prio}] {cid} · {title}"
        url = subprocess.run(
            ["gh", "issue", "create", "--title", full, "--body-file", str(bp), "--label", LABELS],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        num = int(url.rsplit("/", 1)[-1])
        mapping.append(
            {"id": cid, "priority": prio, "number": num, "url": url, "title": title}
        )
        print(f"#{num} {prio} {cid}")
    OUT.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
