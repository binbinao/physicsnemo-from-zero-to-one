#!/usr/bin/env python3
"""Create GitHub issues for review round 2. Run from repo root."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BODY_DIR = Path(__file__).resolve().parent / "bodies"
OUT_JSON = Path(__file__).resolve().parent / "issue_map.json"

ISSUES = [
    ("P0", "增加 START_HERE 中文入门单页", "README 与 book/ch00 未强绑定。交付：docs/START_HERE.md，README 顶部链接。"),
    ("P0", "ch00 路线图与 ch04 默认 Darcy 对齐", "§0.4 仍写 ch4 翼型/ch5 Darcy，与 train_fno_mini 默认不一致。改 ch00 文案 + 链 CH04_GUIDE。"),
    ("P0", "ch01 分路径 A/B 安装（torch vs SDK）", "快速通道仍 pip install sym，与 QUICKSTART_DAY1 矛盾。书内双栏路径。"),
    ("P0", "ch02 正文 Hydra 键名与 conf/config.yaml 对齐", "书中 loss_weights.lam_ic / n_interior；代码 w_ic / n_pde。同步 ch02 或加横幅。"),
    ("P0", "ENVIRONMENT：requirements-full 安装失败预期", "无 GPU/Mac/网络时 physicsnemo 易失败。文档说明可跳过 SDK。"),
    ("P0", "CI 冒烟：check_env + ch01 PINN", "无 .github/workflows。至少 PR/push 跑 check_env 与短 epoch ch01。"),
    ("P0", "文档注明 main 与 Colab clone 版本同步", "本地 commit 超前 origin 时读者 clone 旧版。README/START_HERE 写清。"),
    ("P1", "README 中英入口统一（中文读者指引）", "README 英文为主。增加中文快速链接块。"),
    ("P1", "Badge 与 ENVIRONMENT 版本对齐（PyTorch 2.3+）", "README badge 写 2.0+，与 requirements 2.3+ 不一致。"),
    ("P1", "ch00 磁盘空间说明改为分级", "30GB 吓退用户。区分微缩版 vs 完整数据集。"),
    ("P1", "ch00 PyTorch/CUDA 说明支持 CPU/Mac", "写死 CUDA 12.x。改为有 GPU 再装 CUDA 版。"),
    ("P1", "ch07 章首澄清：DrivAerNet toy vs 书名「自定义 PDE」", "叙事与 drivaernet_optim 目录对齐。"),
    ("P1", "统一 assets 目录说明（book/assets vs 根 assets）", "二维码与配图路径分散。book/assets/README 或 START_HERE 说明。"),
    ("P1", "附录 C 更新 Python 版本表述（去掉 3.9+）", "与 README 3.10+ 一致。"),
    ("P1", "Backlog/README 文案：16/16 done 后更新「正在改进」", "避免读者以为还有 open beginner issues。"),
    ("P1", "TESTED_ENVIRONMENT 更新验证 commit", "仍写 5de0cb7，需随发布更新。"),
    ("P1", "ch04 路径 B 命令与 ckpt 说明防误用", "visualize_airfoil + fno_darcy.pt 易混淆。COMMAND_REFERENCE/ch04 README。"),
    ("P1", "框架切换信息收敛到单一入口", "CH04_GUIDE + ch00 + 多章重复。START_HERE 或 FRAMEWORK_SWITCH.md。"),
    ("P1", "Colab 笔记本显式安装 torch 与版本提示", "依赖 Colab 预装。pip torch 行 + 链 TESTED_ENVIRONMENT。"),
    ("P1", "BASELINE 补充至少一章实测数值样例", "仅定性。results/ch01_*/ 或 BASELINE 表格填数。"),
    ("P2", "README 目录树补充 book/ docs/ notebooks/", "新人找不到教材。"),
    ("P2", "ch00 链接 STUDY_PLAN / QUICKSTART / COMMAND_REFERENCE", "读前言看不到学习计划。"),
    ("P2", "附录 A/B/C 页脚增加附录 D 链接", "附录导航不完整。"),
    ("P2", "修复 README Vibe Publishing 空锚点", "[story](#) 无效。补链接或删锚点。"),
    ("P2", "book/FIGURE_UPGRADE_SUMMARY 移入维护者区或加说明", "避免小白误点。"),
    ("P2", "章末微信公众号二维码说明（占位 vs 正式）", "wechat_qrcode.png 占位说明。"),
    ("P2", "ch01 Python 版本表述改为 3.10+（推荐 3.11）", "与全仓政策一致。"),
    ("P2", "补充 Windows/WSL 环境简短说明", "QUICKSTART 仅一句 Windows。"),
    ("P2", "补充 Apple Silicon / MPS 说明", "M 系列能否跑 ch4。"),
    ("P2", "docs/issues/*.md 历史模板加「已关闭」头", "避免显示附录 B 缺失等过时描述。"),
    ("P3", "可选 requirements-lock.txt 或 pip freeze 样例", "严格复现。#13 延伸。"),
    ("P3", "results/ 增加一条完整实测记录样例目录", "模板仅有说明无示例。"),
    ("P3", "SDK 版本矩阵或 compatibility 表", "physicsnemo API 漂移。"),
    ("P3", "每章 1 页 cheat sheet（快速通道摘要）", "单章过长。"),
    ("P3", "STUDY_PLAN 章节小测验与参考答案", "自学闭环。"),
    ("P3", "配图风格统一性评估与迭代", "matplotlib vs 示意图。"),
    ("P3", "WHATS_NEXT 增加能力矩阵自检表", "读完会什么/不会什么。"),
]

LABELS = "beginner,onboarding,enhancement,documentation"


def main() -> None:
    BODY_DIR.mkdir(parents=True, exist_ok=True)
    mapping: list[dict] = []
    for i, (prio, title, desc) in enumerate(ISSUES, start=1):
        body_path = BODY_DIR / f"r2-{i:02d}.md"
        body_path.write_text(
            f"## 背景\n\n来自小白视角第二轮审阅（2026-05）。\n\n## 问题\n\n{desc}\n\n"
            f"## 优先级\n\n**{prio}**\n\n## 验收\n\n- [ ] 交付物落地\n- [ ] 相关链接可点击\n- [ ] 与 ENVIRONMENT / COMMAND_REFERENCE 无矛盾\n",
            encoding="utf-8",
        )
        full_title = f"[Review-R2][{prio}] {title}"
        cmd = [
            "gh",
            "issue",
            "create",
            "--title",
            full_title,
            "--body-file",
            str(body_path),
            "--label",
            LABELS,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        url = out.stdout.strip()
        num = url.rsplit("/", 1)[-1]
        mapping.append({"local_id": i, "priority": prio, "title": title, "url": url, "number": int(num)})
        print(f"#{num} {prio} {title}")

    OUT_JSON.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_JSON} ({len(mapping)} issues)")


if __name__ == "__main__":
    main()
