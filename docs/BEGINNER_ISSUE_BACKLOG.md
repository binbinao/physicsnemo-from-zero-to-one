# 小白读者体验 · Issue Backlog

> 由「小白视角审阅」整理，用于逐项改进教程与仓库。
> **第一轮 #1–#16**：已全部完成并关闭。  
> **第二轮 #17–#53**：见 [REVIEW_ROUND2_BACKLOG.md](REVIEW_ROUND2_BACKLOG.md)。  
> **第三轮 #166–#176（学习体验）**：见下表 · issue 正文 [issues/round3-ux/](issues/round3-ux/)。

**标签**：`beginner` · `documentation` · `onboarding` · `book` · `enhancement`

| # | 标题 | 状态 | GitHub |
|---|------|------|--------|
| 1 | 补全附录 A/B/C（正文多处引用但仓库缺失） | done | [#1](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/1) |
| 2 | 补全 `book/assets/` 配图（消除 IMAGE-TODO） | done | [#2](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/2) |
| 3 | 统一 Python/依赖版本说明（README vs ch00 vs check_env） | done | [#3](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/3) |
| 4 | 第 1 天「最小安装清单」与分章依赖表 | done | [#4](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/4) |
| 5 | 增强 `scripts/check_env.py`：按章节检查可选依赖 | done | [#5](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/5) |
| 6 | 每章添加 README：首选脚本 + 三档变体说明 | done | [#6](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/6) |
| 7 | 书—代码命令对齐审计（Hydra 参数、文件名） | done | [#7](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/7) |
| 8 | ch04：澄清翼型 vs Darcy、框架切换一页纸 | done | [#8](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/8) |
| 9 | ch05→ch04 代码依赖：跳章警告与依赖图 | done | [#9](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/9) |
| 10 | 可选「PyTorch 30 分钟最小集」或链到外部资源 | done | [#10](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/10) |
| 11 | 6 周学习计划 + 读者自测（选 🟢/🔵/🟡 路径） | done | [#11](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/11) |
| 12 | 云 GPU / Colab 实操指南（原附录 B 内容） | done | [#12](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/12) |
| 13 | 依赖版本锁定（requirements-lock / 测试通过 tag） | done | [#13](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/13) |
| 14 | 读完后路线图（官方 examples、真实数据、部署） | done | [#14](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/14) |
| 15 | 全书训练时长 / 硬件预期汇总表 | done | [#15](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/15) |
| 16 | 可复现 baseline（`results/` 或文档中的参考指标） | done | [#16](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/16) |

---

## Issue 详情（供实现时引用）

### 1 · 补全附录 A/B/C

**问题**：`book/ch00.md`、`ch01.md`、`ch07.md` 等引用「附录 A 数学速查」「附录 B 云 GPU」「附录 C 踩坑 50 问」，仓库中不存在。

**建议交付**：
- `book/appendix_a_math.md`
- `book/appendix_b_cloud_gpu.md`
- `book/appendix_c_troubleshooting.md`
- 在 `book/ch00.md` 与 ch07 结尾更新链接

**验收**：正文所有「见附录 X」链接可点击且文件存在。

---

### 2 · 补全 book/assets 配图

**问题**：各章 Markdown 引用 `assets/f*.png`，仓库无 `book/assets/`，文中有大量 `IMAGE-TODO`。

**建议交付**：优先 ch00–ch02 路线图/对比图；或提供 `book/scripts/generate_all_figures.py` 的可跑说明与占位图。

**验收**：`book/ch01.md` 快速通道所需图片在 GitHub 上可见。

---

### 3 · 统一 Python/依赖版本说明

**问题**：README「Python 3.9+」、`check_env` 要求 3.10+、ch00 推荐 3.11。

**建议交付**：单一事实来源（如 `docs/ENVIRONMENT.md`），README/ch00/check_env 对齐。

---

### 4 · 第 1 天最小安装清单

**问题**：ch01 快速通道建议装 sym，README 说 ch01–ch03 可只装 torch；小白不知第一天装什么。

**建议交付**：`docs/QUICKSTART_DAY1.md` — 仅 ch01 raw 的 pip 列表 + 验证命令。

---

### 5 · 增强 check_env.py

**问题**：全绿不等于能跑 ch07；缺少 Hydra/Optuna/FastAPI 分章提示。

**建议交付**：`python scripts/check_env.py --chapter 4` 或分 tier 输出。

---

### 6 · 每章 README

**问题**：ch02 正文写存在 `README.md` 但实际没有；各章缺「先跑哪个 .py」。

**建议交付**：`ch01_hello/README.md` … `ch07_.../README.md`，模板统一（目标 / 首选命令 / 三档表 / 预计时长）。

---

### 7 · 书—代码命令对齐

**问题**：如 ch04 正文 `dataset=darcy_mini` vs 脚本 `--epochs`。

**建议交付**：逐章 diff 正文命令与 argparse/Hydra；修正书或代码其一。

---

### 8 · ch04 翼型 vs Darcy 澄清

**问题**：目录 `fno_airfoil`，默认训练 Darcy；章标题强调翼型。

**建议交付**：ch04 开篇增加「本章两条路径」框图；README 与书名对齐说明。

---

### 9 · ch05→ch04 依赖文档化

**问题**：ch05 import ch04 的 `fno_model`，书中无醒目跳章警告。

**建议交付**：ch05 README + ch00 路线图标注；ch05 入口脚本友好错误提示。

---

### 10 · PyTorch 最小集（可选）

**问题**：ch00 承诺第 1 章教 PyTorch，实际偏应用。

**建议交付**：`book/appendix_d_pytorch_mini.md` 或链到官方 60min tutorial + 本书术语对照。

---

### 11 · 6 周学习计划 + 读者自测

**问题**：承诺 6 周但无周计划；🟢/🔵/🟡 路径无自测。

**建议交付**：`docs/STUDY_PLAN_6WEEKS.md` + 5 题自测表。

---

### 12 · 云 GPU / Colab 指南

**问题**：无 GPU 读者在 ch4+ 卡住；附录 B 缺失。

**建议交付**：Colab notebook 或逐步截图文档；国内云厂商对比表（附录 B 内容）。

---

### 13 · 依赖版本锁定

**问题**：PhysicsNeMo v2 API 变动；半年后 clone 可能失败。

**建议交付**：`requirements-minimal.txt` + `requirements-full.txt` + README 注明测试通过版本/commit。

---

### 14 · 读完后路线图

**问题**：ch07 指向附录后无「第 8 步」。

**建议交付**：`docs/WHATS_NEXT.md` — 官方 examples、真实数据集、Triton 部署链接。

---

### 15 · 训练时长 / 硬件汇总表

**问题**：各章散落「约 8 分钟」「建议 24GB」，无总表。

**建议交付**：`docs/HARDWARE_EXPECTATIONS.md` 一章一行的 CPU/GPU/显存/wall-clock。

---

### 16 · 可复现 baseline

**问题**：ch02 提到 `results/` 目录但仓库可能没有参考输出。

**建议交付**：文档记录 RTX 4070 上 loss 终值范围，或提供小型 `results/README.md` 说明如何对照。

---

## 建议处理顺序

1. **阻塞阅读**：#1 附录、#2 配图、#3 版本、#6 章 README  
2. **阻塞跑通**：#4 最小安装、#7 命令对齐、#9 ch05 依赖  
3. **降低放弃率**：#12 云 GPU、#15 硬件表、#5 check_env  
4. **长期质量**：#13 版本锁定、#16 baseline、#14 路线图、#10/#11 学习路径 — **已全部完成（16/16）**

---

## 第三轮 · 学习体验（#166–#176）

> 来源：初学者视角深度审阅（书稿 / 代码 / docs 支撑三线）。  
> issue map：[issues/round3-ux/issue_map.json](issues/round3-ux/issue_map.json)

| 本地 | 标题 | 优先级 | 状态 | GitHub |
|:---|:---|:---|:---|:---|
| R3-01 | check_env 补 scipy 自检（ch04/ch05） | P0 | 已实现 · 待关闭 | [#166](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/166) |
| R3-02 | Hydra/argparse 双语法提示 | P0 | 已实现 · 待关闭 | [#167](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/167) |
| R3-03 | 统一 ch01/ch07 README 与代码默认 epoch | P0 | 已实现（扩到 ch06）· 待关闭 | [#168](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/168) |
| R3-04 | 修复 BASELINE.md example_record 死链 | P0 | 已实现（换改法：修过期描述，非死链）· 待关闭 | [#169](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/169) |
| R3-05 | 补 ch02–ch07 example_record 与 BASELINE 数量级 | P1 | 转新需求（模板 + 征集真实记录）· 待关闭 | [#170](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/170) |
| R3-06 | 对齐 Colab notebook 与 6 周计划无 GPU 覆盖 | P1 | 已实现 · 待关闭 | [#171](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/171) |
| R3-07 | 拆分 6 周计划第 6 周过载 | P1 | 已实现 · 待关闭 | [#172](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/172) |
| R3-08 | 三档变体命名对照表 + raw/sdk 不可直接对比 | P1 | 已实现 · 待关闭 | [#173](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/173) |
| R3-09 | ch04 翼型/Darcy 路径 A/B 与示例对齐标注 | P2 | 已实现（仅补路径 B 标注）· 待关闭 | [#174](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/174) |
| R3-10 | ch03/ch06 可选深读 + DDP 无多卡可跳过 | P2 | 已实现 · 待关闭 | [#175](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/175) |
| R3-11 | 体例小错（ch07 编号 / SDK 口径 / 第0章 / 图号） | P2 | 已实现 · 待关闭 | [#176](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/176) |

**决策（实现时采用）**：

1. 三档变体：**保留现有文件名**，各章 README 增加对照表（不统一改名）。  
2. raw/sdk 物理设定不一致：**显式标注「两版问题不同，不可直接对比 loss」**（不对齐参数）。

**状态口径**：本表状态为**本轮实现后的真实状态**，不是 GitHub issue 的已关闭状态。  
issue 的实际关闭（`closes #N` 或手写理由）需要仓库写权限，由凭证持有方按《R3 逐项结案清单》逐条执行后再把本列改为 `已关闭`。

**本轮对 3 项的改法说明**：

1. **R3-04 / #169**：`results/BASELINE.md` 里指向 `results/ch01_hello/example_record/` 的链接**不是死链**（该目录自 `d0e56b3` 起即存在），真正过期的是「验证环境 commit `5de0cb7` 前后」「微缩默认 epoch」两处锚点，以及把示例记录称作「实测样例」。已按此口径修正，故 issue 以「措辞失效、真实问题已修」结案。
2. **R3-05 / #170**：PR #177 曾补 12 个 `results/ch02..ch07/*/example_record/` 数值文件，但文件头自述 `illustrative … (not a single locked seed)`、ch03 甚至写区间 `≈1e-03–5e-03`，属**示意值冒充实测**，不予入库；改为 `results/README.md` 的「提交真实记录的要求」模板 + 征集读者真实记录。
3. **R3-09 / #174**：main 的 `book/ch04.md` 章首路径表、`CH04_GUIDE` 与 `ch04_fno_airfoil/README.md` 已建立路径 A/B 口径，本轮只补 §4.6/§4.7 的「路径 B · 扩展」提示块（原 §4.7.1 把 `in_channels=4` 的翼型骨架写成「仓库可跑入口」，与默认 `train_fno_mini.py` 矛盾）。

**PR 处置**：PR #177 不直接 rebase 合入（其分支含 15.4 MB PDF、`.cursor/` 状态与 4 张零引用 PNG），本轮以 main 为基重新提交等效增量；PR #178 目标分支为 `cursor/setup-dev-environment-5075` 且无独有 R3 增量，不采用。
