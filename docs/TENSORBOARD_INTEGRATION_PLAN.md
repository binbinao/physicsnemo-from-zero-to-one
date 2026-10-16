# 全书默认接入 tensorboardX 训练曲线 · 实施方案

> 目标：训练脚本**默认**记录曲线到 TensorBoard 事件文件，训练中/训练后都能查看。
> 状态：**✅ 已按 4 批实施完毕（2026-09-04）**，ch01–ch07 共 25 个训练脚本全部接入。
> 交付物：`scripts/tb_logger.py`、`scripts/plot_tb_curves.py`；backlog 登记见 `docs/REVIEW_ROUND2_BACKLOG.md` #54（done）。

---

## 1. 现状调研结论

| 项 | 现状 |
|:---|:---|
| tensorboard / tensorboardX 使用 | **26 个训练脚本，0 处使用**，`SummaryWriter` / `add_scalar` 全仓无命中 |
| requirements | `requirements-minimal.txt` 与 `requirements-full.txt` **均不含** tensorboard / tensorboardX |
| 共用工具模块 | **无**（无 `utils.py` / `common.py`）；`scripts/` 下只有环境检查与配图脚本 |
| 跨章 import | 仅 ch05 → ch04（由 `require_ch04.py` 守卫），其余各章零跨章依赖 |
| 指标存放 | 多数为 `history = {...}`（dict of list）；**4 个脚本完全不保存历史**：`ch07/train.py`、`ch07/train_gpu.py`、`ch02/convergence_study.py`、`ch03/heat_sink_inverse*.py` |
| 日志方式 | **全部是 `print`**，无脚本写 csv / json 曲线 |
| 输出目录 | **3 种互不一致的约定**：Hydra 时间戳目录 / SDK·GPU 档用脚本同级 `outputs`、`outputs_gpu` / 早期脚本硬编码相对 `outputs/`（随 CWD 跑） |
| 学习率 | 几乎不记录，需在接入时从 `optimizer.param_groups[0]["lr"]` 现取 |
| checkpoint | 22/26 有；`mlp_spring.py`、`pinn_spring.py`、`convergence_study.py`、`heat_sink_inverse.py` 无 |
| **书码不一致** | `book/ch03.md:504-516`（含 `tensorboard --logdir outputs/three_fin_2d`）与 `book/ch02.md:442,468` 都声称代码支持 TensorBoard，但代码不产出任何事件文件。该缺口已在 `docs/issues/cae-review/` 记为 **C26** |

已在本机环境验证：`tensorboardX 2.6.5` 安装成功，可正常写出 `events.out.tfevents.*`。

---

## 2. 目标 / 非目标

**目标**

1. 所有训练脚本默认记录 loss 等标量曲线（TensorBoard 事件文件）。
2. 训练**中**可实时查看，训练**后**可离线复现曲线。
3. 没装任何可视化工具也能看图（CSV + matplotlib 兜底）。
4. 一条命令关掉（`--no-tb`），不影响现有 `print` 输出与 PNG 产物。

**非目标**

- 不改动训练逻辑、不改变默认超参与随机种子，训练结果保持可复现。
- 不引入重量级依赖（不强制 `tensorflow`；`tensorboard` CLI 仅为可选）。
- 不重写各章 README，只增补「查看曲线」小节。

---

## 3. 设计方案

### 3.1 新增共用记录器 `scripts/tb_logger.py`

```python
class TBLogger:
    def __init__(self, log_dir, enabled=True, timestamp=True): ...
    @classmethod
    def from_args(cls, args, script_stem, default_root=None): ...   # 统一读 --tb-dir / --no-tb
    def log(self, step, metrics: dict) -> None: ...                 # 写 TB + CSV
    def log_figure(self, tag, fig, step) -> None: ...               # 可选，复用已有 matplotlib 图
    def close(self) -> None: ...
```

关键行为：

- **惰性导入 tensorboardX**；导入失败时打印一次 warning 并降级为**只写 CSV**，训练不中断。
- **双写**：`<log_dir>/events.out.tfevents.*`（TensorBoard）+ `<log_dir>/metrics.csv`（列：`step` + 各指标），保证零依赖也能看图。
- 关闭开关：CLI `--no-tb`，环境变量 `PN_TB=0`；根目录可用 `PN_TB_DIR` 覆写。
- 目录约定（覆盖现有 3 种）：

  | 脚本类型 | 默认日志目录 |
  |:---|:---|
  | Hydra 脚本 | `${hydra:runtime.output_dir}/tb`（Hydra 自身已按时间戳隔离） |
  | 有 `--out_dir` / `--output_dir` 的脚本 | `<out_dir>/tb/<YYYYmmdd-HHMMSS>` |
  | 其余脚本 | `<repo_root>/outputs/tb/<script_stem>/<YYYYmmdd-HHMMSS>` |

  DDP 脚本（`*_gpu.py`）仅 rank 0 写日志，避免多进程写同一目录。

  另提供 `--tb-dir PATH` 显式指定、`--tb-tag NAME` 自定义运行名、`--tb-no-timestamp` 关闭时间戳子目录。

- 引用方式沿用 ch05 的先例（插入仓库路径后 import），不新增包结构：

  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  from tb_logger import TBLogger
  ```

### 3.2 新增曲线查看脚本 `scripts/plot_tb_curves.py`

- 默认读 CSV，**零额外依赖**：
  `python scripts/plot_tb_curves.py --logdir outputs/tb --out curves.png`
- 自动发现目录下所有 `metrics.csv` / 运行子目录，一张图一个指标，支持 `--metrics loss/total,val/mae` 与 `--smooth 0.6`（指数平滑）。
- 可选 `--from-tb`：直接读 tfevents（需已装 `tensorboard`），作为 CSV 缺失时的补充。

### 3.3 训练脚本接入范式（每处约 5–10 行）

```python
parser.add_argument("--tb-dir", default=None, help="TensorBoard 日志目录（默认见 docs）")
parser.add_argument("--no-tb", action="store_true", help="关闭 TensorBoard 记录")

tb = TBLogger.from_args(args, script_stem=Path(__file__).stem)
...
for epoch in range(epochs):
    ...
    tb.log(epoch, {"loss/total": loss.item(), "loss/pde": l_pde.item(),
                   "lr": optimizer.param_groups[0]["lr"]})
tb.close()
print(f"TensorBoard: {tb.log_dir}")     # 结尾打印实际目录
```

记录内容建议：`loss/*` 各分量、`loss/total`、验证指标（`val/mae` 等）、`lr`；GPU 档额外记录 `perf/iter_per_sec`、`mem/gb`。

### 3.4 依赖变更

| 文件 | 变更 |
|:---|:---|
| `requirements-minimal.txt` | 追加 `tensorboardX>=2.6`（纯 Python，约 1 MB；默认开启必须） |
| `requirements-full.txt` | 追加 `tensorboard>=2.16`（**可选**，仅用于 `tensorboard --logdir` CLI 查看） |
| `scripts/check_env.py` | 新增一行自检：「TensorBoard 记录（可选）」，检查 `tensorboardX`；`tensorboard` 缺失仅提示不判失败 |

### 3.5 文档更新

- `docs/ENVIRONMENT.md`：新增「训练曲线与 TensorBoard」小节（默认开启、如何关闭、如何查看）。
- 各章 `README.md`：补「查看训练曲线」2–3 行（含示例命令）。
- **书稿对齐（需确认后再动）**：`book/ch02.md:442,468`、`book/ch03.md:504-516` 目前声称有 TensorBoard 而代码没有；实施后这些描述即变为真实，可一并关闭 issue **C26**。是否同步改书稿，请确认。

---

## 4. 分批实施清单（26 个训练脚本）

按「先打通链路、再横向铺开」分 4 批，每批单独提交，便于回滚。

| 批次 | 内容 | 脚本数 | 说明 |
|:---|:---|:---:|:---|
| **B1** | 基础设施（`tb_logger.py`、`plot_tb_curves.py`、requirements、`check_env.py`、ENVIRONMENT.md）+ **ch01**（`mlp_spring.py`、`pinn_spring.py`）+ **ch02**（`heat1d_pinn_raw.py`、`heat1d_pinn_sdk.py`、`heat1d_pinn_gpu.py`、`heat1d_train.py`） | 6 | 覆盖最简 argparse 脚本与最复杂 Hydra 脚本两种形态，验证方案可行性 |
| **B2** | **ch03**：`heat_sink_train.py`、`heat_sink_train_sdk.py`、`heat_sink_train_gpu.py`、`heat_sink_inverse*.py` | 4–5 | 含不保存 history 的反演脚本，改造成本较高 |
| **B3** | **ch04**（`train_fno_mini.py`、`train_fno_sdk.py`、`train_fno_gpu.py`）+ **ch05**（`train_data_fno.py`、`train_physics_fno*.py`） | 6 | ch05 需在跨章 import 处一并处理 |
| **B4** | **ch06**（`train_afno_mini.py`、`train_afno_sdk.py`、`train_afno_gpu.py`）+ **ch07**（`train.py`、`train_sdk.py`、`train_gpu.py`） | 6 | 含不保存 history 的 `train.py` / `train_gpu.py` |

> 排除（无训练循环）：各章 `visualize*.py`、`validator.py`、`dataset.py`、`*_model.py`、`rollout_eval.py`、`optimize*.py`、`export_onnx.py`、`api/app.py`、`tools/cfd_batch/*`。

---

## 5. 验收标准

1. 每个接入脚本跑 `python <script>.py --epochs 20`（或最小可行参数）后：
   - 生成 `events.out.tfevents.*` **且** `metrics.csv`；
   - 结尾打印实际日志目录；
   - 原有 `print` 日志与 PNG 产物不变。
2. `python scripts/plot_tb_curves.py --logdir <dir> --out x.png` 能出图（不依赖 tensorboard）。
3. 装了 `tensorboard` 时，`tensorboard --logdir <dir>` 能看到全部曲线。
4. `--no-tb` 与 `PN_TB=0` 生效：不产生任何日志目录。
5. 卸载 tensorboardX 后脚本仍可训练（降级 CSV，仅一条 warning）。
6. `python scripts/check_env.py` 全绿，新增检查项正确显示。
7. 分批跑通 ch01–ch07 冒烟，确认训练数值与接入前一致（同种子对比最终 loss，允许 ±1e-6 浮点误差）。

---

## 6. Issue 草稿（登记用）

- Backlog：`docs/REVIEW_ROUND2_BACKLOG.md` 追加一行（GH 编号顺延 **#54**）

  | GH | P | 标题 | 状态 |
  |:---|:---|:---|:---|
  | [#54](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues/54) | P1 | 训练脚本默认接入 tensorboardX 曲线记录 | open |

- 关联既有 issue：`docs/issues/cae-review/` 的 **C26「ch03 TensorBoard/Monitor 书码对齐」** —— 本方案实施后可直接关闭。

---

## 7. 需要拍板的 4 件事

1. **依赖层级**：`tensorboardX` 放 `requirements-minimal.txt`（ch01 起默认可用）还是 `requirements-full.txt`（仅 SDK 档）？建议 **minimal**，因为「默认开启」覆盖 raw 脚本。
2. **日志目录**：默认是否带时间戳子目录（目录会随运行次数增长）？建议**带**，另给 `--tb-no-timestamp`。
3. **书稿同步**：是否在 B4 之后一并修正 `book/ch02.md` / `book/ch03.md` 并关闭 C26？（涉及正文改动）
4. **批次数**：按上面 4 批推进，还是先只做 B1（ch01+ch02）看效果再决定？
