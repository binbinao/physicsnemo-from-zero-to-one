# 环境要求（单一事实来源）

> 全仓库以本文档为准；README、`book/ch00.md`、`scripts/check_env.py` 与之对齐。

## Python

| 项 | 要求 |
|:---|:---|
| **最低版本** | Python **3.10** |
| **推荐版本** | Python **3.11** |
| **不支持** | Python 3.9 及以下（`check_env` 会报错） |

### 用 uv 管理环境（推荐）

[uv](https://docs.astral.sh/uv/) 可自动下载 Python 并创建隔离环境，命令在所有平台一致：

```bash
# 安装 uv（Windows PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux / macOS / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv python install 3.11              # 下载推荐版本（无需系统预装 Python）
uv venv --python 3.11 --seed        # 在项目根目录创建 .venv（--seed 附带 pip）
. .venv/bin/activate                # Linux/macOS；Windows 用 .venv\Scripts\activate
uv pip install -r requirements-minimal.txt     # Tier 0–1
uv pip install -r requirements-full.txt        # Tier 2–3（SDK + 部署）
```

要点：

- 若系统已有 Python，也可直接 `uv venv --python 3.12` 复用本机版本，不必重新下载。
- `uv pip` 用法与 `pip` 基本一致，激活环境后可直接用 `python`/`pip`。
- Windows 下 PyTorch 默认下载 CUDA 版 wheel（体积较大）；纯 CPU 可改用
  `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`。
- `.venv/`、`.python-version` 已加入仓库 `.gitignore`。

## 训练曲线（TensorBoard，默认开启）

所有训练脚本默认记录 loss 等标量曲线，无需额外参数：

- **记到哪里**：`scripts/tb_logger.py` 同时写出 TensorBoard 事件文件与 `metrics.csv`。
  目录按脚本类型自动确定（每次运行一个时间戳子目录）：

  | 脚本类型 | 默认日志目录 |
  |:---|:---|
  | Hydra 脚本（如 `ch02/heat1d_train.py`） | Hydra 运行目录 `/tb` |
  | 有 `--out_dir` / `--output_dir` 的脚本 | `<out_dir>/tb/<时间戳>` |
  | 其余脚本 | `outputs/tb/<脚本名>/<时间戳>` |

  训练结束会打印实际目录。

- **训练中看**：`tensorboard --logdir <目录>`（需 `pip install tensorboard`，可选依赖）。
- **训练后看（零依赖）**：`python scripts/plot_tb_curves.py --logdir <目录> --out curves.png`，
  直接读 `metrics.csv` 出图；支持 `--metrics loss/total,val/mae`、`--smooth 0.6`、`--linear-y`。
- **关闭**：环境变量 `PN_TB=0`（**所有脚本通用，Hydra 档也可用**）；纯 argparse 脚本也可加 `--no-tb`，但 Hydra 档脚本（ch02–ch06 的 Hydra 入口）会把 `--no-tb` 当非法 override 报错，只能用 `PN_TB=0`。`PN_TB_DIR` 可覆写日志根目录。
- **自定义**：`--tb-dir PATH`、`--tb-tag NAME`、`--tb-no-timestamp`（仅 argparse 档生效；`--tb-dir` 指定目录时 `--tb-tag` 不参与拼路径）。
- **依赖**：`tensorboardX>=2.6` 已在 `requirements-minimal.txt`。
  未安装时脚本**自动降级为只写 CSV**（打印一条提示），训练不受影响。

详见 [TENSORBOARD_INTEGRATION_PLAN.md](TENSORBOARD_INTEGRATION_PLAN.md)。

## PyTorch

| 项 | 要求 |
|:---|:---|
| **最低版本** | PyTorch **2.3+** |
| **CUDA** | 第 4 章起建议 **CUDA 12.x** 对应 wheel；仅学 ch1–2 可用 CPU |
| **安装** | 见 [pytorch.org](https://pytorch.org/get-started/locally/) |

## 分阶段依赖

### Tier 0 — 第 1 天（ch01 裸 PyTorch）

```bash
pip install "torch>=2.3" numpy matplotlib scipy
# 或：pip install -r requirements-minimal.txt（含 scipy）
```

详见 [QUICKSTART_DAY1.md](QUICKSTART_DAY1.md)。

### Tier 1 — ch01–ch03 裸 PyTorch + ch04/ch05 数据生成（需 scipy）

```bash
pip install "torch>=2.3" numpy matplotlib scipy
# 推荐：pip install -r requirements-minimal.txt
```

### 教材配图（`book/assets/`）

含中文标注的图由 **Matplotlib 程序化出图**（非文生图 AI），须能加载仓库内置 **Noto Sans SC**：

```bash
python scripts/download_book_fonts.py   # 若缺少 book/assets/fonts/NotoSansSC-Regular.otf
python scripts/check_figure_cjk.py
python book/scripts/generate_diagram_figures.py
python book/scripts/generate_all_figures.py
```

详见 [BOOK_FIGURE_ART_QA.md](BOOK_FIGURE_ART_QA.md)、[book/assets/fonts/README.md](../book/assets/fonts/README.md)。

### Tier 2 — SDK 版（ch01–ch07 的 `*_sdk.py`）

```bash
pip install "torch>=2.3" numpy matplotlib scipy nvidia-physicsnemo nvidia-physicsnemo.sym hydra-core
```

> **Hydra 注意**：装上 `hydra-core` 后，部分训练入口改用 `key=value` 语法（如 ch03 `steps=500`、ch04–ch06 `epochs=50`、ch02 `training.steps=500`），不要用 `--epochs` / `--steps`。
> 键名以各章 `conf/config.yaml` 为准；完整对照见 [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md#hydra-vs-argparse必读)。

### Tier 3 — 全书 + 部署（ch07 API / ONNX）

```bash
pip install "torch>=2.3" numpy matplotlib scipy nvidia-physicsnemo nvidia-physicsnemo.sym \
  hydra-core optuna fastapi uvicorn onnx
```

## 按章依赖速查

| 章 | 首选脚本 | 最低 Tier | GPU |
|:---|:---|:---|:---|
| ch01 | `pinn_spring.py` | 0 | 可选 |
| ch02 | `heat1d_pinn_raw.py` 或 `heat1d_train.py` | 0 / 1 | 可选 |
| ch03 | `heat_sink_train.py` | 1 | 建议 8GB |
| ch04 | `train_fno_mini.py` | 1 | 建议 8GB |
| ch05 | `train_data_fno.py` | 1 + **ch04 目录** | 建议 8GB |
| ch06 | `train_afno_mini.py` | 1 | 建议 8GB |
| ch07 | `train.py` | 1；Optuna 要 Tier 3 | 建议 8GB |

完整表：[DEPENDENCIES_BY_CHAPTER.md](DEPENDENCIES_BY_CHAPTER.md) · 硬件/时长：[HARDWARE_EXPECTATIONS.md](HARDWARE_EXPECTATIONS.md)

## 验证

```bash
python scripts/check_env.py           # 全局核心 + Tier 概览
python scripts/check_env.py --list  # 各章 Tier / GPU 速查
python scripts/check_env.py --chapter 4
python scripts/check_env.py --tier 2
```

`--chapter` 会检查该章必需包（含 ch05 的 `ch04_fno_airfoil/` 目录）及可选 SDK 包。

## 云 GPU / Colab

[CLOUD_GPU_GUIDE.md](CLOUD_GPU_GUIDE.md) · [notebooks/colab_quickstart.ipynb](../notebooks/colab_quickstart.ipynb)

## SDK 装不上时怎么办

| 情况 | 建议 |
|:---|:---|
| 无 NVIDIA GPU / Mac Apple Silicon | 先只装 `requirements-minimal.txt`，跑 ch01–ch03 裸 PyTorch；ch04+ 用 [Colab](CLOUD_GPU_GUIDE.md) |
| `pip install nvidia-physicsnemo` 失败 | 查 [附录 C](../book/appendix_c_troubleshooting.md)；暂跳过 `*_sdk.py`，不影响 raw 脚本 |
| 仅需某一章 SDK | 见 [DEPENDENCIES_BY_CHAPTER.md](DEPENDENCIES_BY_CHAPTER.md)，不必一次装全量 |

**Apple Silicon**：PyTorch 可用 MPS 加速部分算子，但 PhysicsNeMo 官方以 Linux + CUDA 为主；本书微缩实验建议在 Colab T4 或 Linux GPU 上跑 ch04–ch07。

**Windows**：推荐 [WSL2](https://learn.microsoft.com/windows/wsl/) + Ubuntu，在 WSL 内按 Linux 步骤安装；原生 Windows 下 sym 常遇路径/编译问题。

## 依赖文件

| 文件 | 用途 |
|:---|:---|
| [requirements-minimal.txt](../requirements-minimal.txt) | ch01–ch03 裸 PyTorch |
| [requirements-full.txt](../requirements-full.txt) | SDK + ch07 |
| [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md) | 最近验证的 commit / Python / PyTorch |
