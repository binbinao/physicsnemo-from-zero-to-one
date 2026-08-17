## Learned User Preferences

- 用户用中文交流，助手回复应使用中文
- 改进工作流：先登记 issues/backlog，再逐项处理；完成某项时用 `closes #N` 触发实现、提交并关闭对应 issue
- 全书级或配图级整改：先输出审阅/QA 文档或清单，待用户 review 确认后再批量改图、提交或合并

## Learned Workspace Facts

- PhysicsNeMo 教程仓库：`book/` 教材与 `ch01_hello/`–`ch07_drivaernet_optim/` 代码 1:1 对应；`scripts/` 为跨章工具
- 每章有三档变体：裸 PyTorch（`*_raw.py`）、PhysicsNeMo SDK（`*_sdk.py`）、GPU 生产（`*_gpu.py`）
- `ch04_fno_airfoil`：目录名表翼型场景，默认训练脚本用 Darcy；见 `ch04_fno_airfoil/CH04_GUIDE.md`
- `ch05_darcy_hybrid` 是唯一显式跨章代码依赖（import `ch04_fno_airfoil`）
- 小白读者改进项跟踪在 `docs/BEGINNER_ISSUE_BACKLOG.md`，GitHub issues 带 `[Beginner]` 标签，正文模板在 `docs/issues/`
- 环境与依赖版本以 `docs/ENVIRONMENT.md` 为单一事实来源
- 教材体例以 `book/STYLE.md` 为准；全书目录 `book/SUMMARY.md`，前置说明 `book/FRONT_MATTER.md`
- 配图生产与图号维护见 `book/assets/FIGURE_MANIFEST.md`；Matplotlib 出图脚本在 `book/scripts/`，无 CJK 字体会出现方块字，整改跟踪 `docs/BOOK_FIGURE_ART_QA.md`

## Cursor Cloud specific instructions

- 本仓库是 Python 教程（PINN / 神经算子），无常驻后端。核心「应用」= 各章一次性训练脚本 + 可选的 ch07 FastAPI 推理服务。无 pytest/unittest 套件，也未配置 linter；事实上的自动化测试 = `.github/workflows/smoke.yml`（CPU 跑 ch01–ch07 裸 PyTorch）。
- 环境/依赖分层的单一事实来源是 `docs/ENVIRONMENT.md`；书—代码命令见 `docs/COMMAND_REFERENCE.md`。
- 系统 Python 是 3.12（Ubuntu 24.04，PEP 668 externally-managed）。用 `pip` 装包必须加 `--break-system-packages`，装到 `~/.local`，直接用 `python3` 即可，无需 venv/激活。启动 update 脚本已装好 `requirements-minimal.txt`（Tier 0/1：torch/numpy/matplotlib/scipy），足够跑 ch01–ch07 裸 PyTorch 与 CI smoke 的训练步骤。
- 无 GPU：`torch` 走 CPU 模式（`check_env.py` 会显示 CUDA ❌，属正常）。`*_gpu.py` 与需 CUDA 的档位不在本环境范围。
- 复现完整 CI smoke（含 ONNX 导出）：ONNX 步骤需额外 `pip install --break-system-packages onnx onnxscript`（CI 也是在该步单独装）。命令序列见 `.github/workflows/smoke.yml`；注意 ch06 需先 `python scripts/generate_toy_weather.py`，ch07 需先 `python data/generate_toy_cars.py`。
- **Hydra 陷阱**：一旦装了 `hydra-core`，ch04/ch05/ch06（及部分 ch02/ch03）入口会切到 Hydra，`smoke.yml` 里的 `--epochs` / `--n_samples` 会报 `unrecognized arguments`。改用覆盖语法，例如 `python train_fno_mini.py epochs=3 n_samples=20`、`python train_data_fno.py epochs=3 n_train=20`、`python train_afno_mini.py epochs=2`。ch07 `train.py` 仍是 argparse。详见 `docs/COMMAND_REFERENCE.md`。
- 可选档位按需安装（均装得上、可靠）：`hydra-core optuna fastapi uvicorn`；`nvidia-physicsnemo` 也能装上（CPU 可 import）。但 `nvidia-physicsnemo.sym` 需源码编译且构建隔离下看不到 torch 会失败（需 `--no-build-isolation` 且面向 GPU）——`*_sdk.py` 依赖它，故 SDK 变体默认跑不了，属可选、非必需。
- 运行 ch07 FastAPI 推理服务：先确保有 checkpoint（`cd ch07_drivaernet_optim && python data/generate_toy_cars.py --n_samples 200 && python train.py --epochs 5 --n_samples 200` 产出 `outputs/best.pt`），再 `python -m uvicorn api.app:app --host 0.0.0.0 --port 8000`（`uvicorn` 可执行文件在 `~/.local/bin`，未必在 PATH，用 `python -m uvicorn` 最稳）。测试：`curl -X POST localhost:8000/predict_cd -H 'Content-Type: application/json' -d '{"body_length":4.5,"body_width":1.8,"body_height":1.4,"front_angle":20,"rear_angle":15,"ground_clearance":0.15,"wheel_diameter":0.65}'`，交互文档在 `/docs`。
- 训练/演示脚本会写出 checkpoint、图片，并会修改 `results/cfd_runs/`、`tools/cfd_batch/manifest.json` 等被跟踪文件；提交前注意别把这些运行产物一起提交。
