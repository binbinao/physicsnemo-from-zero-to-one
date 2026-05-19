# CAE 闭环演示（无需商业 CFD 许可证）

此前三项工作依赖**客户现场**（Fluent/Icepak/OpenFOAM 真网格、长步数联合反演、带工程约束的多目标进 CFD 队列）。本仓库提供 **mock / 脚本化** 路径，在笔记本或 CI 上跑通同一逻辑链；生产环境只需把 mock 换成真实求解器。

## 三项对照

| 原「须客户环境」 | 仓库内可跑方案 | 生产替换 |
|:---|:---|:---|
| Fluent/Icepak/OpenFOAM + `EXECUTE=1` | `tools/cfd_batch/run_batch.py --mode mock` | `--mode execute` + 填写 `solver_cmd` |
| 联合反演 ~3000 steps | 默认 3000 + 早停；`--fast` 用于 CI | 去掉 `--fast`，按需调 `tol` |
| 多目标 Pareto → 工程约束 → CFD 队列 | `optimize_multi.py` + `design_constraints.py` + `--export-hifi` | 按企业规范扩展 `check_engineering_constraints` |

## 一键演示

```bash
# 完整链（约数分钟 CPU）：ch07 训练 → 单/多目标优化 → mock CFD → 汇总 V&V
python scripts/run_cae_closed_loop_demo.py

# 已有 checkpoint 时跳过训练
python scripts/run_cae_closed_loop_demo.py --skip-train

# 仅气动闭环
python scripts/run_cae_closed_loop_demo.py --skip-ch03
```

产出：

- `ch07_drivaernet_optim/outputs/hifi_queue.csv` — 经工程约束筛选的待验证设计
- `results/cfd_runs/case_*/result_cd.txt` — mock 高保真 Cd
- `results/cfd_runs/validation_summary.csv` — 代理 vs mock-CFD 误差表（填入 [VV_REPORT_TEMPLATE.md](VV_REPORT_TEMPLATE.md)）

## 分步命令

### 1. ch07 代理 + 多目标 + 导出队列

```bash
cd ch07_drivaernet_optim
python data/generate_toy_cars.py --n_samples 800
python train.py --epochs 80
python optimize_multi.py --checkpoint outputs/best.pt --n_trials 150 --export-hifi --top-k 5
```

### 2. Mock CFD 批跑

```bash
python tools/cfd_batch/import_hifi_queue.py \
  --csv ch07_drivaernet_optim/outputs/hifi_queue.csv --mode mock
python tools/cfd_batch/run_batch.py --mode mock
python tools/cfd_batch/ingest_results.py
```

`mock` 使用与 `generate_toy_cars.compute_cd` 相同的解析公式并加小噪声，模拟「网格/求解器与代理偏差」，**不**需要网格文件或许可证。

### 3. ch03 联合反演

```bash
cd ch03_heatsink
python heat_sink_inverse_joint.py --target_temp 40          # 默认 3000 步 + 早停 |err|<0.5
python heat_sink_inverse_joint.py --target_temp 40 --fast   # 800 步，CI/演示
```

说明：联合反演已修复 **鳍高梯度**（tip 观测对 `fin_height` 可微、每步按当前 `h` 重采样几何）。演示 PINN 在无量纲坐标下 **tip 温度动态范围有限**，`target_temp=40` 可能无法严格达到；签审请用 `validator.py` + 扫描版 `heat_sink_inverse.py` 对照。闭环脚本中 ch03 仅作「可运行」检查（`--fast`）。

## 接入真实 CFD

1. 将 `tools/cfd_batch/templates/openfoam/` 或 `templates/fluent/` 复制到各 `results/cfd_runs/case_XXXX/`。
2. `import_hifi_queue.py --mode openfoam|fluent` 生成占位 `solver_cmd`。
3. 在 `manifest.json` 中为每个 case 填写可执行命令（或环境变量 `EXECUTE=1` 配合 `run_batch.sh`）。
4. `run_batch.py --mode execute` 在 case 目录执行命令；求解器写出 `result_cd.txt`（格式：`cd=0.312`）。
5. `ingest_results.py` 不变。

详见 [tools/cfd_batch/README.md](../tools/cfd_batch/README.md)。

## 工程约束扩展

编辑 [ch07_drivaernet_optim/design_constraints.py](../ch07_drivaernet_optim/design_constraints.py)：

- `check_engineering_constraints` — 硬约束（导出 hifi 前过滤）
- `penalty_score` — Optuna 软惩罚（`optimize_multi.py` 已接入）
