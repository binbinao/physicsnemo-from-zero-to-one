# CFD 批跑工作流（C36）

## 模式

| 模式 | 命令 | 说明 |
|:---|:---|:---|
| **dry-run** | `run_batch.py`（默认） | 仅打印 `solver_cmd`，不计算 |
| **mock** | `run_batch.py --mode mock` | 仓库内合成 CFD，无需许可证 |
| **execute** | `run_batch.py --mode execute` | 执行 manifest 中的真实求解器命令 |

完整闭环说明：[docs/CAE_CLOSED_LOOP_DEMO.md](../../docs/CAE_CLOSED_LOOP_DEMO.md)

## 流程

```text
ch07 optimize_multi --export-hifi → hifi_queue.csv
       ↓
import_hifi_queue.py (--mode mock|openfoam|fluent) → manifest.json
       ↓
run_batch.py (--mode mock|execute) → results/cfd_runs/case_XXXX/
       ↓
ingest_results.py → validation_summary.csv
```

## 命令

```bash
# 1. 多目标 + 工程约束筛选后导出
cd ch07_drivaernet_optim
python optimize_multi.py --checkpoint outputs/best.pt --export-hifi --top-k 5

# 2. 生成清单（mock 可直接跑通）
python tools/cfd_batch/import_hifi_queue.py \
  --csv ch07_drivaernet_optim/outputs/hifi_queue.csv --mode mock

# 3. Mock 或真实批跑
python tools/cfd_batch/run_batch.py --mode mock
# python tools/cfd_batch/run_batch.py --mode execute   # 需填写 solver_cmd

# 4. 汇总 V&V
python tools/cfd_batch/ingest_results.py
```

亦可一键：`python scripts/run_cae_closed_loop_demo.py`

## 目录

| 路径 | 说明 |
|:---|:---|
| `mock_cfd_solver.py` | 读 `design_params.json`，写 `result_cd.txt` |
| `run_batch.py` | dry-run / mock / execute |
| `templates/openfoam/` | OpenFOAM 占位 |
| `templates/fluent/` | Fluent journal 模板 |
| `manifest.schema.json` | 清单字段说明 |

生产环境：将 `templates/` 换成本机 deck，在 `manifest.json` 中填写 `solver_cmd`，`import_hifi_queue.py --mode fluent|openfoam`。
