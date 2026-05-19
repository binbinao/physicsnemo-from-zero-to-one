# CFD 批跑工作流（骨架 · C36）

> **不包含** Fluent/Icepak 许可证与求解器安装；默认 **dry-run** 打印待执行命令。

## 流程

```text
ch07 optimize → hifi_queue.csv
       ↓
import_hifi_queue.py → manifest.json
       ↓
run_batch.sh (--execute 可选) → results/cfd_runs/case_XXXX/
       ↓
ingest_results.py → 汇总 Cd 供 V&V 报告
```

## 命令

```bash
# 1. 从 ch07 导出设计
cd ch07_drivaernet_optim
python hifi_validation_queue.py --checkpoint outputs/best.pt --top_k 5

# 2. 生成批跑清单
python tools/cfd_batch/import_hifi_queue.py \
  --csv ch07_drivaernet_optim/outputs/hifi_queue.csv \
  --out tools/cfd_batch/manifest.json

# 3. 预览命令（默认）
bash tools/cfd_batch/run_batch.sh

# 4. 回灌结果（填写各 case 的 result_cd.txt 后）
python tools/cfd_batch/ingest_results.py
```

## 目录

| 路径 | 说明 |
|:---|:---|
| `templates/openfoam/` | OpenFOAM case 占位 `README` + `Allrun.template` |
| `templates/fluent/` | Fluent journal 模板 `.jou` |
| `manifest.schema.json` | 清单 JSON 字段说明 |

客户需将 `templates/` 替换为本机求解器 deck，并在 `manifest.json` 中填写 `solver_cmd`。
