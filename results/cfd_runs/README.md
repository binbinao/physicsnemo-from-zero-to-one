# CFD 批跑结果目录（C39）

每工况一个子目录，由 `tools/cfd_batch/run_batch.sh` 创建。

## 必填（回灌用）

`case_XXXX/result_cd.txt` — 单行，例如：

```text
cd=0.251
```

## 可选

- `design_params.json` — 由批跑脚本自动写入  
- 求解器日志、VTK — 客户自行归档  

汇总：

```bash
python tools/cfd_batch/ingest_results.py
# → results/cfd_runs/validation_summary.csv
```

填入 [VV_REPORT_TEMPLATE.md](../../docs/VV_REPORT_TEMPLATE.md) §6。
