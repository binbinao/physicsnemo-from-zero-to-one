# ch03 散热片 PINN · 生产推理包

把 `outputs/heat_sink.pt` 这份**研究性 checkpoint** 变成**可部署、可观测、可校验**的推理制品，服务于 CAE/优化循环等生产调用方。

```
        ┌────────────────────────────────────────────────────────┐
        │ 1. 质量门槛        │ 2. 制品封装 (export.py)            │
  heat_sink.pt ── validator.py 报告(pass_all) ──► model.onnx     │
  + validation_report.json      ◄ 记录到 meta                     │
                                          + model_meta.json       │
        └────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │ 3. 推理服务 (serve.py / Dockerfile)                    │
        │   FastAPI + ONNX Runtime（无需 PyTorch）               │
        │   /v1/temperature · /batch · /field · /readyz /metrics │
        └────────────────────────────────────────────────────────┘
```

## 为什么不是直接把 `.pt` 丢进服务？

| 生产要求 | 本包做法 |
|:---|:---|
| **加载与训练解耦** | `net.py` 独立复刻网络（无 Hydra/matplotlib 依赖）；部署镜像只装 `onnxruntime`（≈200 MB，PyTorch ≈2 GB+） |
| **输入含义随模型走** | 坐标单位 mm、内部 `/scale` 归一化**固化进 ONNX 图**；`model_meta.json` 记录物理/几何/输入域 |
| **不做无意义的预测** | `domain.py` 按几何重建固体域，气隙/域外坐标 → HTTP 422，而不是给一个网络从未见过的数字 |
| **坏模型不许上线** | `export.py --require-quality`：以 `validator.py` 的 `pass_all` 为门槛（当前 `heat_sink.pt` 仅训 30 步，**门槛 FAIL**，会警告/拒绝） |
| **可观测/可运维** | `/healthz` `/readyz` `/metrics`（Prometheus）、`X-Request-Id`、结构化请求日志、批次/网格规模上限 |
| **多端点形态** | 单点 `/v1/temperature`、批量 `/v1/temperature/batch`、整场 `/v1/field`（CAE 最常用） |

## 快速开始（本机）

> **fresh clone 请先 train + export**：`artifacts/` 制品目录不随仓库分发
> （ONNX/权重属训练产物，与 `outputs/`、`*.pt` 同口径被 gitignore），下面的
> 步骤 1-2 会在本地生成 `inference/artifacts/heat_sink/`。

```bash
# 0) 依赖（.venv 已含 torch/fastapi；补运行时库）
.venv\Scripts\python -m pip install -r ch03_heatsink/inference/requirements.txt

# 1) 训练出**通过质量门槛**的模型
#    validator.py 阈值：pde/robin 残差 RMS < 0.1 且温度范围合理（pass_all=true）。
#    经验：默认 3000 步（tanh/无 Fourier）不足以收敛到该残差阈值；当前最佳
#    配方是 exp7（SiLU + hard_bottom + RFF Fourier，8000 步，pass_all 仍为
#    false 但残差已到 ~16/10 量级），复现命令见 ../EXPERIMENTS_PASS.md §9。
.venv\Scripts\python ch03_heatsink/heat_sink_train.py steps=8000 hidden=192 depth=6 activation=silu hard_bottom=true fourier_maps=64

# 2) 导出制品（含质量门槛提示，CI 可加 --require-quality）
cd ch03_heatsink
.venv\Scripts\python inference/export.py --ckpt outputs/heat_sink.pt

# 3) 一致性 + 冒烟
.venv\Scripts\python inference/test_smoke.py --ckpt outputs/heat_sink.pt

# 4) 启动服务
.venv\Scripts\python -m uvicorn inference.serve:app --port 8000

# 5) 调 API
curl -X POST http://127.0.0.1:8000/v1/temperature -H "Content-Type: application/json" \
     -d "{\"x_mm\":0,\"y_mm\":5}"
curl -X POST http://127.0.0.1:8000/v1/temperature/batch -H "Content-Type: application/json" \
     -d "{\"points\":[{\"x_mm\":-20,\"y_mm\":2},{\"x_mm\":0,\"y_mm\":20}]}"
curl -X POST http://127.0.0.1:8000/v1/field -H "Content-Type: application/json" -d "{\"nx\":300,\"ny\":150}"
```

## Docker 部署

```bash
cd ch03_heatsink
python inference/export.py --ckpt outputs/heat_sink.pt     # 先导出
docker build -f inference/Dockerfile -t heat-sink-api:1.0.0 .
docker run --rm -p 8000:8000 heat-sink-api:1.0.0
```

## 制品目录约定

```
inference/artifacts/heat_sink/
  model.onnx       输入 x_mm/y_mm（动态 batch）→ 输出 T；归一化已在图内
  model.onnx.data  ONNX 外部权重数据（torch.onnx.export 自动生成）
  model_meta.json  schema v1：net/physics/geometry/inputs_spec/quality/provenance(sha256)
```

- 该目录由 `export.py` 本地生成，**不入库**（`.gitignore` 覆盖）；三件套必须来自同一次导出，勿手工混搭。

- 把 `artifacts/heat_sink/` 当成**不可变发布单元**：改名=新版本，结合镜像 tag 或对象存储即可做模型版本化/回滚。
- `model_meta.json` 里 `provenance.source_ckpt_sha256` 保证可追溯到训练产物。
- 服务通过 `ARTIFACT_DIR` 环境变量指向新目录即可热换版本（重启后生效）。

## 引擎切换（校验用途）

```bash
# torch 引擎（读原始 ckpt，做 golden 对照）
ENGINE=torch CKPT_PATH=outputs/heat_sink.pt python -m uvicorn inference.serve:app --port 8000
```

## 与本书其他部署先例的关系

- 比 `ch07_drivaernet_optim/api/app.py`（演示级：无校验/无指标/服务直接依赖训练代码）多了：制品化、物理域校验、OOD 拒绝、质量门槛、观测性、双引擎。
- 概念衔接：`docs/CAE_DEPLOYMENT_NOTES.md`（C23–C24 生产差距表）。真·大规模场景可把 `model.onnx` 交给 NVIDIA Triton/TensorRT；本服务覆盖中小规模/内嵌 CAE 循环的默认需求。

## 生产检查单（本包覆盖对照）

- [x] 模型质量门槛（部署前 gate）
- [x] 制品自包含 + 元数据 + sha256 溯源
- [x] 物理输入域校验（几何精确，非仅 AABB）
- [x] 无 PyTorch 的轻量运行时
- [x] 健康/就绪/指标 + 请求 ID
- [x] 批量与整场两种高频调用形态 + 规模上限
- [ ] 认证/授权（网关侧，超出本章范围）
- [ ] 熔断/背压/限流（视 QPS 加在反向代理）
