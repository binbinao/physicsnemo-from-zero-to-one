# 部署与生产化差距（C23–C24）

## 本书演示级

- ch07 `api/app.py`：单进程 FastAPI，无认证、无限流  
- ONNX 导出：单模型文件，无动态 batch 优化  

## 生产建议

| 项 | 建议 |
|:---|:---|
| 推理 | NVIDIA Triton + TensorRT；批处理延迟 SLA 单独测 |
| 多租户 | GPU 队列 / 模型版本路由 |
| 输入 | JSON schema + 设计空间约束（见 `optimize.py` OOD 检查） |
| 集成 | ONNX → 自研 C++ / Fluent UDF / STAR-CCM+ 耦合：需客户侧开发 |

## ONNX 进 CAE 流程（概念）

1. `export_onnx.py` 生成 `cd_model.onnx`  
2. 在优化循环中用 ONNX Runtime 批量评估（替代 PyTorch）  
3. 签审仍依赖 CFD 复核，而非 ONNX 输出 alone  

## API / SDK 变动（C25）

PhysicsNeMo v2.x API 可能变更；生产项目应 **pin 版本** 并跟踪 [NVIDIA releases](https://github.com/NVIDIA/physicsnemo/releases)。
