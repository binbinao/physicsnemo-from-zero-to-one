# 读完全书之后 · 下一步路线图

> 正文 7 章 + 附录 A–D 完成后，按你的目标选路径。

---

## 0. 能力矩阵（自检）

| 能力 | 应能独立完成 | 对应章 |
|:---|:---|:---|
| 跑通环境 + 弹簧 PINN | ✅ / ❌ | ch01 |
| 改 Hydra 配置重训 1D 热 | ✅ / ❌ | ch02 |
| 解释多 BC 散热片 setup | ✅ / ❌ | ch03 |
| 说清 sym→主框架切换 + 跑通 Darcy FNO | ✅ / ❌ | ch04 |
| 对比 data-only vs +physics FNO | ✅ / ❌ | ch05 |
| 做短 rollout 天气 demo | ✅ / ❌ | ch06 |
| train→optimize→ONNX 最小闭环 | ✅ / ❌ | ch07 |

有 ❌：回到 [START_HERE.md](START_HERE.md) 与 [cheatsheets/](cheatsheets/README.md) 补对应章。

---

## 1. 巩固本书内容

| 动作 | 资源 |
|:---|:---|
| 对照命令再跑一遍 | [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) |
| 查硬件/时长是否合理 | [HARDWARE_EXPECTATIONS.md](HARDWARE_EXPECTATIONS.md) |
| 对照 loss 趋势 | [results/BASELINE.md](../results/BASELINE.md) |
| 改进项跟踪 | [BEGINNER_ISSUE_BACKLOG](BEGINNER_ISSUE_BACKLOG.md) · [REVIEW_ROUND2](REVIEW_ROUND2_BACKLOG.md) |

---

## 2. 官方 PhysicsNeMo 生态

| 资源 | 链接 |
|:---|:---|
| 主仓库 | https://github.com/NVIDIA/physicsnemo |
| 文档 | https://docs.nvidia.com/deeplearning/physicsnemo/ |
| Examples | 仓库 `examples/` 目录（FNO、CFD、气候等） |
| 发布说明 | 关注 v2.x API 变更 |

**建议**：选 **一个** 与本书最接近的 official example（如 Darcy FNO 或 CFD）复现，再迁移到你的 PDE/几何。

---

## 3. 换真实数据集

| 领域 | 数据集方向 | 与本书章节 |
|:---|:---|:---|
| 翼型 CFD | AirfRANS | ch04 叙事 |
| 汽车气动 | DrivAerNet | ch07 |
| 天气 | ERA5 / GraphCast 类 | ch06 |
| 多孔介质 | 开源 Darcy 基准 | ch05 |

数据准备往往比训练更耗时——保留本书 **合成数据流程** 作调试基线。

---

## 4. 部署与工程化

| 步骤 | 工具 |
|:---|:---|
| 导出 ONNX | ch07 `export_onnx.py` |
| 推理服务 | [NVIDIA Triton](https://github.com/triton-inference-server/server) |
| API | ch07 `api/app.py`（FastAPI 演示） |
| HPO | Optuna（ch07 `optimize.py`） |

---

## 5. 理论与论文（按兴趣）

| 主题 | 参考 |
|:---|:---|
| PINN | Raissi et al., JCP 2019 |
| FNO | Li et al., ICLR 2021 |
| 物理信息算子 | 本书 ch05 + PhysicsNeMo hybrid 示例 |
| 天气预报 NN | Lam et al., GraphCast |

---

## 6. 社区与贡献

| 动作 | 说明 |
|:---|:---|
| 给本书提 Issue/PR | 改进文档、补 baseline、修命令 |
| PhysicsNeMo Discussions | 官方论坛 / GitHub Discussions |
| 内部推广 | 用 ch04/ch07 demo 做 30 分钟内部分享 |

---

## 7. 推荐 30 天行动（单选聚焦）

```text
Week 1–2: 官方 example 复现（与 ch04 或 ch07 同族）
Week 3:   接入一个真实小数据集（<10GB）
Week 4:   ONNX/Triton 或集成进现有 CAE 脚本
```

祝你从 **Zero** 到 **One** 之后，走向 **Your Problem**。🚀
