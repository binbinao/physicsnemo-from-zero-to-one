# PhysicsNeMo SDK 兼容说明（简表）

> 官方 API 随版本变化；**以本仓库 `*_sdk.py` 与 [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md) 为准**。

| 组件 | 本书用法 | 备注 |
|:---|:---|:---|
| `nvidia-physicsnemo` | ch04–ch07 主框架、`FullyConnected` 等 | 装不上可只跑 raw 脚本 |
| `nvidia-physicsnemo.sym` | ch01–ch03 PINN 符号求解 | ch04 起不再必需 |
| Hydra | ch02 `heat1d_train.py`，部分 ch04+ 脚本 | `hydra-core>=1.3` |
| import 路径 | `physicsnemo.sym.*` vs `physicsnemo.*` | 见 [FRAMEWORK_SWITCH.md](FRAMEWORK_SWITCH.md) |

升级 SDK 后若 import 失败：对照 [NVIDIA 发布说明](https://github.com/NVIDIA/physicsnemo/releases) 与对应章 `*_sdk.py`。
