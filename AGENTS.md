## Learned User Preferences

- 用户用中文交流，助手回复应使用中文
- 改进工作流：先登记 issues/backlog，再逐项处理；完成某项时用 `closes #N` 触发实现、提交并关闭对应 issue

## Learned Workspace Facts

- PhysicsNeMo 教程仓库：`book/` 教材与 `ch01_hello/`–`ch07_drivaernet_optim/` 代码 1:1 对应；`scripts/` 为跨章工具
- 每章有三档变体：裸 PyTorch（`*_raw.py`）、PhysicsNeMo SDK（`*_sdk.py`）、GPU 生产（`*_gpu.py`）
- `ch04_fno_airfoil`：目录名表翼型场景，默认训练脚本用 Darcy；见 `ch04_fno_airfoil/CH04_GUIDE.md`
- `ch05_darcy_hybrid` 是唯一显式跨章代码依赖（import `ch04_fno_airfoil`）
- 小白读者改进项跟踪在 `docs/BEGINNER_ISSUE_BACKLOG.md`，GitHub issues 带 `[Beginner]` 标签，正文模板在 `docs/issues/`
- 环境与依赖版本以 `docs/ENVIRONMENT.md` 为单一事实来源
