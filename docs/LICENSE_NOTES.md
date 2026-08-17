# 许可证与合规说明（C30）

## 本仓库（双协议）

本仓库为"教材 + 配套代码"混合项目，采用双协议分开授权：

- **代码**（`chXX_*/`、`tools/`、`scripts/`、`notebooks/`、配置文件）：[Apache License 2.0](../LICENSE)。含显式专利授权，与上游 NVIDIA PhysicsNeMo 一致。
- **书稿正文与配图**（`book/`、`book_en/`、`docs/`、`article/`、`book/assets/`）：[CC BY 4.0](../LICENSE-docs)。署名即可自由使用、改编、商用。
- 同时兼具二者的文件：可执行代码适用 Apache-2.0，随附的散文/说明适用 CC BY 4.0。

## NVIDIA PhysicsNeMo

- `nvidia-physicsnemo`、`nvidia-physicsnemo.sym` 遵循 **NVIDIA 各包自己的许可证**（常见为 Apache-2.0，以 PyPI / GitHub 为准）。  
- **企业部署前**请法务确认：再分发、SaaS、出口合规。  

## 第三方求解器

- ANSYS、Siemens STAR-CCM+、Icepak、OpenFOAM 等名称仅为 **工作流说明**；需客户自行持有许可证。  
- 本书 **不提供** 求解器安装包或破解指引。  

## 数据

- DrivAerNet、AirfRANS 等公开数据集各有 **引用与使用条款**；商用前请阅读原始协议。

## 教材 AI 配图（路线 B）

- 部分章横幅、路线图、行业示意图可能由 **生成式 AI**（如 Google Gemini、Cursor 图像生成）制作，提示词见 [BOOK_FIGURE_GEMINI_PROMPTS.md](BOOK_FIGURE_GEMINI_PROMPTS.md)。  
- 定稿文件若位于 `book/assets/gemini/`，合并前须经编辑验收；**数据类曲线/场图** 仍为 Matplotlib 程序化生成，非文生图。
