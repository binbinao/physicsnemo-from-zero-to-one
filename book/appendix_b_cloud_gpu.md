# 附录 B · 云 GPU 选型与一键环境

> **用途**：本地无 NVIDIA GPU、或显存不足时，用云主机跑第 4–7 章。  
> **5 分钟目标**：拿到一台带 CUDA 的机器，clone 仓库并跑通 `scripts/check_env.py`。  
> **分步实操（Colab / AutoDL / 阿里云）**：见 [`docs/CLOUD_GPU_GUIDE.md`](../docs/CLOUD_GPU_GUIDE.md)  
> **一键 Colab 笔记本**：[`notebooks/colab_quickstart.ipynb`](../notebooks/colab_quickstart.ipynb)（在 Colab 中打开即可）

---

## B.1 什么时候需要云 GPU？

| 场景 | 建议 |
|:---|:---|
| 只学第 1–2 章 | 本机 CPU 即可 |
| 第 3 章 2D 散热 | 8GB 消费卡或 Colab T4 |
| 第 4–7 章 FNO / 训练 | ≥8GB 显存；完整版建议 16–24GB |
| 第 6 章多卡 DDP | 单机多卡或云上单机 8×GPU（可选） |

---

## B.2 国内云厂商对比（2025–2026 参考）

> 价格与机型随促销变动，以下按**易用性 + AI 镜像**排序，下单前以官网为准。

| 厂商 | 典型 GPU | 适合人群 | 优点 | 注意 |
|:---|:---|:---|:---|:---|
| **AutoDL** | RTX 4090 / A100 按量 | 学生、个人 | 开箱即用 PyTorch 镜像、按小时计费 | 高峰排队 |
| **阿里云 PAI-DSW** | A10 / V100 / A100 | 企业、已有阿里云账号 | 与对象存储/OSS 集成好 | 配置项多 |
| **腾讯云 GPU 云服务器** | T4 / A10 / A100 | 企业 | 文档全、地域多 | 需自己选 CUDA 镜像 |
| **华为云 ModelArts** | Ascend / GPU 实例 | 政企、昇腾生态 | 国内合规 | PhysicsNeMo 以 NVIDIA CUDA 为主，Ascend 需另查兼容性 |

**省钱建议**：先选 **按量计费 + 预装 PyTorch 2.x + CUDA 12** 的镜像；训完即关机。

---

## B.3 五分钟环境（云主机 / 本地 Linux）

```bash
# 1. 克隆仓库
git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
cd physicsnemo-from-zero-to-one

# 2. 创建虚拟环境（推荐）
python3.11 -m venv .venv
source .venv/bin/activate

# 3. 安装 PyTorch（按机器 CUDA 版本选，示例 CUDA 12.1）
pip install --upgrade pip
pip install "torch>=2.3" numpy matplotlib

# 4. 第 1–3 章裸 PyTorch 版到此即可；SDK 版再装：
pip install nvidia-physicsnemo nvidia-physicsnemo.sym hydra-core

# 5. 自检
python scripts/check_env.py
```

**预期**：至少 Python、PyTorch、CUDA 为 ✅；PhysicsNeMo 为可选 ✅（跑 SDK 脚本时需要）。

---

## B.4 Docker / NGC（推荐：减少版本冲突）

```bash
docker pull nvcr.io/nvidia/pytorch:24.10-py3
docker run --gpus all -it --rm -v "$PWD":/workspace \
  nvcr.io/nvidia/pytorch:24.10-py3 bash
cd /workspace/physicsnemo-from-zero-to-one
pip install nvidia-physicsnemo nvidia-physicsnemo.sym hydra-core
python scripts/check_env.py
```

**适用坑**：`import physicsnemo.sym` 失败、PyTorch 与 CUDA 版本不匹配（见附录 C）。

---

## B.5 Google Colab（免费 T4）

**推荐：直接打开配套笔记本**

1. 打开 [Google Colab](https://colab.research.google.com/)
2. **文件 → 在 GitHub 中打开**，粘贴：  
   `https://github.com/binbinao/physicsnemo-from-zero-to-one/blob/main/notebooks/colab_quickstart.ipynb`
3. **运行时 → 更改运行时类型 → T4 GPU**
4. 从上到下运行所有单元格（含 ch01 PINN + ch04 FNO mini）

**手动单行版**（不用笔记本时）：

```python
!git clone --depth 1 https://github.com/binbinao/physicsnemo-from-zero-to-one.git
%cd physicsnemo-from-zero-to-one
!pip install -q numpy matplotlib hydra-core
!python scripts/check_env.py
!python ch01_hello/pinn_spring.py --epochs 800
%cd ch04_fno_airfoil && python train_fno_mini.py --epochs 30
```

**Colab 注意**：会话约 12h 断开；大文件用 Google Drive 挂载（笔记本末 cell 有示例）。详见 [CLOUD_GPU_GUIDE §2](../docs/CLOUD_GPU_GUIDE.md#2-google-colab逐步)。

---

## B.6 Mac（Apple Silicon）说明

- **M1/M2/M3**：可跑第 1–3 章 CPU/MPS 版；PhysicsNeMo 官方以 CUDA 为主，**不建议**在 Mac 上强跑 SDK/分布式章节。  
- 第 4 章及以后请用云 GPU 或 Linux + NVIDIA 显卡。

---

## B.7 磁盘与数据

- 本书示例多为**合成数据**，无需大型下载。  
- 建议预留 **≥30GB**：镜像、pip 缓存、`outputs/`、checkpoint。  
- `.gitignore` 已忽略 `*.pt`、`outputs/`，训练产物不会进 Git。

---

## B.8 安全提醒

- 勿在云端 notebook 中粘贴 API Key、公司 VPN 密码。  
- 公网实例训完删除或关机，避免按量账单累积。

---

## B.9 AutoDL 五步开机（国内）

1. [autodl.com](https://www.autodl.com/) 租用 **PyTorch 2.x + CUDA 12** 镜像 + RTX 4090 / A100  
2. 打开 **JupyterLab** 或 SSH 终端  
3. `git clone` 本仓库并 `pip install`（同 B.3）  
4. `python scripts/check_env.py` 确认 CUDA ✅  
5. 训完 **关机** 停止计费  

分步截图级说明：[CLOUD_GPU_GUIDE §3](../docs/CLOUD_GPU_GUIDE.md#3-autodl国内常用)

---

## B.10 各章在云上的首选命令

| 章 | 命令 |
|:---|:---|
| ch01 | `python ch01_hello/pinn_spring.py --epochs 1000` |
| ch02 | `python ch02_heat1d/heat1d_pinn_raw.py` |
| ch04 | `cd ch04_fno_airfoil && python train_fno_mini.py --epochs 50` |
| ch05 | 先确保 ch04 存在；`cd ch05_darcy_hybrid && python train_data_fno.py` |
| ch06 | `cd ch06_fourcastnet_mini && python train_afno_mini.py --epochs 30` |
| ch07 | `cd ch07_drivaernet_optim && python train.py` |

显存与时长表：[HARDWARE_EXPECTATIONS.md](../docs/HARDWARE_EXPECTATIONS.md)

---

➡️ **附录 A**：[数学速查](appendix_a_math.md)  
➡️ **附录 C**：[常见踩坑 50 问](appendix_c_troubleshooting.md)  
➡️ **附录 D**：[PyTorch 最小集](appendix_d_pytorch_mini.md)

---

*附录 B · v1.0 · 更新：2026-05-15*
