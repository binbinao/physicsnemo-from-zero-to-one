# 附录 B · 云 GPU 选型与一键环境

> **用途**：本地无 NVIDIA GPU、或显存不足时，用云主机跑第 4–7 章。  
> **5 分钟目标**：拿到一台带 CUDA 的机器，clone 仓库并跑通 `scripts/check_env.py`。

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

1. 打开 [Google Colab](https://colab.research.google.com/)，运行时 → 更改运行时类型 → **T4 GPU**。  
2. 新建单元格：

```python
!git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
%cd physicsnemo-from-zero-to-one
!pip install -q "torch>=2.3" numpy matplotlib
!python scripts/check_env.py
```

3. 第 4 章起在 Colab 中运行对应 `train_*_mini.py` 或 `train_*_gpu.py`（注意 Colab 磁盘与会话时长）。

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

➡️ **附录 A**：[数学速查](appendix_a_math.md)  
➡️ **附录 C**：[常见踩坑 50 问](appendix_c_troubleshooting.md)
