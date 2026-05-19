# 云 GPU 与 Colab 实操指南

> 无本地 NVIDIA GPU 时的完整上手路径。附录 B 为速查版；本文档为**分步操作版**。  
> 配套 Colab：[`notebooks/colab_quickstart.ipynb`](../notebooks/colab_quickstart.ipynb)

---

## 1. 选型决策（30 秒）

| 你的情况 | 推荐 |
|:---|:---|
| 只想先跑通、零成本 | **Google Colab**（免费 T4，会话有限） |
| 国内网络、按小时付费、学生 | **AutoDL** |
| 公司已有阿里云 / 腾讯云 | **PAI-DSW** / **GPU 云服务器** |
| 要跑 ch6 多卡 DDP | 云主机 **≥2×GPU** 或单机长时间 A100 |

---

## 2. Google Colab（逐步）

### 2.1 打开 GPU 运行时

1. 打开 [colab.research.google.com](https://colab.research.google.com/)
2. **文件 → 上传笔记本**，选择仓库中的 `notebooks/colab_quickstart.ipynb`  
   或：**文件 → 在 GitHub 中打开**，粘贴  
   `https://github.com/binbinao/physicsnemo-from-zero-to-one/blob/main/notebooks/colab_quickstart.ipynb`
3. **运行时 → 更改运行时类型 → T4 GPU → 保存**

### 2.2 验证 GPU

在首个代码单元运行后应看到 `CUDA available: True` 及 GPU 名称。

### 2.3 按章节运行

| 章节 | Colab 命令（在仓库根目录） | 预计时长 (T4) |
|:---|:---|:---|
| ch01 | `!python ch01_hello/pinn_spring.py --epochs 500` | ~2 min |
| ch02 | `!python ch02_heat1d/heat1d_pinn_raw.py` | ~5–10 min |
| ch04 | `!python ch04_fno_airfoil/train_fno_mini.py --epochs 30` | ~10–20 min |
| ch06 | `!python ch06_fourcastnet_mini/train_afno_mini.py --epochs 20` | ~15–30 min |

### 2.4 Colab 限制与对策

| 问题 | 对策 |
|:---|:---|
| 会话 ~12h 断开 | 定期保存 checkpoint 到 Google Drive：`from google.colab import drive; drive.mount('/content/drive')` |
| 磁盘 ~100GB 临时 | 勿在 Colab 存大量 `outputs/`；训完 `download` 或拷到 Drive |
| 免费版算力排队 | 改 Colab Pro，或换 AutoDL |
| `physicsnemo` 安装慢 | 笔记本中已用 `-q`；失败则分 cell 重装并重启运行时 |

---

## 3. AutoDL（国内常用）

### 3.1 创建实例

1. 登录 [autodl.com](https://www.autodl.com/)
2. **租用实例** → 选 **PyTorch 2.x + CUDA 12.x** 镜像（如 `PyTorch 2.3.0 / 12.1`）
3. GPU：RTX 4090（24GB）或 A100（按预算）；**按量计费**
4. 开机后记下 **SSH** 或 **JupyterLab** 入口

### 3.2 JupyterLab 一键环境

在终端或 notebook 中：

```bash
cd ~
git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
cd physicsnemo-from-zero-to-one
pip install -U pip
pip install "torch>=2.3" numpy matplotlib hydra-core
pip install nvidia-physicsnemo nvidia-physicsnemo.sym  # SDK 章节需要
python scripts/check_env.py
```

### 3.3 跑第 4 章 FNO

```bash
cd ch04_fno_airfoil
python train_fno_mini.py --epochs 50
ls -la data/   # 应生成 darcy_data.pt
```

### 3.4 省钱

- 训完 **关机**（保留镜像不保留 GPU 计费）
- 用 **学术优惠** / 充值活动
- 微缩版 epoch 先跑通再加大

---

## 4. 阿里云 PAI-DSW（简述）

1. 控制台开通 **PAI-DSW**，创建 **GPU 工作空间**（选 A10 / V100 等）
2. 打开 DSW 内置 Terminal，执行与 §3.2 相同的 clone + pip + `check_env`
3. 数据持久化用 **OSS** 挂载；大数据集放 OSS，本地软链

---

## 5. 腾讯云 GPU 云服务器（简述）

1. 购买 **GPU 云服务器**（GN 系列），镜像选 **Ubuntu 22.04 + GPU 驱动**
2. SSH 登录后安装 Miniconda，再执行 §3.2
3. 安全组放行 SSH（22）；**勿对公网开放任意端口**

---

## 6. 各章最低显存（云选型参考）

| 章节 | 微缩版 | 说明 |
|:---|:---|:---|
| ch01–ch02 | 无 GPU 可跑 | Colab CPU 也行，较慢 |
| ch03 | 4–8 GB | 2D 配点多时占显存 |
| ch04 | 8 GB | `train_fno_mini.py` |
| ch05 | 8 GB | 需先跑过 ch04 |
| ch06 | 8–16 GB | rollout 更长 |
| ch07 | 8 GB（toy） / 24 GB+ | Optuna 多 trial 另计 |

详见 [HARDWARE_EXPECTATIONS.md](HARDWARE_EXPECTATIONS.md)。

---

## 7. 故障排查

| 现象 | 处理 |
|:---|:---|
| `CUDA out of memory` | 减 batch、减分辨率、`--epochs` 先改小 |
| Colab 显示 CPU | 重新选 T4 运行时并 **重新连接** |
| `git clone` 失败 | 用 Gitee 镜像或本地上传 zip |
| Sym 导入失败 | 先 `pip install torch` 再装 `nvidia-physicsnemo.sym` |

更多见 [附录 C](../book/appendix_c_troubleshooting.md)。

---

## 8. 相关链接

- [附录 B（速查）](../book/appendix_b_cloud_gpu.md)
- [Colab 笔记本](../notebooks/colab_quickstart.ipynb)
- [环境自检脚本](../scripts/check_env.py)
