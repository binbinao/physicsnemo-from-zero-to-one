# Appendix B · Cloud GPU Selection & One-Click Environment Setup

> **Purpose**: When you don't have a local NVIDIA GPU or have insufficient VRAM, use a cloud instance to run Chapters 4–7.  
> **5-minute goal**: Get a CUDA-enabled machine, clone the repo, and pass `scripts/check_env.py`.  
> **Step-by-step guide (Colab / AutoDL / Alibaba Cloud)**: See [`docs/CLOUD_GPU_GUIDE.md`](../docs/CLOUD_GPU_GUIDE.md)  
> **One-click Colab notebook**: [`notebooks/colab_quickstart.ipynb`](../notebooks/colab_quickstart.ipynb) (open in Colab directly)

---

## B.1 When Do You Need a Cloud GPU?

| Scenario | Recommendation |
|:---|:---|
| Only studying Chapters 1–2 | Local CPU is sufficient |
| Chapter 3: 2D heat dissipation | 8GB consumer GPU or Colab T4 |
| Chapters 4–7: FNO / training | ≥8GB VRAM; full version recommends 16–24GB |
| Chapter 6: Multi-GPU DDP | Single-node multi-GPU or cloud 8×GPU instance (optional) |

---

## B.2 Domestic (China) Cloud Provider Comparison (2025–2026 Reference)

> Prices and instance types change with promotions. Listed below in order of **ease of use + AI image availability**; verify on official websites before ordering.

| Provider | Typical GPU | Target Users | Advantages | Notes |
|:---|:---|:---|:---|:---|
| **AutoDL** | RTX 4090 / A100 pay-as-you-go | Students, individuals | Ready-to-use PyTorch images, hourly billing | Peak-hour queuing |
| **Alibaba Cloud PAI-DSW** | A10 / V100 / A100 | Enterprises, existing Alibaba Cloud accounts | Good integration with Object Storage/OSS | Many configuration options |
| **Tencent Cloud GPU Instances** | T4 / A10 / A100 | Enterprises | Comprehensive docs, many regions | Need to select CUDA image yourself |
| **Huawei Cloud ModelArts** | Ascend / GPU instances | Government/enterprise, Ascend ecosystem | Domestic compliance | PhysicsNeMo primarily targets NVIDIA CUDA; check Ascend compatibility separately |

**Cost-saving tip**: Start with **pay-as-you-go + pre-installed PyTorch 2.x + CUDA 12** images; shut down immediately after training.

---

## B.3 Five-Minute Environment (Cloud Instance / Local Linux)

```bash
# 1. Clone the repository
git clone https://github.com/binbinao/physicsnemo-from-zero-to-one.git
cd physicsnemo-from-zero-to-one

# 2. Create a virtual environment (recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (select based on your CUDA version; example: CUDA 12.1)
pip install --upgrade pip
pip install "torch>=2.3" numpy matplotlib

# 4. For Chapters 1–3 bare PyTorch, you're done here; for SDK version, also install:
pip install nvidia-physicsnemo nvidia-physicsnemo.sym hydra-core

# 5. Self-check
python scripts/check_env.py
```

**Expected result**: At minimum Python, PyTorch, and CUDA should show ✅; PhysicsNeMo is an optional ✅ (required when running SDK scripts).

---

## B.4 Docker / NGC (Recommended: Reduces Version Conflicts)

```bash
docker pull nvcr.io/nvidia/pytorch:24.10-py3
docker run --gpus all -it --rm -v "$PWD":/workspace \
  nvcr.io/nvidia/pytorch:24.10-py3 bash
cd /workspace/physicsnemo-from-zero-to-one
pip install nvidia-physicsnemo nvidia-physicsnemo.sym hydra-core
python scripts/check_env.py
```

**Common pitfall this solves**: `import physicsnemo.sym` failures, PyTorch/CUDA version mismatches (see Appendix C).

---

## B.5 Google Colab (Free T4)

**Recommended: Open the companion notebook directly**

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Open notebook from GitHub**, paste:  
   `https://github.com/binbinao/physicsnemo-from-zero-to-one/blob/main/notebooks/colab_quickstart.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. Run all cells from top to bottom (includes ch01 PINN + ch04 FNO mini)

**Manual single-line version** (when not using the notebook):

```python
!git clone --depth 1 https://github.com/binbinao/physicsnemo-from-zero-to-one.git
%cd physicsnemo-from-zero-to-one
!pip install -q numpy matplotlib hydra-core
!python scripts/check_env.py
!python ch01_hello/pinn_spring.py --epochs 800
%cd ch04_fno_airfoil && python train_fno_mini.py --epochs 30
```

**Colab notes**: Sessions disconnect after ~12h; for large files, mount Google Drive (example in the last cell of the notebook). See [CLOUD_GPU_GUIDE §2](../docs/CLOUD_GPU_GUIDE.md#2-google-colab逐步) for details.

---

## B.6 Mac (Apple Silicon) Notes

- **M1/M2/M3**: Can run Chapters 1–3 in CPU/MPS mode; PhysicsNeMo officially targets CUDA, so running SDK/distributed chapters on Mac is **not recommended**.  
- For Chapter 4 onwards, use a cloud GPU or Linux + NVIDIA GPU.

---

## B.7 Disk & Data

- Most examples in this book use **synthetic data** and require no large downloads.  
- Recommend reserving **≥30GB**: images, pip cache, `outputs/`, checkpoints.  
- `.gitignore` already excludes `*.pt` and `outputs/`; training artifacts won't enter Git.

---

## B.8 Security Reminders

- Do not paste API keys or corporate VPN passwords in cloud notebooks.  
- Delete or shut down public-facing instances after training to avoid accumulating pay-as-you-go charges.

---

## B.9 AutoDL Five-Step Startup (China)

1. Rent a **PyTorch 2.x + CUDA 12** image + RTX 4090 / A100 on [autodl.com](https://www.autodl.com/)  
2. Open **JupyterLab** or SSH terminal  
3. `git clone` this repo and `pip install` (same as B.3)  
4. `python scripts/check_env.py` to confirm CUDA ✅  
5. **Shut down** after training to stop billing  

Step-by-step screenshots: [CLOUD_GPU_GUIDE §3](../docs/CLOUD_GPU_GUIDE.md#3-autodl国内常用)

---

## B.10 Recommended Commands per Chapter on Cloud

| Chapter | Command |
|:---|:---|
| ch01 | `python ch01_hello/pinn_spring.py --epochs 1000` |
| ch02 | `python ch02_heat1d/heat1d_pinn_raw.py` |
| ch04 | `cd ch04_fno_airfoil && python train_fno_mini.py --epochs 50` |
| ch05 | Ensure ch04 exists first; `cd ch05_darcy_hybrid && python train_data_fno.py` |
| ch06 | `cd ch06_fourcastnet_mini && python train_afno_mini.py --epochs 30` |
| ch07 | `cd ch07_drivaernet_optim && python train.py` |

VRAM and runtime estimates: [HARDWARE_EXPECTATIONS.md](../docs/HARDWARE_EXPECTATIONS.md)

---

➡️ **Appendix A**: [Math Quick Reference](appendix_a_math.md)  
➡️ **Appendix C**: [50 Common Pitfalls](appendix_c_troubleshooting.md)  
➡️ **Appendix D**: [PyTorch Minimal Subset](appendix_d_pytorch_mini.md)

---

*Appendix B · v1.0 · Updated: 2026-05-15*
