# Appendix C · 50 Common Pitfalls & FAQ

> **Purpose**: Quick lookup for environment, training, accuracy, and engineering expectation issues.  
> Grouped by topic; use your editor's search to find keywords.

---

## C.1 Environment & Installation (1–12)

**1. `pip install nvidia-physicsnemo` succeeds but `import physicsnemo` fails?**  
Verify Python ≥3.10 and that `pip` uses the same interpreter: `python -c "import physicsnemo"`.

**2. `physicsnemo.sym` not found?**  
You need an additional install: `pip install nvidia-physicsnemo.sym`.

**3. CUDA available: False?**  
Driver not installed, PyTorch is the CPU-only version, or the cloud instance wasn't provisioned with a GPU. Cross-check `nvidia-smi` with `torch.cuda.is_available()`.

**4. CUDA version doesn't match PyTorch?**  
Go to [pytorch.org](https://pytorch.org) and select the install command matching your cu12x version, or switch to the NGC container (Appendix B).

**5. Which Python version should I use?**  
This book targets **Python 3.10+ (3.11 recommended)**; `scripts/check_env.py` will error out below 3.10. See [ENVIRONMENT.md](../docs/ENVIRONMENT.md).

**6. Can I run this on Windows?**  
Chapters 1–2 with bare PyTorch generally work; for the SDK and distributed training, **WSL2 + Ubuntu** or a Linux cloud instance is recommended.

**7. Can Mac M-series run the entire book?**  
Chapters 1–3 work on CPU/MPS; from Chapter 4 onwards, use a cloud GPU (Appendix B).

**8. `check_env` is all green but SDK scripts still error?**  
The SDK may depend on SymPy, Hydra, etc.; install what the error message says with `pip install`, or check the chapter's `README`.

**9. Hydra reports `Config not found`?**  
You must run from the corresponding chapter directory, or specify `--config-path` / `--config-name` (see that chapter's `conf/`).

**10. Docker can't see the GPU?**  
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and add `--gpus all` to `docker run`.

**11. Disk full?**  
Delete each chapter's `outputs/` and `data/*.pt` (regenerable), and clean `~/.cache/pip`.

**12. Corporate intranet can't access GitHub?**  
Use a mirror to clone or copy the repo offline; PyTorch/PhysicsNeMo requires an internal PyPI mirror or pre-downloaded wheels.

---

## C.2 Training & Loss (13–28)

**13. PDE loss stuck at 1e-1?**  
Check `create_graph=True`, learning rate, number of collocation points, and loss weights (are IC/BC weights too small?).

**14. One of the three loss components isn't decreasing?**  
Plot individual component curves; usually the BC weight is too low or boundary sampling is too sparse.

**15. Loss becomes NaN?**  
Lower the learning rate; check for division by zero or exp overflow; if using mixed precision, disable AMP first to diagnose.

**16. Training is very slow?**  
Reduce `n_pde` or network width; confirm you're not accidentally using CPU; use the `*_gpu.py` version for GPU.

**17. Output plot is a straight line / constant?**  
The network hasn't learned anything; check input normalization, output scale, and whether you forgot `model.train()`.

**18. MLP works well but PINN performs poorly?**  
This is one of the normal scenarios: PINNs are more sensitive to hyperparameters; first ensure quantities are on the same scale before comparing.

**19. Can't reproduce the exact loss values from the book?**  
Different random seeds, hardware, and PyTorch versions cause numerical differences; check whether the **trend** is consistent (see C.4 on timing expectations).

**20. How many epochs should I train?**  
Use the validation curve plateau as your guide; book demos typically run hundreds to a few thousand epochs.

**21. Are more collocation points always better?**  
Diminishing returns; for 2D problems, start with 2k–10k interior points.

**22. Non-uniform boundary sampling?**  
Complex geometries need denser sampling at corners and interfaces; the Sym version uses constraint density parameters.

**23. FNO data loss isn't decreasing?**  
Check data shape `(B,C,H,W)`, learning rate, and whether wrong normalization is being used.

**24. ch05 reports `No module named fno_model`?**  
You must clone the complete repository and **do not delete ch04**; ch05 depends on ch04's `fno_model.py`.

**25. Where does `darcy_data.pt` come from?**  
The first time you run the ch04/ch05 training script, it auto-generates in `data/`.

**26. Multi-GPU DDP hangs?**  
Check `torchrun` arguments and `MASTER_ADDR`; ensure firewalls allow the multi-node ports.

**27. Optuna HPO too slow?**  
Reduce `n_trials`, narrow the search space; get single-GPU training working first before HPO.

**28. ONNX export fails?**  
Confirm `model.eval()`, fixed input shape, and that the opset version matches the deployment target.

---

## C.3 PhysicsNeMo & Framework (29–38)

**29. What's the difference between sym and the main framework?**  
sym: PINNs, symbolic PDEs; main: FNO/AFNO, data-driven training. From Chapter 4 onwards, the main framework is primary.

**30. Can I run FNO using only sym?**  
Not recommended; FNO is maintained in the `physicsnemo.models` main framework.

**31. What's the relationship between Modulus and PhysicsNeMo?**  
Modulus was the old name; it has been unified as PhysicsNeMo v2.

**32. API differs from older online tutorials?**  
Use this repo's `*_sdk.py` files and the [official documentation](https://github.com/NVIDIA/physicsnemo) as references; v2 has breaking changes.

**33. Is SymPy mandatory?**  
The bare PyTorch version doesn't need it; the SDK sym version uses SymPy to define PDEs.

**34. Geometry CSG errors?**  
Check whether primitives intersect and parameter ranges; see ch03 `heat_sink_geometry.py`.

**35. Difference between Validator and Constraint?**  
Constraint participates in training loss; Validator only evaluates and does not participate in backpropagation.

**36. How to set weights `w_pde`, `w_ic`?**  
Start with the same order of magnitude; BC/IC often need higher weights (e.g., 10×).

**37. Can I use DeepXDE instead?**  
For learning principles, yes; this book's code and industrial deployment path are designed around PhysicsNeMo.

**38. Relationship between official examples and this book?**  
This book is "build from scratch + three-tier comparison"; official examples are better for looking up API details.

---

## C.4 Accuracy, Timing & Expectations (39–45)

### Are the Timing Comparisons I See Trustworthy?

**39. Is the book's claim "ANSYS 1 hour vs. PINN 0.1 seconds" credible?**  
**They compare different stages**: the traditional side refers to a **single full-precision simulation**; the AI side refers to **inference after training is complete**. Training itself often takes **30 minutes to several hours** (depending on mesh size and data volume). Do not interpret "0.1s inference" as "results from scratch in 0.1s."

**40. Where do the 10–60 minutes in the table come from?**  
Industry experience ranges, depending on mesh, physics fields, and solver settings; **not benchmark measurements from this book**.

**41. Can PINNs replace ANSYS?**  
Generally **not a full replacement**; suitable for surrogate models, rapid parameter sweeps, and inverse problems. Sign-off-grade results typically still require spot-checking with traditional solvers.

**42. What if AI results differ significantly from CFD?**  
Examine relative error distribution, conservation quantities, and boundary layers; if necessary, add physics loss, more data, or grid-aligned training.

**43. How is "1000 operating conditions in 5 minutes" calculated?**  
Assumes **the model is already trained**; 1000 forward inference passes are batched; does not include training or data preparation.

**44. Is "digital twin <100ms" realistic?**  
Achievable with small models + GPU/TensorRT; requires dedicated inference optimization and SLA testing.

**45. How to explain ROI to management?**  
Emphasize reducing **wait times and number of parameter sweeps**, not claiming "lower solver error rate"; list one-time training cost vs. per-simulation labor savings.

---

## C.5 Data, Deployment & Miscellaneous (46–50)

**46. Can I train without real data?**  
All 7 chapters in this book can use synthetic data; production projects should gradually incorporate real simulation/experimental data.

**47. Can a model trained on synthetic data go to production?**  
Domain adaptation and validation are needed; synthetic data is for **learning the workflow**, not for direct sign-off.

**48. What should I watch out for with FastAPI deployment?**  
`model.eval()`, input validation, concurrency and GPU locking; for production, Triton is more appropriate (ch07 further reading).

**49. Book images not displaying?**  
Figures are in [`book/assets/`](../book/assets/README.md); you can regenerate them using scripts in `book/scripts/`.

**50. Still have questions?**  
Open an issue on [GitHub Issues](https://github.com/binbinao/physicsnemo-from-zero-to-one/issues) with your `check_env` output, complete error message, chapter, and command.

---

➡️ **Appendix A**: [Math Quick Reference](appendix_a_math.md)  
➡️ **Appendix B**: [Cloud GPU & Environment](appendix_b_cloud_gpu.md)  
➡️ **Appendix D**: [PyTorch Minimal Subset](appendix_d_pytorch_mini.md)

---

*Appendix C · v1.0 · Updated: 2026-05-15*
