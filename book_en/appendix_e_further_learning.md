# Appendix E · After the Book: Further Learning Roadmap

> **Purpose**: After finishing the Preface + Chapters 1–7, use this appendix for a capability check and to choose your next path.  
> **Companion checklist** (tick-boxes, 30-day plan): [docs/WHATS_NEXT.md](../docs/WHATS_NEXT.md).  
> **Not** a second textbook; recommendations are ordered “use first → deepen → production.”

---

## E.0 Book Review Snapshot: What You Already Covered

This book’s spine is not a pile of model names, but a deliverable AI4Science skill stack:

| Stage | Chapters | Skills you practiced |
|:---|:---|:---|
| Motivation & map | Preface | AI4Science value prop, PhysicsNeMo positioning, hardware expectations |
| Coordinates → derivatives → residuals | ch01–ch02 | ODE/PDE PINNs, loss weights, Hydra experiments |
| Industrial geometry & inverse problems | ch03 | Multiple BCs, Domain/Constraint, inverse intuition |
| Framework switch + operator learning | ch04 | `physicsnemo-sym` → main framework, FNO training loop |
| Data + physics | ch05 | Hybrid losses, λ_physics trade-offs |
| Spatiotemporal & scale | ch06 | AFNO/autoregression, short rollouts, distributed intro |
| Closed-loop delivery | ch07 | Surrogate → optimize → ONNX/API; V&V awareness; UQ concepts |

**Deliberate teaching trade-offs (review conclusions)**:

1. **Default data is mostly synthetic / toy**: keeps CPU/small-GPU reproducibility; full CFD meshes and production datasets are post-book projects.  
2. **Narrative scene ≠ default script** (esp. ch04 airfoil narrative / Darcy default; ch07 “custom PDE” methodology / DrivAerNet directory name): lowers onboarding friction, but requires reading `CH04_GUIDE` and chapter “read-first” notes.  
3. **UQ, full Triton production, multiphysics coupling, unstructured-mesh GNNs**: concepts or Failure notes appear in the text, **not** as required runnable labs—these are the main advanced battlefield.  
4. **Safety-critical boundaries are explicit**: see [SAFETY_CRITICAL_LIMITATIONS.md](../docs/SAFETY_CRITICAL_LIMITATIONS.md); surrogates default to “screen + verify,” not final sign-off substitutes.

If your capability matrix still has ❌ items, return to [WHATS_NEXT §0](../docs/WHATS_NEXT.md) and the matching cheatsheet before advancing here.

---

## E.1 Consolidate First: “Can Modify and Explain”

Before jumping to new architectures, spend **2–3 weeks** on three things (more valuable than another buzzword):

| Action | Acceptance | Resource |
|:---|:---|:---|
| Re-run with the command sheet | Finish ch01→ch04 minimal path without hunting the book | [COMMAND_REFERENCE](../docs/COMMAND_REFERENCE.md) |
| Compare to baseline | Your loss/metrics land in a sane band | [results/BASELINE.md](../results/BASELINE.md) |
| Explain the framework switch | 3 minutes on when to use sym vs main | [FRAMEWORK_SWITCH](../docs/FRAMEWORK_SWITCH.md) |
| Explain ch04 scope | What airfoil narrative vs Darcy default each serves | [CH04_GUIDE](../ch04_fno_airfoil/CH04_GUIDE.md) |

Still stuck on environment: Appendix B + [ENVIRONMENT.md](../docs/ENVIRONMENT.md); still stuck on PyTorch: Appendix D.

---

## E.2 Choose One Path (Single-Focus)

> **Rule**: chasing two paths at once usually finishes neither. Pick one primary path for 30–90 days.

### Path α · Official ecosystem reproduction (steadiest)

Migrate book skills onto NVIDIA official examples, then adapt to your PDE.

| Priority | Suggestion | Links to this book |
|:---|:---|:---|
| 1 | PhysicsNeMo docs + Darcy/CFD examples kin to ch04/ch05 | FNO, hybrid loss |
| 2 | `examples/weather/` style AFNO / FCN | ch06 |
| 3 | Deployment examples (ONNX / serving) | ch07 |

- Repo: https://github.com/NVIDIA/physicsnemo  
- Docs: https://docs.nvidia.com/deeplearning/physicsnemo/

### Path β · Real-data project (closest to industrial pain)

| Domain | Dataset / direction | Start from |
|:---|:---|:---|
| Airfoil / RANS | AirfRANS | ch04 narrative |
| Automotive aero | Full DrivAerNet (not the toy) | ch07 |
| Weather | ERA5 / ARCO-ERA5; GraphCast / FourCastNet papers | ch06 |
| Porous media | Open Darcy benchmarks at higher resolution | ch05 |

**Discipline**: keep the book’s synthetic pipeline as a debug baseline, then swap in real data—data plumbing often costs more than tuning the model.

### Path γ · CAE / sign-off & V&V (senior engineers)

| Topic | Where |
|:---|:---|
| Sign-off workflow entry | [START_HERE_CAE.md](../docs/START_HERE_CAE.md) |
| Acceptance report template | [VV_REPORT_TEMPLATE.md](../docs/VV_REPORT_TEMPLATE.md) |
| Surrogate decision tree | [CAE_SURROGATE_DECISION_TREE.md](../docs/CAE_SURROGATE_DECISION_TREE.md) |
| Industry standards | ASME V&V 40; [NAFEMS](https://www.nafems.org/) AI/ML sessions |

Target skill: write scope / failure modes / high-fidelity review strategy—not only a pretty contour plot.

### Path δ · Research deepening (papers & theory)

For readers aiming at papers or method innovation. Suggested order:

1. **PINN failure modes & fixes**: NTK / causal training / adaptive weights (listed in ch02 further reading)  
2. **Unified neural operators**: Kovachki et al., *Neural Operator*, JMLR 2023  
3. **Physics-informed operators**: PINO / physics-informed FNO (builds on ch05)  
4. **UQ**: Deep Ensemble, MC-Dropout (ch07 §7.10 introduces; implement yourself)

---

## E.3 Method Families Beyond This Book

Ordered **near → far** from the book’s content. Stay near first.

| Direction | Why | Entry anchors |
|:---|:---|:---|
| **DeepONet / Branch-Trunk** | Non-uniform sensors, multi-resolution inputs | Lu et al., *Nat. Mach. Intell.* 2021; PhysicsNeMo DeepONet examples |
| **Geo-FNO / deformed-domain operators** | Push FNO beyond regular grids | Li et al. Geo-FNO series |
| **GNN / MeshGraphNet-style** | Unstructured meshes, industrial STL/volume meshes | Pfaff et al., *ICML* 2021; common for auto/structures |
| **Transformer / AFNO variants** | Long-range dependence, weather & multiscale fields | Builds on ch06; GraphCast, Pangu-Weather |
| **Diffusion / generative physics models** | Uncertainty sampling, super-resolution, conditional generation | PhysicsNeMo diffusion-related examples (check current official docs) |
| **Multiphysics / CHT** | Real products are almost always coupled fields | Extend ch03 heat-sink narrative to CHT / thermal-fluid |
| **Bayesian / multi-objective optimization** | Design search beyond Optuna grids | BoTorch; builds on ch07 `optimize.py` |
| **Active learning + UQ-driven sampling** | Fewer high-fidelity labels | Builds on ch07 §7.10 “high uncertainty → CFD queue” |

> ⚠️ **Selection tip**: discretization drives representation. Regular grids → FNO family; unstructured → GNN/point-cloud; sparse sensors → DeepONet. Do not force the newest paper onto the wrong discrete representation.

---

## E.4 Engineering Next Steps: From Demo to Deliverable

| Layer | Learn next | Book foundation |
|:---|:---|:---|
| Export & check | ONNX numeric parity, input schema, version pinning | ch07 `export_onnx.py` |
| Serving | Triton model repo, dynamic batching, GPU locks | ch07 further reading + [CAE_DEPLOYMENT_NOTES](../docs/CAE_DEPLOYMENT_NOTES.md) |
| Observability & feedback | Prediction logs, drift monitors, failed-sample retrain | ch07 afterword “verification loop” |
| Constraints & manufacturability | Design constraint checks, units/nondim discipline | [CAE_OPTIMIZATION_CONSTRAINTS](../docs/CAE_OPTIMIZATION_CONSTRAINTS.md) · [CAE_UNITS_AND_NONDIM](../docs/CAE_UNITS_AND_NONDIM.md) |
| Safety boundaries | When surrogate-only sign-off is forbidden | [SAFETY_CRITICAL_LIMITATIONS](../docs/SAFETY_CRITICAL_LIMITATIONS.md) |

**Minimal production checklist (post-book project)**:

- [ ] Holdout / extrapolation-split tests  
- [ ] Physics sanity checks (conservation, BCs, dimensions)  
- [ ] Uncertainty visible—or at least ensemble variance  
- [ ] Top-K designs pass one high-fidelity review  
- [ ] API/service has input validation and model version IDs  

---

## E.5 Curated Further Reading

### E.5.1 Must-read papers (≤3 per path at first)

| Topic | Reference |
|:---|:---|
| Classic PINN | Raissi, Perdikaris, Karniadakis, *JCP* 2019 |
| PIML panorama | Karniadakis et al., *Nat. Rev. Phys.* 2021 |
| FNO | Li et al., *ICLR* 2021 |
| Neural operators | Kovachki et al., *JMLR* 2023 |
| Weather AI | Lam et al., GraphCast, *Science* 2023; Pathak et al., FourCastNet |
| Practical UQ | Lakshminarayanan et al., Deep Ensembles, *NeurIPS* 2017 |

Chapter “Further reading” sections are topic-split; this table is the **cross-chapter shortlist**.

### E.5.2 Courses & public materials

| Type | Suggestion |
|:---|:---|
| Official | NVIDIA PhysicsNeMo / Modulus tutorials and NGC container docs |
| DL refresh | Official PyTorch tutorials (after Appendix D) |
| Numerics refresh | Any CFD/FEM intro (mesh, residual, convergence—the surrogate’s “teacher”) |
| Community | PhysicsNeMo GitHub Discussions; this book’s Issues / PRs |

### E.5.3 Second layer inside this repository

| Resource | Who it helps |
|:---|:---|
| [docs/START_HERE_CAE.md](../docs/START_HERE_CAE.md) and `CAE_*` docs | Connecting demos to CAE workflows |
| [docs/CAE_CLOSED_LOOP_DEMO.md](../docs/CAE_CLOSED_LOOP_DEMO.md) | Closed-loop storyline |
| `tools/cfd_batch` notes | Batch CFD / joint inverse extensions |
| Each chapter’s `*_gpu.py` + DDP | Moving from single-GPU to multi-GPU |

---

## E.6 30 / 90-Day Action Templates

### 30 days (pick one path)

```text
Week 1–2  Reproduce an official example (same family as ch04 or ch07)
Week 3    Attach one small real dataset (preferably <10GB)
Week 4    ONNX parity or CAE/script integration; write a one-page V&V note
```

### 90 days (ship a minimal “Your Problem” loop)

```text
Month 1  Problem definition + data contract + reproducible synthetic baseline
Month 2  Real data / hybrid physics + error-report template
Month 3  Optimize or serve + Top-K high-fidelity review + failure-case writeup
```

Tick-box detail: [docs/WHATS_NEXT.md](../docs/WHATS_NEXT.md). If you are still mid–6-week read-through, finish [STUDY_PLAN_6WEEKS.md](../docs/STUDY_PLAN_6WEEKS.md) first.

---

## E.7 One-Line Advice by Reader Type

| You are | Prioritize after the book |
|:---|:---|
| CAE / simulation engineer | Paths γ + β: real geometry + V&V; chase fewer new architectures |
| DL-background researcher | Paths δ + α: PINO/UQ/operator unification, then real meshes |
| Student / career switcher | Path α: official example reproduction, then one small real-data project |
| Tech lead | Path γ: standards, safety bounds, deployment & data feedback; use ch07 for an internal talk |

---

## E.8 How This Appendix Relates to Other Docs

| Document | Role |
|:---|:---|
| **This Appendix E** | In-book review conclusions + path choice + method map |
| [WHATS_NEXT.md](../docs/WHATS_NEXT.md) | Repo-side checklist and short action table |
| [STUDY_PLAN_6WEEKS.md](../docs/STUDY_PLAN_6WEEKS.md) | Week plan for reading this book (before/during) |
| Chapter “Further reading” | Papers and official examples for that chapter |

> Models change; SDKs change. What this book aims to leave you with is the ability to string **physics + data + model + optimization + verification + deployment** into a solution.  
> The keyword for the next stage is only one: **Your Problem**.

---

*Appendix E · v1.0 · Updated: 2026-08-17*
