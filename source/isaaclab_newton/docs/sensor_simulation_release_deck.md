---
marp: true
theme: default
paginate: true
size: 16:9
header: "Sensor Simulation — Isaac for Healthcare"
footer: "NVIDIA Healthcare — Holoscan Team"
style: |
  section { font-size: 22px; }
  h1 { color: #76b900; }
  h2 { color: #76b900; border-bottom: 2px solid #76b900; padding-bottom: 4px; }
  table { font-size: 18px; }
  .done { color: #2e7d32; font-weight: 600; }
  .partial { color: #ef6c00; font-weight: 600; }
  .planned { color: #b71c1c; font-weight: 600; }
  .small { font-size: 16px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Sensor Simulation
## X-Ray–Guided Robotic Catheter Interventional System

**Release Status & Roadmap**

Isaac for Healthcare · NVIDIA Healthcare — Holoscan Team

---

## Agenda

1. **Executive Snapshot** — where we are today
2. **Current Status** — what's running, with numbers
3. **This Release** — features shipping now (Sprint 1)
4. **Next Release** — Sprint 2 priorities
5. **Following Releases** — Sprint 3 & agentic workflow phases
6. **Future Sensors** — extensibility roadmap

---

## Executive Snapshot

- Two sensor modalities **implemented** end-to-end: **X-Ray Fluoroscopy** and **Ultrasound B-Mode**
- Unified Slang GPU rendering backend across both modalities
- **Fused volume + catheter Beer-Lambert** compositing now running at **~25 FPS** on RTX A6000
- **State-based RL pipeline** (PPO, 512 envs) operational
- Three additional sensors — **Endoscopy, CT/CBCT, Force/Torque** — on the planned roadmap
- Critical near-term gap: **multi-env fluoroscopy rendering** (Sprint 2 target)

---

## Sensor Modality Status

| Sensor | Status | Rendering Backend | Differentiable | Multi-Env |
|---|---|---|---|---|
| **X-Ray Fluoroscopy** | <span class="done">Implemented</span> | DiffDRR + Slang (fused volume + catheter Beer-Lambert) | Partial (volume autodiff; catheter forward-only) | Sprint 2 |
| **Ultrasound B-Mode** | <span class="done">Implemented</span> | Slang/CUDA volumetric ray-march + BVH | Yes (6 DOF probe pose) | Sprint 2 |
| Endoscopy / RGB | <span class="planned">Planned</span> | Omniverse RTX renderer | N/A | Via Isaac Lab |
| CT / CBCT | <span class="planned">Planned</span> | Slang ray-cast through μ volume | Possible | TBD |
| Force / Torque | <span class="planned">Planned</span> | Warp contact solver output | Yes (Warp AD) | Yes (batched) |

---

# Part 1 — Current Status

What is built, integrated, and measured today.

---

## Catheter Physics — Three Solver Backends

| Solver | Description | Multi-Env | Key Features |
|---|---|---|---|
| **Production XPBD Cosserat rod** | Production-grade solver | <span class="done">Yes (batched)</span> | Mesh BVH collisions, proximal kinematic control, direct solve |
| Self-contained XPBD solver | Block-Thomas O(n) direct solve | Single env | Warp kernels, floor collision, zero Newton dependency |
| Newton XPBD bridge | External bridge to Newton's XPBD | Single env | Wraps Newton's block-tridiagonal JMJT solver |

All three expose **3D centerline positions in metres**, scaled ×1000 → mm for the compositing pipeline.

---

## Fluoroscopy Rendering — Three Compositing Paths

| Path | Compositing | Performance |
|---|---|---|
| **Slang GPU (unified loop)** | Fused volume + catheter Beer-Lambert in single GPU ray march | **~25 FPS** @ 512×512 (A6000) |
| CPU Beer-Lambert | Per-segment cylinder chord → attenuation map → I = I_DRR × exp(−atten) + scatter + PSF + Poisson | ~2–5 FPS (NumPy) |
| Isaac Lab 3D USD quad | DRR backdrop + capsule markers in Omniverse viewport | Real-time (RTX) |

**Key property:** Beer-Lambert is **multiplicative** — catheter attenuates rather than occludes, with smooth sub-pixel cylinder edges and depth-correct self-crossings.

---

## Beer-Lambert Detector Physics Chain (CPU path)

| # | Step | Purpose |
|---|---|---|
| 1 | Attenuation map: t(d) = 2√(r² − d²) per segment | Exact cylinder chord thickness |
| 2 | I_final = I_DRR × exp(−atten_map) | Beer-Lambert transmission |
| 3 | Veiling glare: σ=18 px Gaussian, 3% re-add | X-ray scatter in tissue + housing |
| 4 | Detector PSF: σ=0.7 px Gaussian | CsI scintillator finite resolution |
| 5 | Poisson noise @ 2000 photons/px | Low-dose pulsed fluoroscopy statistics |

Per-segment **5-zone attenuation profile**: tungsten markers (μ=3.0) → nitinol braid (μ=0.8) → transition → soft polymer (μ=0.15) → platinum tip (μ=5.0).

---

## X-Ray Performance Baseline

| Metric | Target | Current | Notes |
|---|---|---|---|
| Single-env physics FPS | >1,000 Hz | **~1,300 Hz** (20 seg, A6000) | <span class="done">Achieved</span> |
| Slang GPU compositing (fused) | <5 ms @ 512² | ~40 ms (~25 FPS) | Single GPU ray march |
| CPU Beer-Lambert compositing | <2 ms/frame | ~200–500 ms/frame | NumPy; Warp port → 30+ FPS |
| 512-env physics FPS | >60 Hz | <span class="planned">N/A (single env only)</span> | Sprint 2 target |

---

## Ultrasound — What's Implemented

- **Slang ultrasound renderer**: fixed-step volumetric ray-march + BVH-accelerated triangle intersection
- Beer-Lambert attenuation, multi-bounce specular reflection (Phase 2)
- **Differentiable rendering** for 6-DOF probe pose, full PyTorch autograd integration
- Legacy OptiX (C++/CUDA) renderer retained as reference (non-differentiable)

| Metric | Target | Current |
|---|---|---|
| Single-env render (Phase 1) | <5 ms (128 elem, 1024 depth) | ~3–8 ms (est.) |
| Post-processing (PSF + TGC + Hilbert + scan convert) | <2 ms | ~1–3 ms (GPU PyTorch) |
| End-to-end frame | <10 ms (100 Hz) | ~8–20 ms (est.) |
| Backward pass (6 DOF) | <5 ms | ~1–2 ms |

---

## RL Training Pipeline (State-Based)

| Component | Description |
|---|---|
| Environment | Multi-env production solver, proximal push/rotate control, distance-to-target reward |
| RSL-RL wrapper | Standard VecEnv adapter for RSL-RL PPO |
| PPO config | Tuned hyperparameters for catheter navigation |
| Training entry | **512 parallel environments**, 1500 max iterations |
| Smoke test | Validates environment without RL dependencies |

> **State-based only** today. Pixel/fluoroscopy observations land in Sprint 2+.

---

# Part 2 — This Release

Sprint 1 deliverables — **shipping now**.

---

## This Release — X-Ray Sprint 1 (Completed)

| Deliverable | Status |
|---|---|
| Beer-Lambert compositing (physically-correct transmission) | <span class="done">✅ CPU + Slang GPU paths</span> |
| Poisson quantum noise (2000 photons/px) | <span class="done">✅ CPU compositing</span> |
| Veiling glare / scatter (σ=18 px, 3% fraction) | <span class="done">✅ CPU compositing</span> |
| Detector PSF (σ=0.7 px) | <span class="done">✅ CPU compositing</span> |
| Per-segment attenuation profile (5-zone catheter model) | <span class="done">✅ Implemented</span> |
| Cone-beam magnification of projected catheter radius | <span class="done">✅ Implemented</span> |
| Slang GPU fused DRR + catheter single ray march | <span class="done">✅ Implemented</span> |
| Catheter segment data → Slang structured buffer pipeline | <span class="done">✅ Implemented</span> |
| Proximal kinematic control API (push / rotate) | <span class="done">✅ Implemented</span> |
| RL state-based training pipeline (PPO) | <span class="done">✅ Implemented</span> |

---

## This Release — Closed Gaps vs XCATH Requirements

| Capability | Before | This Release |
|---|---|---|
| Catheter Beer-Lambert compositing | Missing | <span class="done">✅ Fused GPU + CPU path</span> |
| Per-segment attenuation profile | Missing | <span class="done">✅ 5-zone model</span> |
| Detector physics (Poisson + scatter + PSF) | Missing | <span class="done">✅ CPU compositing path</span> |
| Cone-beam magnification | Missing | <span class="done">✅ Implemented</span> |
| C-arm projection model | Partial | <span class="done">✅ Standard pinhole + LAO/RAO/CRA/CAU</span> |
| Fused GPU DRR + catheter (single ray march) | N/A | <span class="done">✅ NEW capability</span> |
| State-based RL pipeline | Missing | <span class="done">✅ PPO @ 512 envs</span> |

---

## This Release — Known Carry-Overs

Items **scoped to Sprint 1** but **not delivered** — moved into Sprint 2 / Phase 1 backlog:

| Deliverable | Status | Notes |
|---|---|---|
| Beam hardening (polyenergetic correction) | <span class="planned">❌ Not implemented</span> | Monoenergetic only |
| Vessel mask input + vessel boost (μ × A=8) | <span class="planned">❌ Not implemented</span> | Required for DSA |
| DSA pipeline (4-step subtraction) | <span class="planned">❌ Not implemented</span> | Primary clinical mode |
| Gamma correction (γ=0.8) | <span class="planned">❌ Not implemented</span> | Clinical display TF |
| Misregistration jitter | <span class="planned">❌ Not implemented</span> | Required for realistic DSA |
| Physics-based scatter (3D) | <span class="partial">⚠ Partial — 2D veiling glare only</span> | Realism gap |
| C-arm clinical presets (4 vendors) | <span class="partial">⚠ In docstrings only</span> | Needs registry |

---

# Part 3 — Next Release

Sprint 2 — Multi-Environment, Collision, Image Observations.

---

## Next Release — X-Ray Sprint 2 (Weeks 3–4)

**Theme: Scale & contact realism**

- **Multi-env XPBD solver** — 512+ parallel rods with batched physics
- **SDF / mesh collision** for vessel walls
- **Structured observation dict + reward signals** for RL
- **Multi-env fluoroscopy rendering** — batched Slang dispatch for all envs in one frame
- Target: **>60 Hz @ 512 envs**

**Status today:** production solver supports multi-env; self-contained XPBD remains single-env. Catheter state environment already provides structured obs + rewards for state-based RL.

---

## Next Release — Ultrasound Sprint 1 (Isaac Lab integration)

| Deliverable | Description |
|---|---|
| Ultrasound sensor wrapper | First-class Isaac Lab sensor module |
| CT-to-volume pipeline | HU → acoustic impedance / scattering coefficient mapping |
| Observation dict integration | RL-ready B-mode frames as observations |

---

## Next Release — Phase 1 Fidelity Items (X-Ray)

These are the remaining **simulation fidelity** items, in dependency order:

| Effort | Deliverable |
|---|---|
| ~3 days | Vessel mask input + vessel boost + DSA pipeline (4-step) |
| ~1 day | Gamma correction + scatter convolution + misregistration jitter |
| ~0.5 days | C-arm clinical presets (4 vendors: Philips Azurion, Siemens Artis, GE, Canon) |
| ~1 day | Per-frame μ volume update in cine rendering |
| ~2 days | Bolus tracking Stage 2 (gamma-variate + per-frame μ) |

Without these, the agentic workflow **cannot generate clinically realistic DSA** for sim-to-real transfer.

---

# Part 4 — Following Releases

Sprint 3 + agentic workflow integration.

---

## Sprint 3 — Training Readiness (Weeks 5–6)

**X-Ray and Ultrasound (parallel tracks)**

- **Domain randomization** — probe frequency, attenuation, scattering, noise, C-arm angles
- **Gymnasium wrapper** — standard RL ecosystem compatibility
- **CUDA graph integration** — eliminate per-step launch overhead
- **Differentiable pose optimization** (ultrasound)
- **Automated pytest suite** — regression coverage for both modalities

---

## Agentic Workflow — Phase 2 (Skill Packaging)

Wrap each pipeline stage as a portable **OpenClaw / NemoClaw Skill**:

| Skill | Purpose |
|---|---|
| `patient-digital-twin` | CTA → μ volume + vessel mask + centerline + arrival map |
| `catheter-physics-sim` | Newton rod solver + compositing config |
| `sensor-sim-xray` | DRR / DSA / vessel-boost rendering modes |
| `dataset-creation` | Paired HDF5/WebDataset (frames + pose + GT) |
| `reward-function` | RL reward configuration (target, contact, dose, progress, success) |
| `policy-training` | IL (GR00T-H) → RL (PPO/SAC) → SIL evaluation |
| `evaluation` | Success rate, navigation time, contact force, FID, registration accuracy |

---

## Agentic Workflow — Phase 3 (Agent Integration)

| Week | Deliverable |
|---|---|
| 5–6 | **Skill discovery & chaining** — parse skill definitions, resolve I/O dependencies |
| 6 | **Natural language → config mapping** — *"use Philips Azurion, DSA mode"* → YAML overrides |
| 6–7 | **Iterative refinement loop** — agent runs evaluation, analyzes metrics, proposes changes, re-runs |
| 7 | **Slack / IDE integration** — agent posts progress, visualizations, and final reports |

**Outcome:** experiment cycle compresses from **weeks → hours**, with the agent running continuously.

---

## Following Releases — Remaining Capability Gaps

| Missing Feature | Owning Skill | Why It Matters |
|---|---|---|
| DSA pipeline + vessel boost | `sensor-sim-xray` | Primary clinical rendering mode for training |
| Bolus tracking (VMTK + Dijkstra arrival map) | `sensor-sim-xray` | Enables temporal contrast sequences |
| Per-frame μ update in cine rendering | `sensor-sim-xray` | Catheter advancement + contrast propagation |
| Beam hardening | `sensor-sim-xray` | Closes realism gap vs DeepDRR |
| Max-attenuation volume compositing | `catheter-physics-sim` | True depth-correct volumetric instrument injection |
| **Multi-env fluoroscopy rendering** | `sensor-sim-xray` | Required for batched 512+ env RL training |
| Image-based RL observations | `reward-function`, `policy-training` | Image-guided policies (current env is state-only) |
| Realism metrics (FID, SSIM, vessel visibility) | `evaluation` | Quantitative loop for iterative refinement |

---

# Part 5 — Future Sensors

Beyond X-Ray and Ultrasound.

---

## Future Sensor Roadmap

| Sensor | Priority | Integration Effort | Prerequisite |
|---|---|---|---|
| **Force / Torque** | High | Low — data already in collision solver | Sprint 2 collision integration |
| Endoscopy / RGB | Medium | Low — Isaac Lab camera sensor exists | Realistic vessel interior USD assets |
| IVUS | Medium | Moderate — adapt existing US renderer | High-res vessel wall models |
| CBCT | Low–Medium | Moderate — batched DRR + FDK reconstruction | GPU ray-caster (X-ray future work) |
| Pressure / Flow | Low | High — new hemodynamic solver needed | 1D fluid solver, vessel centerlines |

---

## Summary

**Today**
- X-Ray fluoroscopy + Ultrasound B-Mode implemented end-to-end
- Fused GPU Beer-Lambert at ~25 FPS; physics at ~1,300 Hz
- State-based PPO @ 512 envs operational

**This Release (Sprint 1)**
- Full detector physics chain — Poisson + scatter + PSF + cone-beam
- Slang GPU fused DRR + catheter single-pass ray march
- 5-zone catheter attenuation profile
- 7 of XCATH's required capabilities **closed**

**Next Release (Sprint 2)**
- Multi-env fluoroscopy + SDF/mesh collision
- Image observations for RL
- Ultrasound Isaac Lab sensor module

**Following Releases**
- Sprint 3 training readiness (DR, Gymnasium, CUDA graphs)
- Phase 2 skill packaging → Phase 3 agent integration
- Future sensors: Force/Torque → Endoscopy → IVUS → CBCT → Hemodynamics

---

<!-- _class: lead -->

# Questions?

**Sensor Simulation — Isaac for Healthcare**
NVIDIA Healthcare — Holoscan Team
