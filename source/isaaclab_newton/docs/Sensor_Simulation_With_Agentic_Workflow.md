# Sensor Simulation

## X-Ray–Guided Robotic Catheter Interventional System — Isaac for Healthcare

---

## Executive Summary

This document describes the sensor simulation stack for the XCath robotic catheter interventional system.
Two sensor modalities are currently implemented — X-ray fluoroscopy and ultrasound — with a unified
architecture designed to accommodate future sensor types. A new section details the agentic workflow
that orchestrates the full simulation-to-policy pipeline using OpenClaw skills.

| Sensor | Status | Rendering Backend | Differentiable | Multi-Env |
|--------|--------|-------------------|----------------|-----------|
| X-Ray Fluoroscopy | **Implemented** | DiffDRR + Slang (fused volume + catheter Beer-Lambert) | Partial (Slang volume autodiff; catheter forward-only) | No (Sprint 2) |
| Ultrasound B-Mode | Implemented | Slang/CUDA volumetric ray-march + BVH | Yes (6 DOF probe pose) | No (Sprint 2) |
| Endoscopy / RGB | Planned | Omniverse RTX renderer | N/A | Via Isaac Lab |
| CT / CBCT | Planned | Slang ray-cast through μ volume | Possible | TBD |
| Force/Torque | Planned | Warp contact solver output | Yes (Warp AD) | Yes (batched) |

---

## PART 1: CURRENT STATE

### ☢ SENSOR A: X-RAY FLUOROSCOPY

#### 1.1 Catheter Physics Solvers

Three solver implementations exist:

| Solver | Description | Multi-Env | Key Features |
|--------|-------------|-----------|--------------|
| Production XPBD Cosserat rod solver | Production-grade solver | Yes (batched) | Mesh BVH collisions, proximal kinematic control, direct solve |
| Self-contained XPBD solver | Block-Thomas O(n) direct solve | Single env | Warp kernels, floor collision, zero Newton dependency |
| Newton XPBD bridge | External bridge to Newton's XPBD rod solver | Single env | Wraps Newton's block-tridiagonal JMJT solver, exposes (1, N, 3) positions |

All three solvers expose 3D centerline positions in metres, consumed by the compositing pipeline after ×1000 scaling to millimetres.

#### 1.2 Fluoroscopy Rendering

**Two compositing paths are implemented**, plus a legacy opaque overlay:

| Path | Compositing | Performance |
|------|-------------|-------------|
| **Slang GPU (unified sim loop)** | Fused volume + catheter Beer-Lambert in single GPU ray march | **~25 FPS** on RTX A6000 (512×512) |
| **CPU Beer-Lambert** | Per-segment cylinder chord → attenuation map → `I = I_DRR × exp(−atten)` + scatter + PSF + Poisson | ~2–5 FPS CPU (NumPy) |
| **Isaac Lab 3D** | USD quad DRR backdrop + capsule markers in Omniverse viewport | Real-time (Omniverse renderer) |

**Beer-Lambert compositing** is fully implemented and produces physically correct fluoroscopy appearance:
- Multiplicative darkening (catheter attenuates rather than occludes)
- Smooth sub-pixel edges from exact cylinder chord thickness: `t(d) = 2√(r² − d²)`
- Additive attenuation in log-space — self-crossings produce deeper shadows
- Zone-specific per-segment attenuation profile matching real catheter construction
- Background anatomy remains visible through the catheter body

The **Slang GPU path** ray-marches through the CT μ-volume AND catheter geometry in a **single fused pass**, producing depth-correct compositing entirely on GPU with zero CPU compositing overhead. The shader accumulates both volume μ and catheter μ in the same Beer-Lambert integral: `I = I₀ × exp(−∫μ_total ds)`.

#### 1.3 Beer-Lambert Compositing — Implementation Detail

The CPU compositing path implements a complete detector physics chain:

**Step 1 — Attenuation map construction:** For each of `N−1` segments, compute the projected cylinder chord thickness at every pixel within the segment's bounding box. The chord `t(d) = 2√(r² − d²)` is normalised by diameter and weighted by the segment's μ value, then accumulated additively into a single-channel attenuation map.

**Step 2 — Beer-Lambert transmission:** `I_final(u,v) = I_DRR(u,v) × exp(−atten_map(u,v))`.

**Step 3 — Veiling glare / scatter:** `I_blocked = I_DRR × (1 − T)`, blurred with σ=18 px Gaussian, 3% re-added. Models X-ray scatter in patient tissue and detector housing.

**Step 4 — Detector PSF:** Small Gaussian blur (σ=0.7 px) models finite CsI scintillator spatial resolution.

**Step 5 — Poisson quantum noise:** Intensity scaled to photon counts (2000 photons/px), Poisson-sampled, scaled back. Matches low-dose pulsed fluoroscopy statistics.

#### 1.4 Per-Segment Attenuation Profile

Material-specific effective attenuation coefficients are assigned to each segment:

| Zone | Segments | μ Value | Physical Material |
|------|----------|---------|-------------------|
| Proximal marker band | 0–1 | 3.0 | Tungsten |
| Braided shaft | 2 → 60% | 0.8 | Nitinol braid |
| Transition zone | 60% → 85% | 0.8→0.2 | Sparse braid + polymer |
| Soft polymer tip | 85% → 95% | 0.15 | Pure polymer (PEBAX) |
| Distal tip marker | Last 3 | 5.0 | Platinum coil |

This profile is used by both the CPU compositing and the catheter segment data passed to the Slang renderer.

#### 1.5 C-arm Projection Model

A standard pinhole camera model parameterised by interventional C-arm geometry:

- **Intrinsics:** `f_px = SID / pixel_spacing` (default: 1000/0.81 ≈ 1235 px), principal point at detector centre.
- **Extrinsics:** `R = Rx(cran/caud) × Ry(LAO/RAO)`, source at `(0, 0, −SOD)` in camera frame. Iso-centre at world origin.
- **Cone-beam magnification:** `r_px = r_physical × f_px / z_cam` — depth-dependent segment radius matching real cone-beam geometry.

#### 1.6 Slang GPU Unified Sim Loop

The unified sim loop integrates XPBD catheter physics with Slang GPU fluoroscopy rendering in a tight loop:

```text
┌──────────────────────────────────────────────────────────┐
│  XPBD Rod Solver step(dt) × steps_per_frame              │
│  → positions (N, 3) metres                               │
└──────────────────┬───────────────────────────────────────┘
                   │  ×1000 → mm, offset by volume centre
                   ▼
┌──────────────────────────────────────────────────────────┐
│  Catheter Segment Data (positions, radii, mu_values)     │
│  → structured array → GPU buffer (n_seg × 8) float32     │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│  Slang DiffDRR Renderer — render with catheter           │
│  Slang shader: ray march with catheter                   │
│  → SINGLE fused ray march through CT μ-volume +         │
│    catheter cylinder geometry                            │
│  → I = I₀ × exp(−∫(μ_volume + μ_catheter) ds)           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
        512×512 fluoroscopy frame (GPU → CPU readback)
        + HUD overlay + optional video save
```

**Slang shader internals:** The catheter attenuation kernel loops over all catheter segments, checks ray–cylinder proximity, and accumulates `μ × √(1 − d²/r²)` — the same chord model as CPU compositing but evaluated per-ray-step rather than per-pixel.

**Catheter segment data** (structured GPU buffer):
- `positions`: (N, 3) float32 in mm, world frame
- `radii`: per-segment or scalar, mm
- `mu_values`: per-segment or scalar, mm⁻¹
- Structured array → (n_seg, 8) float32: [p0(3), p1(3), radius, mu] for the GPU structured buffer

#### 1.7 X-Ray Performance Baseline

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Single-env physics FPS | >1,000 Hz | ~1,300 Hz (20 seg, A6000) | Achieved |
| Slang GPU compositing (fused DRR + catheter) | <5 ms at 512×512 | ~40 ms (~25 FPS) | Single GPU ray march, no CPU compositing |
| CPU Beer-Lambert compositing | <2 ms/frame | ~200–500 ms/frame | NumPy; Warp GPU kernel would achieve 30+ FPS |
| 512-env physics FPS | >60 Hz | N/A (single env only) | Sprint 2 target |

#### 1.8 RL Training Pipeline (State-Based)

A complete state-based RL training pipeline exists (no pixel observations):

| Component | Description |
|-----------|-------------|
| Environment | Multi-env production solver, proximal push/rotate control, distance-to-target reward |
| RSL-RL wrapper | Standard VecEnv adapter for RSL-RL PPO |
| PPO config | PPO hyperparameters for catheter navigation |
| Training entry | 512 parallel environments, 1500 max iterations |
| Smoke test | Validates environment without RL dependencies |

---

### ⬤ SENSOR B: ULTRASOUND

#### 1.9 Slang Ultrasound Renderer

Production Slang ultrasound renderer: fixed-step volumetric ray marching + BVH-accelerated ray-triangle intersection, Beer-Lambert attenuation, multi-bounce specular reflection (Phase 2), differentiable rendering (6 DOF), PyTorch autograd integration. OptiX Renderer (legacy): C++/CUDA/OptiX, non-differentiable, not supported for multi-env.

#### 1.10 Ultrasound Performance Baseline

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Single-env render (Phase 1) | <5 ms (128 elem, 1024 depth) | ~3–8 ms (est.) | BVH build timing only |
| Post-processing | <2 ms | ~1–3 ms (GPU PyTorch) | PSF + TGC + Hilbert + log + scan convert |
| End-to-end frame | <10 ms (100 Hz) | ~8–20 ms (est.) | Render + post + transfer |
| Backward pass (6 DOF) | <5 ms | ~1–2 ms (reported) | Phase 1 only |

---

## PART 2: NEXT STEPS

### ☢ X-RAY FLUOROSCOPY — ROADMAP

#### 2.1 Sprint 1 — Close the Simulation Loop (Weeks 1–2) ✅ COMPLETED

| Deliverable | Status |
|-------------|--------|
| Beer-Lambert compositing (physically-correct transmission) | ✅ Implemented — CPU + Slang GPU paths |
| Poisson quantum noise (2000 photons/px) | ✅ Implemented in CPU compositing |
| Veiling glare / scatter (σ=18px, 3% fraction) | ✅ Implemented in CPU compositing |
| Detector PSF (σ=0.7px) | ✅ Implemented in CPU compositing |
| Per-segment attenuation profile (5-zone catheter model) | ✅ Implemented |
| Cone-beam magnification of projected catheter radius | ✅ Implemented |
| Slang GPU fused DRR + catheter single ray march | ✅ Implemented |
| Catheter segment data → Slang structured buffer pipeline | ✅ Implemented |
| RL state-based training pipeline (PPO) | ✅ Implemented |

**Remaining from Sprint 1:**

| Deliverable | Status |
|-------------|--------|
| Proximal kinematic control API | ✅ Implemented (push/rotate) |
| Beam hardening (polyenergetic correction) | ❌ Not implemented |

#### 2.2 Sprint 2 — Multi-Environment + Collision (Weeks 3–4)

Multi-env XPBD solver (512+ parallel rods), SDF/mesh collision, structured observation dict + reward signals. **Status: in progress** — production solver supports multi-env; self-contained XPBD solver remains single-env. Catheter state environment provides structured obs + rewards for state-based RL.

#### 2.3 Sprint 3 — Training Readiness (Weeks 5–6)

Domain randomization, Gymnasium wrapper, CUDA graph integration, automated pytest suite.

### ⬤ ULTRASOUND — ROADMAP

#### 2.4 Sprint 1 — Isaac Lab Sensor Module

Ultrasound sensor wrapper, CT-to-volume pipeline, observation dict integration.

#### 2.5 Sprint 2 — Multi-Environment + Performance

Batched rendering, GPU memory pipeline (zero-copy), pre-computed scan conversion grid.

#### 2.6 Sprint 3 — Training Readiness

Domain randomization (probe frequency, attenuation, scattering, noise), Gymnasium wrapper, differentiable pose optimization, automated test suite.

---

## PART 3: AGENTIC WORKFLOW — X-RAY GUIDED CATHETER INTERVENTION

### 🤖 OPENCLAW-ENABLED AGENTIC CO-DEVELOPMENT

#### 3.1 Motivation

The I4H v0.6 release positions the X-ray guided catheter intervention workflow as the flagship demo. The manual pipeline — preprocess CTA, configure C-arm, inject catheter, render fluoroscopy, create dataset, train policy, evaluate, iterate — currently takes days to weeks per experiment cycle. An OpenClaw-based agent can compress this to hours by automating each stage and reasoning about the results between stages.

The agent is not a fixed script. It is a persistent, reasoning loop that uses a skills system and persistent memory. A developer describes their intent in natural language, and the agent orchestrates the full simulation-to-policy pipeline.

#### 3.2 XCATH Gap Analysis — What XCATH Built vs What We Have

XCATH has built 5 simulation methods on top of CTA volumes (DRR, Vessel Boost, DSA, Bolus Label, Bolus Centerline). They need paired synthetic training data (CTA + fluoroscopy + GT pose) that does not exist clinically.

| Feature | XCATH Has | Fluoro-Simulator Has | Gap |
|---------|-----------|---------------------|-----|
| Base DRR (Beer-Lambert) | ✓ | ✓ Slang DiffDRR | None |
| HU→μ mapping | ✓ | ✓ Preprocessor | None |
| **Catheter Beer-Lambert compositing** | ✓ | **✓ Fused GPU + CPU path** | **None (CLOSED)** |
| **Per-segment attenuation profile** | ✓ | **✓ 5-zone catheter model** | **None (CLOSED)** |
| **Detector physics (Poisson + scatter + PSF)** | ✓ | **✓ CPU compositing path** | **None (CLOSED)** |
| **Fused GPU DRR + catheter** | N/A | **✓ Slang ray march with catheter** | **None (NEW)** |
| Vessel Boost (μ × A=8) | ✓ | ✗ | MISSING |
| DSA Simulation (4-step) | ✓ | ✗ | MISSING |
| Bolus / Contrast Dynamics | ✓ VMTK centerline | ✗ | MISSING |
| Gamma Correction (γ=0.8) | ✓ | ✗ | MISSING |
| Poisson Noise | ✓ | **✓ both paths** | **None (CLOSED)** |
| Scatter Simulation | ✓ physical model | **≈ 2D veiling glare (CPU path)** | PARTIAL |
| Misregistration Jitter | ✓ | ✗ | MISSING |
| Clinical C-arm Presets | ✓ 4 vendors | ≈ in docstrings (C-arm geometry config) | PARTIAL |
| Vessel Mask Input | ✓ | ✗ | MISSING |
| Differentiable Rendering | Uses DiffDRR | ✓ Slang autodiff (volume only; catheter forward-only) | PARTIAL |
| Temporal Sequences | ✓ 150 frames @ 5fps | ✓ Cine rendering (static μ) | Needs bolus model |

**Gaps closed since last revision:** Beer-Lambert catheter compositing (CPU + GPU), per-segment attenuation profile, detector physics chain (Poisson + scatter + PSF), cone-beam magnification, fused GPU ray march, C-arm projection model, and state-based RL pipeline.

#### 3.3 End-to-End Pipeline (What the Agent Orchestrates)

The agentic workflow decomposes the catheter intervention problem into 4 stages, each producing artifacts consumed by the next:

**Stage 1: Patient Digital Twin Generation**

CTA Volume (NIfTI/DICOM) → HU→μ volume (Preprocessor) → Vessel mask + Centerline + Arrival map (VMTK).

Output: V(t=0) = μ volume + vessel mask + bolus arrival map.

**Stage 2: Physics Simulation + Fluoroscopy Compositing**

Two compositing modes are now available:

| Mode | Description |
|------|-------------|
| **Fused GPU (Slang)** | XPBD solver → catheter segment data → Slang single-pass ray march through μ-volume AND catheter geometry. Beer-Lambert: `I = I₀ × exp(−∫(μ_vol + μ_cath) ds)`. ~25 FPS. |
| **CPU Beer-Lambert** | XPBD solver → project 3D→2D → cylinder chord attenuation map → `I = I_DRR × exp(−atten)` + scatter + PSF + Poisson. Full detector physics. |
| **Max-attenuation (planned)** | `μ_composited(v) = max(μ_anatomy(v), μ_instrument)` via atomic max on GPU. Needed for true volumetric instrument injection. |

Output: Composited fluoroscopy frame per timestep (512×512 BGR uint8).

**Stage 3: Sensor Simulation (Fluoro Rendering)**

The Slang GPU path performs stages 2 and 3 in a single fused pass — no separate "render" step is needed. For the CPU path, the DRR backgrounds are pre-rendered and compositing happens in image space.

Realism pipeline (CPU path, applied after Beer-Lambert):
1. Veiling glare: `I_scatter = 0.03 × GaussianBlur(I_blocked, σ=18px)`
2. Detector PSF: `GaussianBlur(I_final, σ=0.7px)`
3. Poisson noise: `Poisson(I × 2000) / 2000`

Output: Synthetic fluoroscopy per frame + HUD telemetry (sim time, FPS, tip position, C-arm angle).

**Stage 4: Policy Training & Evaluation**

Currently implemented for **state-based** RL:
- Multi-env catheter environment with proximal push/rotate actions and distance-to-target reward
- PPO training via RSL-RL with 512 parallel environments
- No pixel/fluoroscopy observations yet (Sprint 2+)

Future: Dataset (images + kinematics + GT) → IL/RL via GR00T-H → SIL Testing in Isaac Sim → metrics.

#### 3.4 OpenClaw/NemoClaw Skills Architecture

Each pipeline stage is wrapped as a portable Skill — a skill definition file alongside configuration templates and example inputs/outputs. The agent selects, configures, and chains skills based on the developer's natural-language request.

**Skill 1: patient-digital-twin**

Purpose: Convert raw CTA data into simulation-ready inputs (μ volume, vessel mask, centerline, arrival map).

Agent decides: Segmentation threshold, injection root selection, velocity model parameters, hemisphere selection for selective injection.

| Input | Output |
|-------|--------|
| CTA NIfTI/DICOM path | HU→μ mapped 3D volume |
| | Binary vessel mask |
| | VMTK vessel centerline graph |
| | Dijkstra travel-time per voxel (arrival map) |

**Skill 2: catheter-physics-sim**

Purpose: Configure catheter–vessel physics in Isaac Lab + Newton, and composite the catheter for X-ray rendering.

Agent decides: Catheter stiffness, damping, vessel friction, SDF collision margin, compositing mode, data collection mode.

**Implemented compositing modes:**

| Mode | Status | Description |
|------|--------|-------------|
| Cylinder Beer-Lambert (image-space) | ✅ Implemented | Per-segment cylinder chord attenuation |
| Fused volume + catheter (GPU) | ✅ Implemented | Slang single ray march |
| Max-attenuation (atomic max) | ❌ Planned | Volumetric instrument injection into μ grid |

**Skill 3: sensor-sim-xray**

Purpose: Render fluoroscopy/DSA from the composited anatomy+catheter volume.

Agent decides: Rendering mode, C-arm preset, realism parameters, domain randomization ranges.

**Rendering modes:**

| Mode | Description | Status | Requires |
|------|-------------|--------|----------|
| DRR | Standard DRR — `I = I₀·exp(−∫μ ds)` | ✅ Implemented | μ volume + pose |
| DRR with catheter | **Fused DRR + catheter Beer-Lambert** — `I = I₀·exp(−∫(μ_vol + μ_cath) ds)` | ✅ **Implemented** | μ volume + pose + catheter segment data |
| CPU Beer-Lambert | **CPU Beer-Lambert compositing** on pre-rendered DRR PNGs with full detector physics | ✅ **Implemented** | DRR images + solver positions |
| Vessel boost | DRR with μ×A on vessel voxels (A=8) | ❌ Not implemented | μ volume + vessel mask + pose |
| DSA | 4-step DSA: contrast DRR, mask DRR + jitter, subtraction, post-process (k=20, γ=0.8) | ❌ Not implemented | μ volume + vessel mask + pose |
| DSA temporal | DSA with bolus dynamics — per-frame μ update | ❌ Not implemented | μ volume + vessel mask + arrival map + time |

**Implemented realism features:**

| Feature | CPU Path | Slang GPU Path | Notes |
|---------|----------|---------------|-------|
| Beer-Lambert transmission | ✅ | ✅ | Core compositing |
| Per-segment μ profile | ✅ | ✅ | 5-zone catheter model |
| Cone-beam magnification | ✅ | ✅ (implicit in 3D geometry) | Depth-dependent radius |
| Veiling glare / scatter | ✅ (σ=18px, 3%) | ❌ | 2D approximation |
| Detector PSF | ✅ (σ=0.7px) | ❌ | CsI scintillator model |
| Poisson quantum noise | ✅ (2000 photons/px) | ❌ | Low-dose fluoroscopy |
| Gamma correction (γ=0.8) | ❌ | ❌ | Clinical display TF |
| Scatter (physics-based) | ❌ | ❌ | Only 2D veiling approximation |
| Beam hardening | ❌ | ❌ | Monoenergetic only |
| Misregistration jitter | ❌ | ❌ | DSA requirement |

**Skill 4: dataset-creation**

Purpose: Generate paired multimodal datasets from simulation runs. Output HDF5/WebDataset with fluoroscopy frames, catheter pose, C-arm ground truth, contact forces, and timestamps.

Agent decides: Dataset size, domain randomization ranges, train/val/test split, storage format.

**Skill 5: reward-function**

Purpose: Define and configure RL reward for catheter navigation.

| Component | Formulation | Weight |
|-----------|-------------|--------|
| Target proximity | `r = −‖tip − goal‖₂` | 1.0 |
| Wall contact penalty | `r = −λ_c · Σ max(0, F − F_threshold)` | 0.5 |
| Procedure time penalty | `r = −λ_t · Δt` | 0.1 |
| Tip force penalty | `r = −λ_f · ‖F_tip‖` (avoid perforating) | 0.3 |
| Fluoroscopy dose | `r = −λ_d · n_frames` (minimize imaging) | 0.05 |
| Progress reward | `r = Δ(dist_to_target) / Δt` | 0.2 |
| Success bonus | `r = +R_bonus if ‖tip − goal‖ < ε` | 10.0 |

**Current implementation:** Distance-to-target reward with proximal push/rotate actions. Multi-env via production solver. Full reward table above is the target specification.

**Skill 6: policy-training**

Purpose: Train catheter navigation policy via imitation learning and RL.

Pipeline: Teleop demos → IL (GR00T-H behavioral cloning) → RL fine-tuning (PPO/SAC) → SIL checkpoint evaluation.

Agent decides: IL epochs, RL algorithm, learning rate schedule, checkpoint frequency, early stopping criteria.

**Current implementation:** PPO via RSL-RL, 512 parallel envs, state observations only. No IL pipeline yet.

**Skill 7: evaluation**

Purpose: Run simulation-in-the-loop (SIL) policy evaluation and report metrics.

| Metric | Description |
|--------|-------------|
| Success rate | % episodes reaching target within threshold |
| Mean navigation time | Average time-to-target |
| Max wall contact force | Safety metric |
| Fluoroscopy dose | Number of imaging frames used |
| FID | Fréchet Inception Distance (simulated vs real fluoroscopy) |
| Registration accuracy | Synthetic-trained model tested on real DSA |

Agent behavior after evaluation: Analyzes results, identifies failure modes (e.g., policy fails at ICA-MCA bifurcation), proposes config changes, and re-runs the pipeline.

#### 3.5 Agent Interaction Model

The agent operates via Slack or IDE. A typical session:

> **Developer:** "Train a catheter navigation policy for cerebral ICA → MCA. Patient CTA #42, Philips Azurion C-arm, DSA mode, 50 teleop demos."
>
> **Agent [patient-digital-twin]:** Load CTA NIfTI → HU→μ volume (512×512×300). VMTK centerline: 5,243 nodes, 176 cells. Dijkstra arrival map from ICA root. Selective injection: left hemisphere only. "Patient twin ready."
>
> **Agent [catheter-physics-sim]:** Configure Newton: stiffness=2.5 N/m, friction=0.3. Build vessel SDF. **Beer-Lambert compositing: per-segment μ profile (tungsten markers μ=3.0, nitinol shaft μ=0.8, platinum tip μ=5.0).** Collect 50 teleop demos, mean episode: 45s. "Ready for sensor sim."
>
> **Agent [sensor-sim-xray]:** Philips Azurion 7: SDD=1240, SID=780, 2480×1920. **Rendering via Slang fused DRR + catheter: single GPU ray march at ~25 FPS.** DSA mode: k=20, γ=0.8. Domain randomization enabled. **Detector physics: Poisson noise (2000 photons/px), veiling glare (σ=18px, 3%), detector PSF (σ=0.7px).** "Rendered 10,000 paired frames across 200 episodes."
>
> **Agent [dataset-creation]:** Package as HDF5: fluoroscopy + catheter_pose + GT_carm_pose. Train/val/test: 80/10/10. "Dataset: 10,000 frames, 200 episodes, 2.3 GB HDF5."
>
> **Agent [policy-training]:** IL from 50 teleop demos → behavioral cloning, 100 epochs. RL fine-tuning: PPO, 2000 epochs. "Training complete. Best checkpoint: epoch 1600."
>
> **Agent [evaluation]:** SIL: 100 evaluation episodes. Success rate: 73%, mean time: 38s, max contact: 0.08N. "Analysis: 27% failures cluster at ICA-MCA bifurcation. Recommendation: add curriculum stage for bifurcation navigation."
>
> **Agent → re-runs pipeline:** Updated config → success rate: 84%. "Remaining failures are distal MCA. Suggest collecting 20 more teleop demos targeting distal branches."

#### 3.6 Missing Features That Enable the Agent

The missing features identified in the gap analysis are not optional for the agentic workflow — they are the simulation fidelity upgrades without which the agent produces unrealistic data that fails sim-to-real transfer.

**Features now implemented (removed from blockers):**
- ✅ Beer-Lambert catheter compositing (CPU + GPU Slang fused path)
- ✅ Per-segment attenuation profile (5-zone catheter model)
- ✅ Detector physics: Poisson noise, veiling glare, detector PSF
- ✅ C-arm projection model with cone-beam magnification
- ✅ State-based RL training pipeline (PPO, 512 envs)

**Remaining missing features:**

| Missing Feature | Which Skill Needs It | Why the Agent Cannot Work Without It |
|----------------|---------------------|-------------------------------------|
| Vessel mask input | patient-digital-twin, sensor-sim-xray | Cannot configure DSA, vessel boost, or selective injection |
| DSA pipeline | sensor-sim-xray | Primary rendering mode for catheter navigation training |
| Vessel boost | sensor-sim-xray | Quick verification mode before committing to full DSA render |
| Gamma correction | sensor-sim-xray | Domain gap: without clinical display transfer function, synthetic images fail FID |
| Physics-based scatter | sensor-sim-xray | DSA without proper scatter is unrealistically clean; policies overfit |
| Misregistration jitter | sensor-sim-xray | DSA mask subtraction requires jitter for realism |
| C-arm preset registry | sensor-sim-xray | Agent needs to select geometry from natural language ("use Philips Azurion") |
| Bolus tracking | sensor-sim-xray (temporal) | Cannot generate contrast arrival sequences for temporal training |
| Per-frame μ update | sensor-sim-xray | Cine rendering uses static μ — no catheter advancement or contrast propagation |
| Beam hardening | sensor-sim-xray | Monoenergetic rendering identified as realism gap vs DeepDRR |
| Max-attenuation volume compositing | catheter-physics-sim | Volumetric instrument injection for true depth-correct DRR |
| Multi-env fluoroscopy | sensor-sim-xray | Batched rendering for 512+ parallel training envs |
| Image-based RL observations | reward-function, policy-training | Current env is state-only; fluoroscopy pixel obs needed for image-guided policy |
| Realism metrics (FID) | evaluation | Iterative refinement loop requires quantitative realism feedback |

#### 3.7 Implementation Roadmap

The roadmap has three phases, ordered by dependency. **Phase 1 is partially complete.**

**PHASE 1: Simulation Fidelity (Weeks 1–3)**

| Week | Deliverable | Effort | Status |
|------|-------------|--------|--------|
| 1 | Beer-Lambert catheter compositing (CPU + GPU) | ~3 days | ✅ Done |
| 1 | Detector physics (Poisson, scatter, PSF) | ~1 day | ✅ Done |
| 1 | Per-segment attenuation profile | ~0.5 days | ✅ Done |
| 1 | Slang fused DRR + catheter single ray march | ~2 days | ✅ Done |
| 1 | Vessel mask input + vessel boost + DSA pipeline (4-step) | ~3 days | ❌ Remaining |
| 1 | Gamma correction + scatter convolution + jitter | ~1 day | ❌ Remaining |
| 1 | C-arm presets (4 vendors) + contrast amplification | ~0.5 days | ❌ Remaining |
| 2 | Per-frame μ volume update in cine rendering | ~1 day | ❌ Remaining |
| 2 | Bolus tracking Stage 2 (gamma-variate + per-frame μ) | ~2 days | ❌ Remaining |
| 2–3 | Bolus tracking Stage 1 (VMTK centerline + Dijkstra + arrival map) | ~1 week | ❌ Remaining |
| 3 | Selective injection (hemisphere masking) | ~1 day | ❌ Remaining |
| 3 | Realism evaluation module (FID, SSIM, vessel visibility metrics) | ~2 days | ❌ Remaining |

**PHASE 2: Skill Packaging (Weeks 3–5)**

| Week | Deliverable |
|------|-------------|
| 3–4 | Skill definition + entry points for patient-digital-twin (CTA → μ + mask + arrival map) |
| 4 | Skill definition + entry points for sensor-sim-xray (μ + mask + pose → DRR/DSA) |
| 4 | Skill definition + entry points for catheter-physics-sim (Newton rod solver + compositing) |
| 4–5 | Skill definition + entry points for dataset-creation (paired HDF5 output) |
| 5 | Skill definition + config templates for reward-function and policy-training |
| 5 | Skill definition + evaluation reporting for evaluation |

**PHASE 3: Agent Integration (Weeks 5–7)**

| Week | Deliverable |
|------|-------------|
| 5–6 | Agent skill discovery and chaining logic — parse skill definitions, resolve input/output dependencies |
| 6 | Natural language → config mapping — translate "use Philips Azurion, DSA mode" into YAML overrides |
| 6–7 | Iterative refinement loop — agent runs evaluation, analyzes metrics, proposes changes, re-runs |
| 7 | Slack/IDE integration — agent posts progress, visualizations, and final reports |

#### 3.8 What This Enables for XCATH

With the agentic workflow in place, XCATH's current manual pipeline collapses into an agent-driven loop:

| Current (Manual) | Agentic |
|-----------------|---------|
| Manually run VMTK in conda env, save arrival map | Agent runs patient-digital-twin skill |
| Manually configure C-arm geometry, set DSA parameters | Agent selects C-arm preset from "use Philips Azurion, clinical DSA" |
| Manually render 150 frames, inspect visually | Agent runs sensor-sim-xray (temporal), auto-compares FID against reference |
| Manually export paired data, manage files | Agent runs dataset-creation, outputs HDF5 with proper splits |
| Manually train PoseNet, evaluate on held-out set | Agent runs policy-training + evaluation, proposes hyperparameter changes |
| ~2–3 weeks per experiment cycle | **~hours per cycle**, agent runs continuously |

The critical path is: ~~build the missing rendering features first (Phase 1)~~ **Phase 1 core compositing is complete** — remaining items are DSA/vessel-boost/bolus. Then wrap as skills (Phase 2), then connect to the agent (Phase 3).

#### 3.9 PRD Requirements Mapping

| PRD Requirement | Agentic Workflow Component | Status |
|----------------|---------------------------|--------|
| RQ-02-1 (Reference workflow for catheter navigation) | 4-stage pipeline: patient twin → physics sim → sensor sim → policy training | ✅ Core pipeline implemented |
| RQ-02-2 (Simulation + asset stack) | Missing features (DSA, vessel boost, bolus, beam hardening, C-arm presets) + material library + Newton catheter physics | ⚠️ Partial — Beer-Lambert + physics done; DSA/bolus pending |
| RQ-02-3 (OpenClaw-enabled agentic co-development) | 7 skills with definitions, agent orchestration, iterative refinement loop | ❌ Skill packaging not started |
| RQ-06 (Unified sensor sim API) | sensor-sim-xray skill wraps fluoro-simulator; same pattern for ultrasound skill | ⚠️ Partial — renderer exists, skill wrapper pending |
| LHAs: XCATH, Remedy | Skills configured with XCATH's validated parameters (k=20, γ=0.8, v_ref=150mm/s, Philips/GE presets) | ❌ Preset registry not implemented |

---

## PART 4: FUTURE SENSORS

The sensor simulation architecture is designed to be extensible. The following sensors would complete the XCath simulation environment.

| Sensor | Priority | Integration Effort | Prerequisite |
|--------|----------|-------------------|--------------|
| Force/Torque | High | Low — data already in collision solver | Sprint 2 collision integration |
| Endoscopy / RGB | Medium | Low — Isaac Lab camera sensor exists | Realistic vessel interior USD assets |
| IVUS | Medium | Moderate — adapt existing US renderer | High-res vessel wall models |
| CBCT | Low–Medium | Moderate — batched DRR + FDK recon | GPU ray-caster (X-ray future work) |
| Pressure / Flow | Low | High — new hemodynamic solver needed | 1D fluid solver, vessel centerlines |
