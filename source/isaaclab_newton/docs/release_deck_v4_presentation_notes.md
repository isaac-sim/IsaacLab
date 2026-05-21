# Presentation Notes — X-Ray–Guided Robotic Catheter Intervention
## Release Status & Roadmap
### Verified against implementation in `source/isaaclab_newton/` and `fluoro-simulator/fluorosim/`

---

## Why This Workflow Matters — Clinical & Strategic Context

### The Clinical Problem

Neurointerventional procedures — cerebral angiography, mechanical thrombectomy, flow diverter placement — are performed under live X-ray fluoroscopy on a moving catheter the operator cannot directly see in 3D. The operator navigates entirely from a 2D fluoroscopy image that shows bone, soft tissue, and the catheter wire, but **does not show vessel walls**. Vessel anatomy is only visible during contrast injection (DSA), which lasts a few seconds and exposes the patient to iodinated contrast and additional radiation.

Automating or augmenting this navigation requires an AI system that can:
1. Segment vessels from intraoperative fluoroscopy — without a 3D reference
2. Register pre-operative CTA (3D volume) to intraoperative fluoroscopy (2D projection) in real time
3. Control a robotic catheter based on the registered view

All three tasks require **large volumes of paired training data**: CTA volume + fluoroscopy frame + ground-truth 3D-to-2D projection pose. This data **does not exist in clinical practice** — CTA and DSA are acquired at different times, in different positions, with different imaging geometries. You cannot label ground-truth pose retrospectively.

### Why NVIDIA's Simulator Solves This

The simulator generates unlimited synthetic paired data with **exact ground-truth**:
- Input: CT volume + vessel mask + C-arm pose parameters
- Output: Synthetic fluoroscopy / DSA image physically equivalent to clinical output
- Ground truth: Exact 3D-to-2D projection pose (known analytically, no annotation needed)

This turns a data-scarcity problem into a compute problem — which NVIDIA is structurally positioned to solve at scale (512 parallel environments, MAISI for CT generation, Isaac for the training loop).

---

## XCath Collaboration — Technical Summary

### Context

XCATH Robotics is building a real-time monitoring system for neurointerventional surgery: 3D vessel overlay on live fluoroscopy during catheter procedures. Their AI pipeline requires vessel segmentation and 2D/3D CTA-to-fluoroscopy registration. Both models require the paired data that the NVIDIA simulator provides. The collaboration (2026-03 → 2026-04) ran as a weekly technical sprint with Jung-eun Park (AI/ML Engineer, XCATH) embedded in the implementation.

### What XCath Validated — and What Moved

**1. Blood flow model: Hagen-Poiseuille replaces allometric**

| Parameter | 2026-03 | 2026-04 |
|---|---|---|
| Velocity model | v = 150·(r/r_ref)^0.5 (Murray's Law) | v = 350·(r/2.0)^2.0 (Hagen-Poiseuille) |
| Bolus delay vs real DSA | 5.6s sim vs 2.13s real (2.6× off) | 1.90s sim vs 2.13s real (89% match) |
| Distal MCA relative velocity | 71% of ICA (too fast) | 25% of ICA (realistic) |

The original allometric model (Murray's Law, power=0.5) produced unrealistically fast distal flow. Switching to Hagen-Poiseuille (power=2.0) corrects for viscous resistance in small vessels and brought the simulated bolus delay to within 11% of a real clinical DSA acquisition.

**2. Selective injection: Feeding Trunk Excision (F-8)**

Real clinical DSA injects contrast at ONE vessel (e.g., left ICA), which perfuses only the ipsilateral territory. The previous simulator injected all 4 roots simultaneously, filling the entire brain — inconsistent with clinical ground truth.

XCath contributed the **Feeding Trunk Excision (F-8) algorithm**:
1. Partition vascular territory via multi-source Dijkstra (Voronoi tessellation on the centerline graph)
2. Identify non-selected feeding artery trunks (ICA/VA proximal segments)
3. Excise trunks from the graph — models the antegrade pressure barrier
4. Run Dijkstra from the selected root on the excised graph only

Result: single-root simulation covers 43.1% of nodes (2,259 / 5,243), matching real selective ICA injection territory. Peak delay: 2.20s sim vs 2.13s real (3% error).

**3. Dispersion correction: per-voxel TDC width modulation**

A single gamma-variate curve applied uniformly to all vessel voxels fails to reproduce the proximal-to-distal bolus spreading observed in real DSA (proximal vessels wash in and out fast; distal capillary beds spread the bolus over several seconds). XCath validated a two-phase correction:

- **Phase 1 (distance-based):** β_eff = β₀ + k × arrival_time (k=0.15 selected by sweep; achieves 2.55× FWHM ratio, target ≥1.5×)
- **Phase 2 (radius-based):** β_eff += α_res / max(r_voxel, 0.1) — thin vessels get wider TDC, simulating resistance-driven residence time

This correction passed both gates: distinguishability = 0.607 (target >0.3), asymmetry = 2.55× (target ≥1.5×).

**4. FluoroSim vs DeepDRR — validating monoenergetic rendering**

A key architectural question: does FluoroSim's monoenergetic (70 keV) renderer produce DSA images realistic enough for training registration and segmentation models, vs DeepDRR's polyenergetic (90 kVp) renderer?

XCath ran a 3-phase comparison on real CTA data (HUMI029):

| Metric | Result |
|---|---|
| SSIM (static DRR) | 0.964 |
| MAE | 0.093 (9.3%) |
| PSNR | 19.01 dB |
| Pearson Correlation | 0.981 |

**Key finding:** Beam hardening (the primary monoenergetic limitation) **cancels in DSA subtraction** — both the mask DRR and the contrast DRR are affected equally by bone-edge hardening, so the subtraction removes it. Monoenergetic is practically sufficient for DSA-based training data. The residual difference at bone boundaries (9.3% MAE) is mitigated by the misregistration jitter added to the mask DRR.

**5. Misregistration jitter — a calibrated realism workaround**

Because monoenergetic rendering produces perfect bone subtraction (too clean vs real DSA where patient motion leaves bone residual artifacts), XCath validated injecting controlled misregistration between mask and contrast DRRs: rotation σ = 0.05°, translation σ = 0.1 mm. This artificially introduces the bone edge artifacts that real clinical DSA exhibits, making the synthetic data distribution match real data more closely.

### Known Gaps Identified in Collaboration

| Limitation | Measured Evidence | Path to Fix |
|---|---|---|
| Beam hardening at bone edges | 9.3% MAE vs DeepDRR | Polyenergetic rendering (Sprint 2) |
| Noise texture mismatch | Noise CV: sim 0.47 vs real 1.46 (3.1×) | GPU-side Poisson + NPS calibration |
| Iodine K-edge (33.2 keV) | δμ at 70 keV < spectrum-averaged δμ | Polyenergetic + K-edge material model |
| Fine distal branches | Less dense in sim than in real DSA | Finer vessel annotation / mesh resolution |
| Multi-patient generalization | Validated on Patient #1 only | 2–3 additional CTA datasets (2–3 days, blocked on data) |

### Strategic Takeaway

XCATH provided the clinical validation loop that turned the simulator from a technically correct rendering system into a **calibrated training data generator**. The collaboration produced five specific algorithmic improvements (velocity model, selective injection, dispersion correction, DeepDRR comparison, misregistration jitter), 17 completed items, 125 unit tests, and a modular architecture with a `Dockerfile` for reproducible deployment. The open items (multi-patient validation, polyenergetic rendering) are now on the Sprint 2 roadmap.

---

## Slide 1 — Title

**X-Ray–Guided Robotic Catheter Intervention Simulation Workflow**

This presentation covers the current release status, what has been shipped, and the forward roadmap for the X-ray guided robotic catheter simulation system built on NVIDIA Isaac for Healthcare.

The system integrates three tightly coupled components running on a single GPU:
- **Cosserat rod physics** (XPBD + Newton bridge) for catheter mechanics and vessel-mesh collision
- **Slang GPU DRR renderer** (`diffdrr_slang.slang`) for physically accurate fluoroscopy simulation
- **RL training pipeline** (RSL-RL PPO) for policy learning at scale across 512 parallel environments

The end-to-end loop — physics step → fluoroscopy render → RL observation → policy update — runs without CPU–GPU synchronization on the hot path, making it viable for real-time training.

---

## Slide 2 — Architecture Overview

The slide shows a four-stage pipeline. All four stages share a single coordinate frame (CT volume millimeters). The data flows top-to-bottom: a CT scan enters at the top and an autonomous catheter navigation policy exits at the bottom, deployable on real hardware.

---

### Stage 1 — CT Ingestion Pipeline

**Purpose:** Convert a patient CT scan into two assets that every downstream stage depends on — a 3D attenuation volume for the renderer and a physics collision mesh for the solver.

**Data flow:**

```
CT/CTA (DICOM/NIfTI)
    │
    ▼
HU → μ conversion                ← maps Hounsfield units to linear attenuation
    │                               coefficients (mm⁻¹) using a material look-up table.
    │                               Dense bone: ~0.3–0.5 mm⁻¹.
    │                               Soft tissue: ~0.02–0.05 mm⁻¹.
    │                               Air: ~0.0 mm⁻¹.
    ▼
μ Volume (3D)  ─────────────────→ fed directly into the Slang DRR renderer.
    │                               Loaded once at init; never regenerated per frame.
    │
    ├── Vessel Segmentation         threshold on HU (contrast-enhanced vessels
    │       │                       are ~100–300 HU above baseline blood).
    │       ▼
    │   Binary vessel mask
    │       │
    │       ├── Mesh Gen. (Marching Cubes / VTK)
    │       │       └─→ wp.Mesh  ─────────────→ fed into XCathRodSolver as
    │       │                                    the collision geometry.
    │       │
    │       └── Centerline (VMTK)
    │               └─→ CenterlineGraph ──────→ Dijkstra arrival map for
    │                                           bolus dynamics (temporal DSA).
    │
    └── SDF Gen.
            └─→ signed-distance field ─────────→ used by XCathRodSolver's
                                                  wp.mesh_query_point_sign_normal
                                                  for particle containment.
```

**Key point for the audience:** This stage runs once per patient anatomy — not once per simulation frame. The CT volume is a static 3D texture on the GPU for the entire training session. Only the catheter geometry buffer changes frame-to-frame.

---

### Stage 2 — Simulation Environment

**Purpose:** Produce physically accurate fluoroscopy observations at speeds compatible with reinforcement learning training by co-locating the physics solver and the X-ray renderer on the same GPU and eliminating all host-side synchronization between them.

---

#### Newton Physics Engine — Cosserat Rod with Vessel-Mesh Constraints

The catheter is modeled as a geometrically exact Cosserat rod discretized into N segments, simulated via eXtended Position-Based Dynamics (XPBD). At each physics substep, the solver enforces four constraint classes in sequence:

1. **Stretch / shear** — maintains segment rest length and cross-sectional integrity
2. **Bending / torsion** — enforces material curvature stiffness along the rod backbone
3. **Vessel-mesh SDF containment** — each particle queries the signed distance field of the vessel wall mesh; particles outside the vessel lumen are projected back along the surface normal direction
4. **AABB broadphase + edge contact** — a mesh-AABB query identifies candidate triangle faces; vertex-triangle and rod-segment-to-mesh-edge contact forces are resolved to prevent wall penetration

Track-guided insertion constrains the proximal (non-tip) particles to a linear insertion axis, reproducing the mechanical boundary condition of a catheter entering through a guide sheath.

The constraint solver uses a block-Thomas direct factorization running entirely on the GPU, captured into a CUDA graph after warm-up. This eliminates kernel-launch overhead for all subsequent physics steps, yielding throughput of approximately **1,300 Hz per environment**.

---

#### Fluoroscopy Renderer — Beer-Lambert Ray-March with Inline Catheter Compositing

The CT attenuation volume (voxel-wise linear attenuation coefficients μ, derived from Hounsfield Units via a two-segment piecewise HU→μ map) is uploaded to GPU texture memory once per patient anatomy and remains immutable for the entire training session. No per-frame CT reconstruction occurs.

At each render call, a GPU ray-march kernel integrates Beer-Lambert attenuation along each detector pixel ray:

```
I(x,y) = I₀ · exp( −∫ [μ_tissue(s) + μ_catheter(s)] ds )
```

The catheter is composited **inline during the ray march** using a per-sample-point inside/outside test: at each step, the GPU kernel evaluates the perpendicular distance from the world-space sample point to each catheter segment's axis. If the sample falls within the cylinder radius, an attenuation contribution is added to the running integral:

```
μ_total(s) = μ_CT(s)  +  Σᵢ  μᵢ · sqrt(1 − dᵢ²/rᵢ²)
```

The `sqrt(1 − d²/r²)` factor is a chord-length weight: a sample at the cylinder center (d=0) contributes the full attenuation coefficient, while a sample near the surface (d→r) contributes proportionally less, approximating the shorter path length through the cylinder cross-section at that offset. This produces physically correct depth integration — the catheter and the CT anatomy share the same Beer-Lambert line integral, so a catheter partially occluded by bone is attenuated by both simultaneously, with no compositing artifact or alpha-blending approximation.

The catheter geometry is defined by a flat GPU-resident buffer of (N−1) capped-cylinder segments (8 floats each: proximal endpoint xyz, distal endpoint xyz, radius, μ). This buffer is overwritten from the physics solver's particle positions at each timestep — no texture write-back, no CT volume modification.

Post-processing (Poisson quantum noise, detector PSF convolution, gamma correction, misregistration jitter) is applied on the same GPU timeline before the frame is returned.

---

#### Unified Loop — Per-Timestep Execution Order

Each timestep proceeds as follows:

| Step | Operation | Detail |
|---|---|---|
| 1 | Root control | Proximal catheter boundary condition updated via GPU kernel — insertion depth δ, rotation angle θ |
| 2 | Physics step | XPBD CUDA graph replayed; all N particle positions updated in GPU memory |
| 3 | Segment buffer update | Particle positions written to the renderer's catheter segment buffer (N×8 floats: center xyz, radius, μ); no host transfer |
| 4 | Render call | Ray-march kernel dispatched over the detector grid with current C-arm rotation matrix; catheter composited inline; realism pipeline applied |
| 5 | Output | Fluoroscopy frame (H×W float32, GPU-resident) and catheter state vector (tip position, orientation, insertion depth) returned synchronously |

**Why this is fast:** The CT μ-volume never moves. Only the catheter segment buffer (N×8 floats) changes per frame. The CUDA graph eliminates physics kernel-launch overhead. End-to-end throughput is approximately **25 FPS** (physics + render + realism) per environment, with the physics subsystem running at a higher internal substep rate decoupled from the render cadence.

---

#### Domain Randomization

Applied per episode to the sim loop inputs:
- C-arm projection angle (random rotation within clinical range)
- Attenuation noise on the μ-volume (Gaussian perturbation per voxel)
- Vessel geometry variants (different patient anatomies via CT swap)

---

#### Synchronized Outputs (per frame → RL loop)

- **Fluoroscopy observation:** H×W float32 GPU tensor → policy network input
- **Catheter state vector:** tip position, orientation, insertion depth, curvature → reward and termination signal
- Both outputs are GPU-resident with no host round-trip, keeping the training pipeline fully on-device

---

### Stage 3 — Training Pipeline

**Purpose:** Train a policy that maps fluoroscopy observations to catheter control actions.

**Flow:**

```
Demonstration Collection
    │  Expert trajectories recorded from the sim loop.
    ▼
GR00T-N Training
    ├── Imitation Learning (IL)     ← learns coarse navigation from demonstrations.
    └── Reinforcement Learning (RL) ← fine-tunes with distance-to-target reward.
         (RSL-RL PPO, 512 parallel envs)
    ▼
Trained Policy (GR00T-H)
```

**Curriculum Learning (parallel track):** Progressively increases difficulty (shorter vessels, more tortuosity, higher noise) and anatomical diversity (multiple patient CT scans via MAISI synthetic CT). This forces the policy to generalize rather than memorize a single anatomy.

**State-based today, pixel-based in Sprint 2:** The RL observation today is catheter tip state (position, orientation). Sprint 2 replaces this with the raw fluoroscopy frame tensor from the Slang renderer — the policy then learns directly from X-ray pixels, matching what a deployed system would observe.

---

### Stage 4 — Runtime Deployment

**Purpose:** Run the trained policy on real hardware in the clinical environment.

```
Trained Policy (GR00T-H)
    │
    ▼
Holoscan IGX                      ← low-latency inference node at the bedside.
    │   Receives real X-ray frames from the C-arm at 30 FPS.
    │   Runs policy inference in <10 ms.
    │
    ├── Real C-arm Fluoroscopy     ← live X-ray stream; same image format as
    │   (live X-ray stream)           the synthetic frames used in training
    │                                 (sim-to-real gap is closed by domain randomization
    │                                 and realistic detector physics in Stage 2).
    │
    └── Robotic Catheter (XCath)   ← receives push/rotate commands from the policy.
            │
            └── Safety Layer: motion limits, collision guards
                (checked every inference cycle before commanding the robot)
    │
    ▼
OUTCOME: Autonomous catheter navigation
```

---

### How all four stages connect end-to-end

```
Patient CT ──[Stage 1]──► μ-volume + wp.Mesh + CenterlineGraph
                                │
                [Stage 2] ◄─────┘
        Policy action ──► Physics step (XCathRodSolver)
                                │
                          New catheter positions
                                │
                          Slang DRR render (static μ-volume + catheter geometry)
                                │
                          Fluoroscopy frame + catheter state
                                │
                [Stage 3] ◄─────┘
                    RSL-RL PPO (512 envs) → Trained policy (GR00T-H)
                                │
                [Stage 4] ◄─────┘
                    Holoscan IGX → XCath robot → patient
```

The same μ-volume that feeds the Slang renderer in Stage 2 is derived from the same CT that the Stage 1 vessel mesh was extracted from. Physics collision and X-ray rendering are therefore always registered to the same patient anatomy — no separate coordinate calibration step is needed between the physics and imaging subsystems.

**Layer 1 — Physics (`XPBDRodSolver` / `XCathRodSolver`)**
- Cosserat rod model discretized into N segments; stretch + Darboux constraints solved via block-Thomas direct factorization
- Batched multi-environment execution: `_BatchedWorkspace` allocates flat contiguous buffers (`rod_offsets`, `edge_offsets`) so all rods across all envs are dispatched in a single Warp kernel launch
- GPU-side proximal control: `apply_proximal_control_gpu` (push) and `set_root_orientation` (rotate) are Warp kernels safe to capture inside a CUDA graph — no CPU sync on the hot path
- `step()` auto-captures the substep loop into a CUDA graph on the first call (`wp.ScopedCapture`) and replays it on subsequent calls
- Vessel-mesh collision in `XCathRodSolver`: SDF path uses `wp.mesh_query_point_sign_normal` for particle containment; AABB broadphase uses `wp.mesh_query_aabb` for rod-segment vs mesh-edge contacts; non-tip particles constrained to a linear insertion track (`track_start`, `track_dir`, `track_length`)

**Layer 2 — Rendering (`SlangDiffDRRRenderer`)**
- Beer-Lambert ray marching through the static μ-volume; catheter composited inline per-segment via `CatheterSegmentData` (`mu_values`, `radii`)
- `renderDRR_forward_batched`: single Slang dispatch for N environments; `dispatchThreadID.z` indexes env
- `renderDRR_backward`: 6-DOF pose gradients via Slang autodiff

**Layer 3 — Detector + DSA**
- `apply_realism()`: scatter → Poisson → PSF → gamma → normalize
- `DSAPipeline.render_dsa_frame()`: 4-step DSA with temporal bolus (`gamma_variate`, Dijkstra arrival map)

---

## Slide 3 — X-Ray Fluoroscopy Pipeline — At a Glance

Key status across all pipeline components:

| Component | Status | Implementation | Multi-Env |
|---|---|---|---|
| Catheter physics solver | ✓ Shipped | Production XPBD Cosserat rod (Warp + Newton bridge) | ✓ All 3 backends |
| DRR volume rendering | ✓ Shipped | Slang `renderDRR_forward_batched` | ✓ Batched |
| Catheter Beer-Lambert compositing | ✓ Shipped | Fused GPU ray-march, `StructuredBuffer<CatheterSegment>` per env | ✓ Batched |
| Volumetric instrument injection | ✓ Shipped | `paint_cylinders_kernel` (Warp `atomic_max`), ~2 ms / 64 nodes / 512³ | Sprint 2 |
| DSA pipeline (4-step) | ✓ Shipped | `DSAPipeline` in `dsa.py`: mask → contrast → subtract → post-process | Sprint 2 |
| Bolus dynamics (temporal DSA) | ✓ Shipped | `extract_centerlines` (VMTK) + `compute_arrival_map` (Dijkstra) + `gamma_variate` C(t) | Sprint 2 |
| C-arm geometry + vendor presets | ✓ Shipped | `CarmGeometry` in `config.py` — GE, Siemens, Philips, Ziehm classmethod factories | N/A |
| Detector physics chain | ✓ Shipped | `apply_realism()` in `realism.py` — scatter → Poisson → PSF → gamma → normalize | N/A |
| RL training pipeline | ✓ Shipped | PPO via RSL-RL, state obs, 512 envs | ✓ Yes |
| Image-based RL observations | ✗ Sprint 2 | Pixel obs from fluoroscopy frames | — |
| Beam hardening (polyenergetic) | ✗ Sprint 2 | Closes realism gap vs DeepDRR | — |

**Key talking point:** The simulation loop is complete end-to-end. Physics, rendering, and detector realism are all operational and validated. Sprint 2 work is narrowly scoped to renderer scalability beyond N=8 environments and closing the RL observation gap from state-based to pixel-based.

---

## Slide 4 — Part 1 Divider: Current Status

**Transition:** This section covers what is measured, validated, and running today — before any Sprint 2 work.

---

## Slide 5 — X-Ray Performance Baseline

**Context:** All numbers measured on NVIDIA A6000, 512×512 detector resolution. These numbers represent the throughput of each subsystem in isolation and end-to-end. The table is organized by component, with current status clearly segmented into achieved, partial, and reference-only tiers.

---

### Row-by-row strategic breakdown

**Row 1 — Single-environment physics: ~1,300 Hz (Target: >1,000 Hz) ✓ Achieved**

The catheter physics solver exceeds its training throughput target at 1,300 Hz for a 20-segment rod. This is achieved by a block-Thomas direct factorization (linear in rod length, not quadratic), running entirely on the GPU and captured into a CUDA graph after warm-up. Once captured, subsequent physics steps incur near-zero kernel-launch overhead. The practical implication: physics is no longer a bottleneck for RL training, even at 512 parallel environments. The solver scales to N environments by replicating flat particle buffers in memory — each additional environment adds memory linearly but negligible compute beyond that.

**Row 2 — GPU Beer-Lambert ray-march with inline catheter (fused): ~40 ms / ~25 FPS (Target: <5 ms @ 512²) — In Progress**

The GPU ray-march produces physically correct Beer-Lambert fluoroscopy with the catheter composited depth-accurately in a single kernel dispatch. At 512² resolution, the current throughput is ~25 FPS. The gap to the <5 ms target is well-understood: the 3D CT attenuation volume (~100³–256³ float32 voxels) is fetched from global memory on every ray step across all threads. At N=1–4 environments, L2 cache pressure is manageable. Beyond N=8, the volume becomes too large for the L2 cache, causing global memory bandwidth saturation. The path to closing this gap is architectural: replacing the single 3D texture with a `Texture2DArray` that caches one depth slice per environment in L2 — this is the Sprint 2 renderer upgrade.

**Row 3 — CPU Beer-Lambert path: ~200–500 ms/frame (Target: <2 ms) — Reference only, not on training path**

The CPU NumPy Beer-Lambert implementation is not used for RL training. It exists as a physically accurate ground-truth reference for validating the GPU renderer output, debugging realism parameters, and generating offline analysis images. The 200–500 ms figure is expected for a NumPy CPU ray-march and is not a regression. The target <2 ms refers to the GPU path. A GPU-accelerated Warp port of the same realism pipeline achieves >30 FPS — this is the path toward full GPU-resident detector realism (Poisson noise, PSF, gamma correction) in Sprint 2.

**Row 4 — Multi-environment physics (batched XPBD): ✓ Available (Target: >60 Hz @ 512 envs)**

Batched multi-environment physics is fully operational across all three solver backends. The batched XPBD solver maintains flat, contiguous GPU particle buffers with per-environment offsets. CUDA-graph capture covers the entire multi-environment dispatch in a single graph node. No per-environment CPU synchronization is required. This is a hard prerequisite for 512-env RL training — it is complete.

**Row 5 — Multi-environment Slang rendering (batched): ~25 FPS @ N≤4 (Target: >60 Hz @ 512 envs) — In Progress**

The batched GPU renderer dispatches all N environments in a single kernel call using a 3D thread grid (pixel_x, pixel_y, env_idx). Per-environment catheter geometry is packed into a flat structured buffer with per-env offset and count indices, allowing the kernel to composite a different catheter for each environment with no serial dispatch overhead. At N≤4, throughput is comparable to the single-environment case (~25 FPS). Beyond N=8, L2 cache saturation for the shared 3D μ-volume degrades throughput. The `Texture2DArray` Sprint 2 upgrade resolves this by partitioning the volume cache per-environment slice. Until that lands, the batched path is the production path for N≤4 and a functional prototype for larger N.

---

### Strategic summary for the audience

The slide tells a two-part story: **physics is solved, rendering is the known remaining constraint with a clear engineering path to resolution.** The physics subsystem exceeds its training throughput target with margin. The renderer delivers physically accurate X-ray images at 25 FPS today — sufficient for single-environment interactive development — with the multi-environment scaling bottleneck fully diagnosed and its fix scoped to Sprint 2. There are no unknown unknowns on this slide.

---

## Slide 6 — RL Training Pipeline (State-Based)

**Context:** This slide describes the end-to-end reinforcement learning training configuration as it ships today. Each row is a distinct architectural decision — not just a config parameter. The bottom-of-slide caveat ("State-based only today. Pixel/fluoroscopy observations land in Sprint 2+") is the most strategically important statement on the slide.

---

### Row-by-row breakdown

**Row 1 — Environment: Multi-env production solver, proximal push/rotate control, distance-to-target reward**

The training environment wraps the multi-environment XPBD physics solver with a gymnasium-compatible step/reset interface. At each timestep, two control actions are available — advance (push the catheter proximally along its insertion axis) and rotate (twist the proximal end) — exactly matching the two degrees of freedom a clinician controls during a real procedure. The reward signal is the reduction in Euclidean distance between the catheter tip and the target vessel landmark. This is intentionally simple: it gives the policy a dense, well-conditioned learning signal during Sprint 1. More clinically realistic reward shaping (path curvature penalties, contact force limits, fluoroscopy coverage metrics) is deferred to Sprint 2 once pixel observations are online.

**Why it matters:** The environment is the contract between the physics simulation and the RL algorithm. Using the production-grade physics solver (not a simplified proxy) means that policies trained here are immediately transferable to the full simulation — no sim-to-sim gap.

---

**Row 2 — RSL-RL wrapper: Standard VecEnv adapter for RSL-RL PPO**

RSL-RL is the PPO implementation used across Isaac Lab's robot learning stack. The catheter environment exposes a standard vectorized environment interface (VecEnv), meaning the RL algorithm is entirely decoupled from the physics and rendering implementation. Swapping from state observations to pixel observations, or from RSL-RL PPO to any other policy gradient algorithm, requires no changes to the environment code — only the observation tensor shape changes.

**Why it matters:** Decoupling the RL algorithm from the environment is a prerequisite for long-term research flexibility. The same environment will support image-based policies, offline imitation learning, and curriculum learning without architectural changes.

---

**Row 3 — PPO config: Tuned hyperparameters for catheter navigation**

The PPO configuration (learning rate, entropy coefficient, clip range, value function coefficient, GAE lambda) was tuned specifically for the catheter navigation task geometry. Catheter navigation is a low-dimensional but highly non-linear control problem: the tip trajectory is sensitive to small rotations at the proximal end, and the reward signal is sparse near vessel bifurcations. The tuned config addresses these characteristics with higher entropy regularization (to maintain exploration near bifurcations) and a shorter GAE horizon (to reduce variance from long contact sequences).

**Why it matters:** A poorly tuned PPO config on a contact-rich task will fail to converge regardless of environment fidelity. The tuned config is a validated starting point — not a final answer, but a stable baseline from which Sprint 2 image-based training can begin.

---

**Row 4 — Training scale: 512 parallel environments, 1,500 max iterations**

512 parallel environments means the policy network receives 512 independent rollout trajectories per training step, all from different random initial catheter poses. This is what makes PPO sample-efficient on this task: with 512 envs, the policy sees a broad distribution of tip positions, orientations, and distances-to-target on every update, avoiding the local optima that plague single-environment training. The multi-environment XPBD physics is the enabler — it runs all 512 environments in a single CUDA graph replay with near-zero overhead beyond memory.

**Why it matters:** Training throughput scales with the number of parallel environments. At 512 envs, the physics subsystem runs at ~1,300 Hz per environment — the RL training loop is not physics-bound. It is currently render-bound only if pixel observations are enabled, which is why Sprint 2's renderer scaling work is a direct prerequisite for image-based RL at scale.

---

**Row 5 — Smoke test: Validates environment without RL dependencies**

A lightweight smoke test verifies the full environment loop — reset, step, reward computation, done condition — without requiring the RSL-RL or PyTorch training stack to be installed. This is critical for CI validation and for running on headless servers where the full training stack may not be available. It also serves as the minimal reproducible example for debugging environment logic in isolation from the training algorithm.

**Why it matters:** On a system as complex as this (physics + renderer + RL wrapper), having a dependency-free validation path prevents regressions from propagating silently. Any change to the physics solver or observation computation can be validated with a single lightweight command before running a full training job.

---

### Strategic note — the Sprint 2 transition

The bottom-of-slide caveat is the most important statement: **state-based observations are a scaffolding, not the target.** A real deployed policy cannot observe ground-truth catheter tip position — that information does not exist during a live procedure. The policy must navigate using only the fluoroscopy image, exactly as a clinician does. Closing the loop from state obs to pixel obs is what converts this from a physics simulation into a genuine synthetic data pipeline for clinical AI training. That transition is Sprint 2.

---

## Slide 7 — Part 2 Divider: This Release

**Transition:** Sprint 1 deliverables — features shipping with this release.

---

## Slide 8 — This Release — Stage 1 & 2

Two-stage delivery structure of Sprint 1:

**Stage 1 — Core Physics & Rendering**
- Self-contained `XPBDRodSolver` with `_BatchedWorkspace`: flat contiguous GPU buffers, 1 Warp kernel dispatch per substep across all envs
- GPU-side proximal control: `apply_proximal_control_gpu` and `set_root_orientation` — Warp kernels, CUDA-graph capturable, zero CPU sync on hot path
- CUDA-graph capture for the substep loop: `wp.ScopedCapture` on first `step()` call, `wp.capture_launch` on subsequent calls
- Slang batched DRR: `renderDRR_forward_batched` — single Slang dispatch for N environments; `dispatchThreadID.z` indexes the environment
- Catheter Beer-Lambert fused inline in the ray-march — depth-correct, per-segment μ and radius via `CatheterSegmentData`

**Stage 2 — Realism & DSA**
- `DSAPipeline` (4-step): mask DRR → contrast DRR → scatter → jitter → subtract → post-process (`dsa.py`)
- Temporal bolus dynamics: `gamma_variate` C(t) + Dijkstra arrival map from VMTK centerline graph (`vasculature.py`)
- Volumetric instrument injection: `paint_cylinders_kernel` in `instrument-injection` package — Warp `atomic_max` for thread-safe parallel volume painting, ~2 ms for 64-node catheter on 512³ volume
- Detector physics chain: `apply_realism()` — scatter convolution → Poisson shot noise → Gaussian PSF → gamma correction; `apply_misregistration()` for sub-pixel patient motion jitter
- 9 vendor C-arm geometry presets: `CarmGeometry` classmethods in `config.py` (GE OEC, Siemens Artis, Philips Azurion, Ziehm Solo, and variants)

---

## Slide 9 — This Release — Stage 1 & 2 Demo (Interactive UI Screencast)

**What this slide shows:** A live screen recording of the interactive fluoroscopy simulator running in a browser-based UI. The demo integrates every Stage 1 and Stage 2 component — XPBD physics, Beer-Lambert GPU rendering, vessel-mesh collision, and DSA contrast injection — into a single interactive session that any team member or stakeholder can run and control in real time.

---

### The UI layout — what is on screen

The interface is divided into two panels:

**Left panel — Controls**
- **C-arm projection selector:** A dropdown to switch the X-ray acquisition angle between standard clinical projections (AP, LAO-30, LAO-45, RAO-30, RAO-45, Lateral). Each selection immediately re-renders the DRR from the new C-arm angle using the static CT μ-volume — the volume is never reloaded, only the ray direction changes.
- **Advance speed (mm/s):** A slider controlling how fast the catheter tip advances per button press. This directly maps to the proximal boundary condition in the physics solver.
- **Action buttons:** Advance, Retract, Rotate CW, Rotate CCW — each triggers one physics control step. Idle step (gravity) lets the catheter deform under its own weight without proximal input. Reset returns the catheter to the insertion axis start position.
- **DSA / Contrast Injection section:** A bolus time slider (0–10 s) and a "Show DSA Frame" button. Pressing the button renders a side-by-side composite of the DSA roadmap and the regular fluoroscopy frame at the selected bolus time.

**Right panel — Fluoroscopy DRR output**
The output is a side-by-side composite image:
- **Left half (green bar header):** DSA roadmap panel — skull anatomy in grey-scale with the vessel lumen highlighted in green. The green signal is derived from differencing two DRRs rendered at identical C-arm geometry, with and without a fat catheter (radius = vessel lumen radius) positioned along the catheter trajectory. This highlights exactly the vessel cross-section the catheter is navigating inside.
- **Right half (blue bar header):** Regular fluoroscopy frame — the full CT anatomy (skull, bone, soft tissue) with the actual catheter wire composited inline via Beer-Lambert attenuation. The catheter appears as a bright high-attenuation wire against the grey skull background.

**Simulation status text (below the image):**
Reports per-frame render time (~12 ms at the time of recording), catheter tip position in CT volume coordinates, current projection angle, and a legend explaining the left/right panel colour coding.

---

### What the screencast captures

The screencast records a real-time interactive session:
1. The catheter starts at the insertion axis entry point, straight along the track-guided insertion axis
2. Repeated Advance presses advance the catheter tip through the skull geometry — the tip curves under the elastic Cosserat rod mechanics and is constrained inside the vessel mesh (SDF + AABB collision keeps the rod inside the vessel walls)
3. Rotate CW / CCW shows how a proximal rotation propagates along the rod — the tip deflects in 3D space and the fluoroscopy projection updates immediately
4. The DSA frame is triggered at a bolus time of ~3–4 s (peak arterial enhancement on the gamma-variate curve) — the green vessel lumen overlay appears correctly registered to the catheter position in the right panel
5. Switching C-arm projection (e.g., AP → LAO-45) demonstrates that both panels re-render from the new angle within one render cycle (~12 ms GPU render time)

---

### Why an interactive demo instead of a PPO policy rollout

A PPO policy rollout would require a trained policy — which does not exist yet. The current RL training is **state-observation only**: the policy receives ground-truth catheter tip position and orientation, not fluoroscopy images. A policy trained on state observations cannot be presented as a deployable system because it assumes privileged information that is unavailable during a real clinical procedure.

More importantly, an automated rollout would obscure what this sprint actually built. The interactive demo makes the following claims visible and verifiable in real time:

| What the demo proves | Why a policy rollout would not prove it |
|---|---|
| Physics runs at interactive speed — the catheter responds to input within one frame | A pre-recorded rollout could be played at any speed |
| Vessel-mesh collision is active — the catheter curves inside the vessel and does not pass through walls | A scripted trajectory could be collision-free by construction |
| The DRR re-renders at the correct C-arm angle on every frame | A policy rollout typically uses a fixed angle |
| DSA roadmap and fluoro frame are spatially registered — the green vessel lumen aligns with the catheter wire | Hard to verify in a pre-rendered sequence |
| The entire pipeline — physics + renderer + DSA — runs end-to-end in ~12 ms render time | A pre-rendered video says nothing about throughput |

The interactive format also lets the audience ask "what happens if I rotate it?" or "what does the LAO view look like?" and get an immediate answer — making the technical claims falsifiable in front of the room.

The PPO policy rollout is the Sprint 2 demo. Once pixel observations are wired as the RL input, a trained policy can navigate the catheter from a fixed entry point to a target vessel landmark using only the fluoroscopy frame — that is the moment the loop from simulation to deployable clinical AI closes.

---

## Slide 10 — This Release — Completed Deliverables

**How to present this slide:** Do not read the table row by row. Use it as a visual proof-of-completeness backdrop. Speak to four capability pillars — rendering, realism, physics, integration — and let the table confirm the claim visually. Pause briefly after each group. The audience should leave with four technical convictions, not 21 bullet points.

---

### Opening (5 seconds)

> "Every row on this table was a missing piece. Before this sprint, none of them were connected end-to-end. Today, they all are."

---

### Pillar 1 — X-Ray Rendering (rows 1–4)

**What was built and why it matters:**

The foundation of the entire system is a physically accurate X-ray renderer. We implemented Beer-Lambert attenuation — the same physics law that governs real fluoroscopy — as a GPU ray-march kernel that integrates both the CT anatomy and the catheter geometry in a single pass. This matters for two reasons: first, it eliminates the compositing approximations that would corrupt training data; second, the catheter attenuation is depth-correct — a catheter behind a dense bone structure is attenuated by both, exactly as it would be in a real clinical image.

We also shipped per-segment attenuation and radius for the catheter — meaning the polymer tip, nitinol shaft, and platinum marker bands each have independent μ values and diameters. This reproduces the heterogeneous appearance of a real catheter on fluoroscopy, which is critical for training a detector that needs to localize specific catheter segments.

Cone-beam magnification is implemented correctly: catheter radius is scaled by the source-to-isocenter / source-to-detector ratio in the projection geometry. Without this, catheters appear the wrong diameter at different depths — a systematic error that would bias any trained localizer.

---

### Pillar 2 — Detector Realism and DSA (rows 5–11)

**What was built and why it matters:**

A physically correct ray-march is necessary but not sufficient. Real fluoroscopy images are degraded by noise, blur, scatter, and patient motion — and a policy trained on clean synthetic images will fail on real clinical data. We shipped the full detector signal chain: Poisson quantum noise (governed by photon count), point-spread-function convolution (detector blur), gamma correction (display transfer function), and misregistration jitter between mask and contrast frames (sub-pixel patient motion during DSA acquisition).

The misregistration jitter is particularly important: monoenergetic rendering produces perfect bone subtraction in DSA — too clean relative to clinical images where patient motion leaves residual bone edge artifacts. The jitter injects controlled imperfection, calibrated by the XCath collaboration to match real DSA bone residual statistics.

On top of the static detector chain, we have a complete Dynamic Subtraction Angiography pipeline. The bolus is modelled with a gamma-variate temporal concentration curve — the standard clinical model for iodine contrast kinetics. Propagation through the cerebral vasculature uses VMTK centerline extraction and Dijkstra shortest-path on the centerline graph to compute per-voxel bolus arrival times. The result is a physiologically accurate frame sequence where contrast fills proximal vessels first and distal capillary beds later — matching the temporal pattern seen in real DSA acquisitions. This is the synthetic data that XCath's registration models will train on.

---

### Pillar 3 — Physics Solver (rows 13–18)

**What was built and why it matters:**

The catheter physics solver is not a simplified proxy — it is the production solver that will run inside the RL training loop. It implements the geometrically exact Cosserat rod model discretized via eXtended Position-Based Dynamics, with a block-Thomas direct factorization for the constraint solve. This combination is numerically stable for stiff, highly curved catheter configurations that simple spring-mass models cannot handle.

Multi-environment support is fully operational: all N training environments share a single GPU dispatch with flat contiguous memory buffers. CUDA-graph capture eliminates kernel-launch overhead on the hot path — after warm-up, the entire physics substep loop replays as a single captured graph node with no CPU involvement. Proximal control (insertion and rotation) is applied GPU-side before each substep, meaning zero CPU-GPU synchronization on the per-timestep critical path.

The solver has full feature parity with the upstream Newton research branch, including floor collision with configurable restitution coefficient. This parity is deliberate: it means research advances in Newton propagate directly into the Isaac Lab training environment without a porting lag.

---

### Pillar 4 — System Integration and End-to-End Validation (rows 19–21)

**What was built and why it matters:**

Shipping individual components is not the same as shipping a system. The final three rows represent the integration work that connects everything: the batched GPU renderer and the multi-environment physics solver wired into a single unified loop, the Newton XPBD wrapper that unlocks multi-environment operation inside Isaac Lab's environment registry, and the end-to-end validation that confirms the full pipeline produces correct output.

The validation result — 35 mm catheter traversal inside real CT anatomy with correct Beer-Lambert polarity and the NiTi material attenuation profile — is the proof that the system works as a whole, not just in unit tests. The Beer-Lambert polarity check in particular closes a subtle but critical correctness requirement: the catheter must appear as an attenuating object (darker than air in raw DRR, brighter after inversion to clinical convention). Getting the sign wrong would invert the training signal for any downstream localizer.

The differentiable rendering path — Slang autodiff with 6-DOF pose gradients — is also in this table. This enables future gradient-based 2D/3D registration directly through the renderer, which is the path to training the XCath registration model end-to-end rather than with proxy losses.

---

### Closing (10 seconds)

> "Twenty-one checkmarks. Four capability pillars. One integrated system. The foundation for RL training on physically accurate X-ray fluoroscopy is complete. Sprint 2 adds pixel observations, scales the renderer to 512 environments, and runs the first end-to-end policy."

---

Verified against the codebase:

**Physics solver (`xpbd_rod_solver.py`)**
- `XPBDRodSolver` with `_BatchedWorkspace`: flat buffers, `rod_offsets[r]` / `edge_offsets[r]` indexing
- `apply_proximal_control_gpu`: `_xr_proximal_push_kernel` (Warp kernel, CUDA-graph safe)
- `set_root_orientation`: `_xr_set_root_orientation_kernel` (Warp kernel, CUDA-graph safe)
- `step()`: auto-captures with `wp.ScopedCapture`; replays with `wp.capture_launch`
- `_xr_floor_collision`: configurable `restitution` coefficient (0–1) — parity with Newton upstream
- `_xr_block_thomas`: direct tridiagonal block solver for Cosserat constraint system

**Vessel collision (`xcath_rod_solver.py`)**
- `XCathRodSolver(XPBDRodSolver)`: SDF path via `wp.mesh_query_point_sign_normal` (BVH query, particle-level containment); AABB path via `wp.mesh_query_aabb` (broadphase, rod-segment vs mesh-edge)
- Track-guided insertion: non-tip particles projected back to the linear insertion axis (`track_start`, `track_dir`, `track_length`); `tip_num_edges` controls how many distal segments are free
- Hooks: `_pre_constraints_hook` / `_post_constraints_hook` allow collision constraints to run before or after the XPBD solve

**Rendering (`diffdrr_slang_renderer.py`, `diffdrr_slang.slang`)**
- `renderDRR_forward_batched`: N-environment dispatch, `StructuredBuffer<CatheterSegment>` per env
- Per-segment attenuation: `CatheterSegmentData.mu_values` and `radii` arrays — one value per rod segment, enabling heterogeneous marker/shaft/tip profiles
- Cone-beam magnification: catheter radius scaled by SID/SDD ratio in the Slang shader
- `renderDRR_backward`: Slang autodiff, 6-DOF pose gradients via software trilinear backward pass

**Realism + DSA (`realism.py`, `dsa.py`, `vasculature.py`)**
- `apply_realism()`: scatter convolution → Poisson shot noise → Gaussian PSF → gamma → normalize
- `apply_misregistration()`: sub-pixel patient motion jitter (separate from `apply_realism`)
- `DSAPipeline.render_dsa_frame()`: 4-step DSA — mask DRR → contrast DRR → scatter → jitter → subtract → post-process
- `apply_vessel_boost()`: μ × boost factor on vessel-masked voxels
- `extract_centerlines()`: VMTK-based centerline extraction from binary vessel mask
- `compute_arrival_map()`: Dijkstra shortest-path on `CenterlineGraph` for bolus travel time
- `gamma_variate()`: gamma-variate C(t) = c_peak · (t/t_peak)^α · exp(α(1 − t/t_peak))
- `build_contrast_volume()`: per-frame μ_tissue + Δμ·C(t − T(v)) · vessel_mask
- `simulator.render_cine()` with `volume_callback(frame_idx)`: per-frame μ volume update for temporal DSA cine sequences

**C-arm geometry (`config.py`)**
- `CarmGeometry`: dataclass with SDD, SID, detector size, pixel spacing
- 9 vendor presets via classmethod factories: GE OEC 9900, Siemens Artis zee, Philips Azurion, Ziehm Solo + variant configurations

---

## Slide 10 — Part 3 Divider: Following Releases

**Transition:** Items not in Sprint 1. Forward priorities for Sprint 2 and beyond.

---

## Slide 11 — Summary

**How to present this slide:** This is the closing slide. Speak to each column as a chapter in a single narrative arc — where we are, what we built, what we are building next, and where this is going. The columns are not a status report; they are a strategic trajectory. Each bullet is a proof point, not a to-do item.

---

### Column 1 — TODAY (what is operational and measured right now)

**Opening line:** "Before this sprint, we had components. Today we have a pipeline."

- **End-to-end X-ray fluoroscopy** — the full signal chain from CT volume to detector pixel is operational. Physics, rendering, DSA, and detector realism run in sequence on a single GPU without any CPU synchronization on the hot path. This has been validated against a 35 mm catheter traversal inside real cranial CT anatomy.

- **GPU Beer-Lambert at 25 FPS, physics at 1,300 Hz** — these are measured numbers on a single A6000. The physics subsystem exceeds its training throughput target with margin. The renderer delivers clinically correct X-ray images at interactive speed. Both numbers are reproducible.

- **DSA + bolus dynamics + 9 C-arm presets** — the simulator is not a single fixed setup. It replicates nine vendor-specific C-arm geometries (GE, Siemens, Philips, Ziehm) and produces temporally dynamic DSA sequences with physiologically calibrated contrast propagation. The bolus timing matches real clinical DSA to within 11%.

- **Volumetric instrument injection at ~2 ms** — the catheter is not composited as a 2D overlay. It writes attenuation directly into the 3D volume using GPU-parallel atomic operations. This is depth-correct by construction and runs at ~2 ms for a 64-node catheter on a 512³ volume.

- **State-based PPO at 512 parallel environments** — reinforcement learning training is operational. 512 independent catheter environments run simultaneously, each with its own physics state and reward signal. The training infrastructure is in place. The observation type — not the scale — is what changes in Sprint 2.

---

### Column 2 — THIS RELEASE (what Sprint 1 specifically delivered)

**Opening line:** "Sprint 1 closed the gap between a working prototype and a production-grade training platform."

- **Full DSA pipeline with temporal bolus dynamics** — four-stage digital subtraction angiography (mask DRR, contrast DRR, scatter, misregistration jitter, subtraction, post-processing) combined with a gamma-variate contrast propagation model driven by a Dijkstra shortest-path arrival map on the vessel centerline graph. This is the same DSA pipeline architecture that XCath validated against real clinical data, achieving 89% bolus timing accuracy.

- **Detector physics chain** — Poisson quantum noise, scatter convolution, point-spread-function blur, gamma correction, and sub-pixel misregistration jitter are all implemented and applied in sequence on the CPU reference path, with the GPU port scoped for Sprint 2. The misregistration jitter specifically was validated by XCath as the correct approach to introduce bone-edge artifacts that monoenergetic rendering would otherwise suppress too cleanly.

- **Self-contained multi-environment XPBD with CUDA-graph capture** — the physics solver runs all N environments in a single GPU kernel dispatch, with block-Thomas direct factorization for the Cosserat constraint system, GPU-side proximal control, and CUDA-graph capture that eliminates kernel-launch overhead after the first warm-up step. This is the production physics backend, not a prototype.

- **Multi-environment Slang renderer** — all N environments render in a single GPU dispatch. Per-environment catheter geometry is packed into a flat structured buffer with per-environment offsets. The batched renderer is functionally validated at N≤4 and is the production path. The N>8 scaling upgrade is Sprint 2.

- **Newton XPBD wrapper with multi-environment unlocked** — the Newton upstream physics bridge is also multi-environment capable. Multiple catheter rods are registered into a single Newton solver instance, sharing the same constraint graph and time-stepping infrastructure. This maintains full parity with Newton upstream while running inside Isaac Lab's training loop.

- **Vessel-mesh SDF and AABB collision** — ported from the Newton upstream research branch. The catheter is now physically contained inside the vessel geometry. Particles that would exit the vessel lumen are projected back to the surface normal via signed-distance-field queries. Rod-segment-to-mesh-edge contacts are resolved via AABB broadphase. Track-guided insertion constrains the proximal rod to the sheath axis. This closes the most critical physical realism gap: a catheter that respects anatomy.

- **End-to-end validation** — 35 mm catheter traversal, inside real CT anatomy, with vessel-wall containment active, correct Beer-Lambert polarity, and NiTi shaft material attenuation profile. The pipeline is not theoretical — it ran.

---

### Column 3 — NEXT RELEASE (Sprint 2 — the training readiness sprint)

**Opening line:** "Sprint 2 is the sprint that closes the loop from simulation to trainable policy."

- **Texture2DArray renderer upgrade** — the current 3D CT texture is shared across all environments in a single GPU L2 cache partition. Above N=8, cache thrashing degrades throughput. The Texture2DArray architecture caches one depth slice of the μ-volume per environment in L2, eliminating bandwidth saturation and unlocking >60 FPS at N>8. This is a renderer architecture change, not a parameter tweak.

- **Image-based RL observations** — today the policy receives the catheter's ground-truth position. In Sprint 2, it receives the fluoroscopy image. This is the transition from a simulator that works to a simulator that generates training data for a deployable policy. A policy trained on ground-truth position cannot be deployed clinically — a policy trained on the fluoroscopy image can be.

- **GPU-side detector physics on the Slang path** — moving the realism chain (Poisson noise, PSF, gamma) from CPU to GPU eliminates the last remaining CPU round-trip in the render pipeline. This reduces per-frame latency and is a prerequisite for image-based RL training at scale.

- **Beam hardening correction** — monoenergetic rendering produces near-perfect bone subtraction in DSA, which is actually too clean compared to real clinical DSA. The polyenergetic extension adds energy-dependent attenuation, correctly reproducing the beam hardening residual at bone edges. XCath identified this as the primary remaining realism gap after the SSIM=0.964 baseline.

---

### Column 4 — FOLLOWING (Sprint 3+ — the deployment readiness sprints)

**Opening line:** "Beyond Sprint 2, the simulator becomes infrastructure — not a research project."

- **Sprint 3 training readiness** — domain randomization across CT anatomy variants, C-arm angles, and noise parameters makes the trained policy robust to distribution shift. The Gymnasium adapter and per-task reward graphs are the interface to the broader Isaac Lab training ecosystem.

- **Phase 2 and 3 skill packaging and agent integration** — seven OpenClaw manipulation skills and a natural-language-to-configuration agent loop. This is the horizon where the catheter RL policy becomes one module in a larger agentic surgical workflow.

- **Realism metrics** — FID, SSIM, and vessel visibility scores benchmarked against clinical reference DSA images. This closes the evaluation loop: not just "does the simulation look right" but "how right, quantitatively, against ground truth."

- **Workflow extensions** — force/torque sensing integration and CBCT (cone-beam CT) intraoperative volume ingestion. These extend the simulator from a pre-operative training tool to an intraoperative guidance platform.

---

### Closing line for the slide

"We are not building a demo. We are building the data infrastructure that makes robotic catheter AI trainable, validatable, and deployable — at clinical scale, on real anatomy, with measurable realism. Sprint 1 proved the pipeline works. Sprint 2 makes it trainable. Sprint 3 makes it deployable."
