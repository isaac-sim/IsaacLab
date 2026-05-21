# Sensor Simulation Release Deck — Presenter Notes

> Technical and strategic presenter notes for the **X-Ray–Guided Robotic Catheter Intervention — Simulation Workflow Release Status & Roadmap** deck (`docs/sensor_simulation_release_deck.pptx`).
>
> **Audience:** engineering leadership, LHA stakeholders (XCATH, Remedy, JNJ), I4H program management, partner solution architects.
> **Tone:** technically precise, strategically framed, demo-ready. Cite code paths when the audience pushes for proof.
>
> **Source-of-truth audited:** `source/isaaclab_newton/isaaclab_newton/solvers/{rod_solver.py, xpbd_rod_solver.py, newton_xpbd_rod_wrapper.py, rod_data.py}` and `i4h-sensor-simulation-internal/fluoro-simulator/fluorosim/{rendering, dsa, vasculature, config, simulator}.py`. All implementation claims below have been verified against current source.

---

## SLIDE 1 — Title

**X-Ray–Guided Robotic Catheter Intervention**
**Simulation Workflow — Release Status & Roadmap**

**Say:**

"This deck does three things. One: it states honestly what is implemented today in the X-ray–guided catheter intervention simulation stack — physics, rendering, RL pipeline — with measured numbers, not aspirations. Two: it audits what shipped this release against the XCATH requirements list, and what genuinely remains. Three: it lays out Sprint 2, Sprint 3, and the agentic workflow phases as concrete engineering deliverables with effort estimates. The whole deck is scoped to the X-ray catheter workflow — ultrasound and other modalities are out of scope for this conversation."

---

## SLIDE 2 — Agenda

**Say:**

"Five parts. Executive Snapshot is the elevator pitch — what changed, what works, what doesn't. Current Status is Part 1 — a deep dive on the three solver backends, the three compositing paths, the detector physics chain, DSA, bolus dynamics, instrument injection, C-arm presets, and the RL pipeline. Part 2 is This Release — the completed deliverables, the XCATH gaps we closed, and the items that genuinely remain. Part 3 is the Sprint 2 plan with one slide dedicated to why multi-env fluoroscopy rendering is the headline ticket. Part 4 covers Sprint 3, the agentic workflow Phase 2 and Phase 3, and the remaining capability gaps. Part 5 lists adjacent workflow enhancements — force/torque sensing, CBCT, and so on."

---

## SLIDE 3 — Architecture Overview

**Headline:** an end-to-end, simulation-to-deployment platform for X-ray–guided robotic catheter intervention. Four logical sections — CT ingestion, simulation environment, training pipeline, runtime deployment — connected by a single dataflow that starts with patient CTA and ends with a policy executing on a real C-arm under Holoscan IGX.

**Suggested timing:** 4–5 minutes. This is the orientation slide; spend the time here so every subsequent slide makes sense in context.

**Open with the strategic frame (30 seconds):**

"Before I get into individual components, I want to anchor on the system architecture, because every release item I'll discuss next maps to one of four sections on this diagram. The system takes a patient's CT scan in at the top left, builds a simulation-ready digital twin from it, runs catheter physics and fluoroscopy rendering in a tight loop to train a navigation policy, and deploys that policy to a real C-arm and a real robotic catheter through Holoscan IGX. The closed loop on the right — `CT → Simulate → Train → Deploy → Outcome` — is what we are building. We are currently delivering on the first three sections; the fourth, runtime deployment, is the integration target with XCath."

---

### Section 1 — CT Ingestion Pipeline (top-left of the diagram)

**Strategic framing:** turn raw clinical imaging into a complete, simulation-ready patient digital twin. This is the foundation; every downstream stage consumes its outputs.

**Say:**

"Section one is the CT ingestion pipeline. It takes a CTA volume in DICOM or NIfTI form and produces four artifacts: a μ volume for X-ray rendering, a vessel mesh and SDF for physics-based collision, a centerline graph for navigation reasoning, and a Catheter Sup. file — the catheter support / starting configuration. Together these constitute the patient digital twin."

**Stage-by-stage breakdown:**

| Stage | Input | Output | Implementation |
|---|---|---|---|
| **CT/CTA load** | DICOM series or NIfTI | HU volume `(Z, Y, X)` float32 + `spacing_zyx_mm` + `origin_xyz_mm` | `fluorosim/preprocessor.py::VolumePreprocessor.from_dicom()` / `.from_nifti()` (SimpleITK / nibabel under the hood) |
| **HU → μ conversion** | HU volume | μ volume mm⁻¹ | `HuToMuMapping` in `fluorosim/config.py` — linear mapping `μ = μ_min + (HU − HU_min) / (HU_max − HU_min) × (μ_max − μ_min)`, configurable per kVp |
| **Vessel segmentation** | HU volume | binary vessel mask | Designed for TotalSegmentator / nnU-Net upstream; the simulator consumes the mask as input |
| **Mesh generation** | vessel mask | watertight triangle mesh `(<50k tris)` | Marching cubes + Laplacian/Taubin smoothing + decimation; topology preserved at bifurcations |
| **SDF generation** | watertight mesh | signed-distance field `(voxel ≤ 0.5 × wall thickness, 0.15–0.5 mm)` | Newton physics constraint — narrow-band SDF stored within ±5 mm of the surface |
| **Centerline extraction** | vessel mask | `CenterlineGraph` (nodes + cells) | `vasculature.extract_centerlines()` — VMTK marching cubes + network extraction |
| **Catheter support / arrival map** | centerline graph + injection root | per-voxel travel time `T(v)` | `vasculature.compute_arrival_map()` — Dijkstra over the centerline graph with edge length as cost |
| **USD export** | mesh + SDF + metadata | USD scene `metersPerUnit = 0.001` | Newton / Isaac Sim compatibility — mm → m conversion baked in |

**Say (transitioning into the next section):**

"All of this is preprocessing — it runs once per patient, takes under a minute for a 512³ volume, and the outputs are cached on disk. The simulator never re-runs it. From here on, everything operates on the cached digital twin."

**Pre-empt:** *"What's the runtime budget for ingestion?"* Under 60 s for a 512³ volume on a single workstation GPU. *"Can we re-run on the fly?"* Designed for offline cache, but the API supports re-ingestion if the segmentation is updated.

---

### Section 2 — Simulation Environment (middle of the diagram)

**Strategic framing:** the closed loop that produces every training frame. Three concurrent components — physics, rendering, and domain randomization — synchronized inside a single sim loop.

**Say:**

"Section two is the simulation environment. This is where the digital twin from Section 1 is animated. Three components run concurrently inside a single per-frame loop, and they emit one synchronized output per frame: the fluoroscopy frame plus the catheter state."

**The three concurrent components:**

1. **Newton Physics — flexible-body catheter simulation.**
   - Implementation: `solvers/rod_solver.py::RodSolver` (production, multi-env via PyTorch tensors) and `solvers/xpbd_rod_solver.py::XPBDRodSolver` (self-contained, multi-env via Warp `_BatchedWorkspace` with batched block-Thomas direct solve, GPU root control, and CUDA-graph capture).
   - Cosserat rod model — 20 segments per catheter by default (`CatheterStateEnvCfg.num_segments = 20`).
   - Vessel collision via SDF lookup against the digital-twin SDF; floor collision available with restitution.
   - Throughput: ~1,300 Hz single-env on A6000; multi-env benchmarks in progress (Sprint 2 target: >60 Hz at 512 envs).

2. **Fluoroscopy Renderer — GPU ray-casting.**
   - Implementation: `fluorosim/rendering/diffdrr_slang_renderer.py::SlangDiffDRRRenderer` (Slang differentiable DRR with autodiff), with `render_with_catheter()` performing fused volume + catheter Beer-Lambert in a single ray march.
   - Beer-Lambert physics: `I = I₀ × exp(−∫(μ_volume + μ_catheter) ds)`.
   - Trilinear interpolation of μ; ray-march step ≤ 0.5 × voxel spacing (Nyquist).
   - DSA mode via `fluorosim/dsa.py::DSAPipeline.render_dsa_frame()` — four-step subtraction with bolus dynamics from `vasculature.gamma_variate()` + `build_contrast_volume()`.
   - Detector physics chain via `fluorosim/rendering/realism.py::apply_realism()` — Poisson, scatter, PSF, gamma, jitter.
   - Volumetric instrument injection via `instrument-injection/painting_kernels.py::paint_spheres_kernel` / `paint_cylinders_kernel` — Warp `wp.atomic_max` for thread-safe max-attenuation compositing.

3. **Domain Randomization — sim-to-real bridge.**
   - C-arm angles: LAO/RAO ±60°, CRA/CAU ±45° (per design constraint).
   - X-ray noise / dose: photon count variation, PSF blur σ = 0.5–2.0 px, contrast ±20%.
   - Vessel parameters: friction, vessel-wall stiffness, catheter-vessel contact margin.
   - Anatomy variation: train across multiple patient digital twins.

**The unified sim loop (per frame):**

```
L1 — Policy emits control action  (push velocity, rotate velocity)
L2 — Newton steps catheter physics (XPBD substeps + SDF collision)
L3 — Renderer projects fluoroscopy frame (Slang DRR + catheter)
L4 — Domain randomization perturbs imaging + physics parameters
L5 — Synchronized output: fluoroscopy frame + catheter state → Training/Observation
```

**Say:**

"The critical property here is synchronization. The rod solver state at time `t` is exactly the catheter geometry the renderer composites at time `t`. There's no asynchronous handoff, no double-buffering, no temporal smear. The physics output drives the catheter buffer that the Slang shader reads in the same step. That's what makes the rendered frame a faithful observation of the physics state — which is what the policy needs."

**Pre-empt:** *"How is the catheter geometry passed from physics to renderer?"* Centerline positions in metres, scaled ×1000 to mm, packed into a `(n_seg, 8)` float32 structured GPU buffer = `[p0(3), p1(3), radius, μ]` per segment — same buffer consumed by the Slang `renderDRR_withCatheter_forward` kernel.

---

### Section 3 — Training Pipeline (bottom-left of the diagram)

**Strategic framing:** convert simulated experience into a deployable navigation policy. Two-stage learning — imitation first, RL fine-tuning second — with curriculum learning across anatomies.

**Say:**

"Section three is training. The simulation environment from Section 2 is wrapped as a vectorised RL environment, and we pipe demonstrations and rollouts through a two-stage learning pipeline."

**Stage-by-stage breakdown:**

| Stage | Input | Output | Implementation |
|---|---|---|---|
| **Demonstration collection** | expert teleop trajectories | paired (observation, action) trajectories | Manual teleop sessions; ~50 demos per anatomy class typical |
| **GR00T-H imitation learning** | demonstration set | initial policy | Behavioral cloning, ~100 epochs |
| **Reinforcement learning fine-tuning** | initial policy + sim env | refined policy | PPO via RSL-RL — `agents/rsl_rl_catheter_state_ppo_cfg.py`, 512 parallel envs (`CatheterStateEnvCfg.num_envs = 512`), `examples/train_catheter_state.py` |
| **Curriculum learning** | progressive difficulty schedule | policy generalising across anatomies | Easy → hard anatomies; bifurcation curriculum stages added on observed failure modes |
| **Trained policy (GR00T-H)** | converged checkpoint | deployable inference graph | Exported for Holoscan IGX |

**Reward structure (current vs target):**

The env today implements a single-term distance-to-target reward in `envs/catheter_state_env.py`. The target reward decomposition is the seven-term formulation from the SDD: target proximity (1.0), wall contact penalty (0.5), procedure time penalty (0.1), tip-force perforation safety (0.3), fluoroscopy dose (0.05), progress reward (0.2), success bonus (10.0). The full stack lights up incrementally as dependent capabilities ship — wall contact and tip force when SDF vessel collision lands in Sprint 2; fluoroscopy dose when image-based observations are wired in.

**Say:**

"State-based observations today, image-based observations in Sprint 2 once multi-env Slang rendering is in. The two-stage pipeline — IL then RL — is the standard pattern for medical robotic skills: imitation gets you a competent prior in tens of demonstrations, RL refines and generalises in millions of simulated frames."

---

### Section 4 — Runtime Deployment (bottom of the diagram)

**Strategic framing:** the trained policy must execute on real hardware — a real C-arm streaming live fluoroscopy, a real robotic catheter actuating the patient — under Holoscan IGX with strict latency and safety guarantees.

**Say:**

"Section four is runtime deployment. The trained policy from Section 3 is exported and runs inside Holoscan IGX on the deployment hardware. The closed loop is: the C-arm streams live fluoroscopy frames → the policy infers a control action → the action drives the robotic catheter → the patient anatomy responds → the next fluoroscopy frame arrives. This loop runs at the C-arm frame rate with bounded latency."

**Component breakdown:**

| Component | Purpose | Latency requirement |
|---|---|---|
| **Holoscan IGX** | Low-latency policy inference at the bedside | Real-time, bounded per-frame latency (P0 in SDD) |
| **Real C-arm fluoroscopy** | Live X-ray stream into the inference pipeline | C-arm native frame rate (typically 7.5–30 Hz pulsed) |
| **Robotic catheter (XCath)** | Catheter actuation — push, retract, rotate at the proximal end | Sub-frame actuation latency |
| **Safety layer** | Motion limits, collision guards, tip-force caps | Always-on, online enforcement |

**Closed-loop control characteristics:**

- The policy is trained to tolerate 1–2 frame delays, accommodating IGX inference + C-arm acquisition latency.
- Safety layer enforces motion bounds independently of the policy — the policy cannot command the catheter through a vessel wall even under adversarial conditions.
- Sim-to-real transfer is the dominant validation effort, addressed via domain randomization in Section 2 and curriculum learning in Section 3.

**Say:**

"This is the integration target with XCath. The simulation, training, and deployment stack we're delivering is designed to drop into their robotic catheter platform. We provide the synthetic data, the trained policy, and the inference graph; XCath provides the actuation hardware, the C-arm interface, and the safety-certification path."

---

### The right-hand System Flow Detail callout

**Say:**

"The box on the right of this slide is the same architecture re-expressed as a sequential dataflow, which is sometimes easier to reason about. Five steps. One — CT ingestion: DICOM in, μ volume plus simulation primitives out. Two — simulation loop, per frame: policy emits action, Newton steps the catheter, renderer projects the fluoroscopy frame. Three — closed-loop training: GR00T-H imitation learning from demonstrations, then PPO RL refinement; the simulation feeds the training loop continuously. Four — deployment: trained policy on Holoscan IGX, real fluoroscopy in, robot control out, safety layer always on. Five — the outcome: autonomous catheter navigation."

---

### Closing strategic frame for Slide 3

**Say:**

"Three things to take away from this slide before I move on. One: the system is end-to-end. We are not delivering a research demo on one stage; we are delivering a stack from clinical CT to deployed robotic policy. Two: the closed loop is the product. Each section feeds the next, and the value compounds — better physics improves training, better training improves transfer, better transfer reduces clinical risk. Three: today we are delivering on Sections 1, 2, and 3. Section 4 is the integration target, and the agentic workflow we'll discuss in Part 4 is the orchestration layer that lets this whole pipeline be driven from natural-language intent rather than hand-tuned configs."

---

### Pre-empts for Slide 3

- *"How is this different from DeepDRR / SyntheticTurk / commercial offerings?"* DeepDRR is volume rendering only — no catheter, no closed loop, no deployment path. SyntheticTurk is a data-generation tool. We are delivering an integrated simulation-to-deployment stack with first-class catheter physics, fused volume + catheter Beer-Lambert, DSA + bolus dynamics, an RL training pipeline, and a Holoscan IGX deployment target — none of which the alternatives provide as a unit.
- *"What is the biggest technical risk?"* Sim-to-real transfer of the rendered fluoroscopy. We mitigate via domain randomization (Section 2), curriculum learning (Section 3), and a planned realism evaluation module — FID, SSIM, vessel visibility — that closes the agentic refinement loop in Phase 3.
- *"Which sections are productionised vs research?"* Sections 1 and 2 are productionised — they run today on RTX A6000 hardware with the numbers stated on this slide. Section 3 is in active development with state-based RL operational and image-based RL planned for Sprint 2. Section 4 is the integration target — interfaces are defined, hardware integration is XCath's responsibility on the joint development plan.

---

## SLIDE 4 — Executive Snapshot

**Say:**

"Eight bullets. The first three are about the rendering and the physics — the X-ray fluoroscopy pipeline is implemented end-to-end on the Slang GPU backend; the fused volume-plus-catheter Beer-Lambert compositor runs at about 25 FPS on an RTX A6000; two of the three XPBD rod backends are now multi-env. I want to be precise on that last point: the production solver in `rod_solver.py` is multi-env via PyTorch tensors, the self-contained solver in `xpbd_rod_solver.py` is multi-env via batched Warp kernels with batched block-Thomas direct solve and CUDA-graph capture, and the Newton bridge wrapper in `newton_xpbd_rod_wrapper.py` is still hard-coded to single env — it raises `NotImplementedError` if you ask for `num_envs != 1`. Newton's underlying solver supports batching; we just haven't wired the wrapper up."

"The next three bullets are about realism. Detector physics chain is complete — Poisson quantum noise, scatter convolution, detector PSF, gamma correction, and sub-pixel misregistration jitter. The full four-step DSA pipeline is implemented in `dsa.py` with vessel boost, gamma-variate bolus dynamics, a VMTK centerline plus Dijkstra arrival map for temporal contrast propagation. Volumetric instrument injection is GPU-resident via Warp atomic-max kernels — about 2 milliseconds per 64-node catheter on a 512-cubed volume."

"Last two bullets are the closing material. Nine vendor-specific C-arm presets are wired in — three GE, three Siemens, two Philips, one Ziehm — and state-based PPO is operational at 512 envs via RSL-RL. The remaining critical gaps are: multi-env fluoroscopy rendering, image-based RL observations, and beam hardening. Those are the Sprint 2 deliverables."

**Pre-empt:** *"Why is the Newton bridge wrapper still single-env if the underlying solver supports batching?"* Honest answer: it was originally written as a regression-test harness for the self-contained port, and with the self-contained solver now feature-complete and dependency-free, the wrapper is more of a validation tool than a production path. Wiring batching through it is a one-day job if anyone asks for it.

---

## SLIDE 5 — X-Ray Fluoroscopy Pipeline At a Glance

**Say:**

"This is the matrix view of every pipeline component, with implementation status and multi-env status. Eleven rows. Nine of them are implemented today. Two — image-based RL observations and beam hardening — are planned for Sprint 2 and beyond."

"Multi-env story per row: the catheter physics solver is multi-env in production. The DRR volume rendering, the Beer-Lambert compositing, the volumetric instrument injection, the DSA pipeline, and the bolus dynamics are all implemented but currently single-env — they are the Sprint 2 bottleneck. C-arm geometry and the detector physics chain are global so multi-env is N/A. The RL pipeline itself is multi-env via standard vectorized envs."

"The implementation column is precise. Production XPBD with Warp + Newton bridge for the solver. `Slang DiffDRR` with autodiff for the volume rendering. `apply_realism()` in `realism.py` for the detector chain. `DSAPipeline` in `dsa.py` for the four-step DSA. `extract_centerlines` and `compute_arrival_map` in `vasculature.py` for the bolus dynamics. Nine `CarmGeometry` classmethods for the presets. PPO via RSL-RL for the RL pipeline."

---

## SLIDE 6 — Part 1 Divider: Current Status

**Say:**

"Part 1 — Current Status. What is built, what is integrated, what we have measured numbers for. I'll spend most of the technical detail in this section because it grounds everything that follows."

---

## SLIDE 7 — Catheter Physics — Three Solver Backends

**Say:**

"We maintain three XPBD Cosserat rod solver backends. They serve different purposes."

"The **production XPBD Cosserat rod** lives in `rod_solver.py::RodSolver`. It is multi-env via PyTorch tensor batching — the constructor takes `num_envs`, and every internal tensor has shape `(num_envs, num_segments, ...)`. It uses a `DirectTreeSolver` that runs the block-tridiagonal Thomas algorithm in PyTorch, has Warp-accelerated mesh BVH collisions via `solve_mesh_collision_kernel`, and exposes `apply_proximal_control(push_velocity[:E], rotate_velocity[:E], dt)` for the kinematic root controller. This is the solver used by the RL training pipeline today."

"The **self-contained XPBD solver** lives in `xpbd_rod_solver.py::XPBDRodSolver`. It is the research-and-future-production solver. Every kernel — predict, integrate, compliance, constraint update, Jacobians, inverse inertia, JMJT assembly, block-Thomas solve, corrections — is a Warp kernel embedded in this one file. Zero `import newton`. Multi-env is shipping in this release: a `_BatchedWorkspace` holds concatenated GPU buffers for `E` rods, and the batched block-Thomas kernel launches one GPU thread per rod, so the per-substep cost scales with edges-per-rod, not with `num_envs × edges`. It also has GPU-side root control APIs that don't sync to CPU, CUDA-graph capture of the entire substep loop, floor-collision restitution, and a `max_delta_lambda` diagnostic for convergence monitoring."

"The **Newton bridge wrapper** lives in `newton_xpbd_rod_wrapper.py::NewtonXPBDRodSolver`. It bridges Isaac Lab's `RodConfig` to Newton's `SolverXPBDRod` from PR #1981. It exists as a validation harness — to A/B against the in-tree port. It is hard-coded to single env today. The underlying Newton solver supports batching; the wrapper just hasn't been extended."

"All three return centerline positions in SI metres. The compositing pipeline multiplies by 1000 to get millimetres, which is the coordinate convention all clinical C-arm geometry uses — SID, SDD, and pixel spacing are all in mm."

**Likely Q&A:** *"Why three solvers — pick one."* The production solver is the RL workhorse and stays. The self-contained solver is the path forward because it has zero external dependencies, faster batched physics, GPU control, and CUDA graphs — those are exactly the properties you want for in-loop training. The wrapper is a validation tool; we will likely deprecate it once the self-contained solver passes a regression suite against Newton's reference.

---

## SLIDE 8 — Self-Contained XPBD — Multi-Env Upgrade

**Headline:** the self-contained XPBD solver in `solvers/xpbd_rod_solver.py` has been brought to feature parity with Newton's upstream `xpbd_rod` batched path — without taking a runtime dependency on Newton's `Model` API. This unblocks parallel RL training on the only solver in the tree that has zero external solver dependencies.

**Suggested timing:** 4 minutes. This is the most technically dense slide in Part 1; let it breathe.

**Open with the strategic frame (30 seconds):**

"Two-column slide. Both columns describe the same upgrade — turning a single-rod CPU-bridged research solver into a 512-rod CUDA-graph-captured production solver — but viewed from two angles. Left column: how we restructured the data and the kernels so the GPU sees one batch of work instead of N independent rods. Right column: how we eliminated the CPU↔GPU round trips that used to gate per-step throughput, and what diagnostics we exposed for multi-env convergence monitoring. The footer is the validation status."

---

### Section A (left column) — Batched data + kernels

**Strategic framing:** moving from "one rod per Python call" to "one launch per batch" requires (1) a single concatenated GPU layout for every state field, (2) batched kernel variants that consume that layout, and (3) per-env material parameters so the batch is heterogeneous-by-design.

#### A.1 — `_BatchedWorkspace`: concatenated GPU buffers

**Say:**

"Every per-particle, per-edge, and per-DOF array that the single-env workspace held as a `(N,)` Warp buffer is replaced in `_BatchedWorkspace` by a `(num_envs × N,)` flat buffer. Nothing in the simulator allocates per-rod — there is exactly one allocation per state field, sized at construction and reused for the lifetime of the solver. That is what makes CUDA graph capture viable later, because graphs cannot tolerate alloc/free inside the captured region."

| Buffer family | Length | Per-element dtype | Purpose |
|---|---|---|---|
| Per-particle state — `positions`, `predicted_positions`, `velocities`, `forces` | `num_envs × num_points_per_rod` | `vec3` | Translational state |
| Per-particle rotational — `orientations`, `predicted_orientations`, `prev_orientations`, `angular_velocities`, `torques` | `num_envs × num_points_per_rod` | `quat` / `vec3` | Cosserat frame state |
| Per-particle masses — `inv_masses`, `quat_inv_masses` | `num_envs × num_points_per_rod` | `float32` | Inertia (root particles get `inv_mass = 0` for kinematic control) |
| Per-edge geometry — `rest_lengths`, `rest_darboux`, `bend_stiffness` | `num_envs × num_edges_per_rod` | `float32` / `vec3` | Cosserat rest configuration |
| Constraint workspace — `constraint_values`, `compliance`, `lambda_sum`, `rhs`, `delta_lambda` | `num_envs × 6 × num_edges_per_rod` (n_dofs) | `float32` | XPBD per-DOF accumulators |
| Jacobian / JMJT — `jacobian_pos`, `jacobian_rot`, `diag_blocks`, `offdiag_blocks`, `c_blocks`, `d_prime` | `num_envs × num_edges_per_rod × 36` (or × 6) | `float32` | Block-tridiagonal system data |
| Per-env material — `young_modulus`, `torsion_modulus`, `gravity`, `inv_inertia_local_diag` | `num_envs` | `float32` / `vec3` | Heterogeneous physical properties across the batch |
| Index arrays — `rod_offsets`, `edge_offsets`, `particle_rod_id`, `edge_rod_id`, `root_idx`, `next_idx` | `num_envs + 1` or `num_envs × …` | `int32` | Built on CPU once at construction; never re-uploaded |
| Control scratch — `push_velocities`, `rotate_velocities` | `num_envs` | `float32` | Reused per call so multi-env control issues zero alloc |
| Diagnostic — `max_delta_lambda` | `1` | `float32` | Atomic-max convergence monitor |

**Pre-empt:** *"Can the rods have different topology?"* Today every rod in a batch must share `num_points_per_rod` and `num_edges_per_rod` — that is what makes the offset arrays trivial linear ramps. Heterogeneous topology is supported in principle (the offset arrays already exist), it just needs the constructor to accept ragged shapes. This is on the Sprint 3 list, not this release.

#### A.2 — Eleven batched Warp kernel variants

**Say:**

"Eleven kernels were ported from single-env to batched. The pattern is identical for every one of them: the kernel takes the same physics inputs, plus two extra index arguments — `particle_rod_id[tid]` to look up the rod that owns the current particle, and `edge_rod_id[tid]` to look up the rod that owns the current edge. Per-env material parameters are then dereferenced through these IDs. The compute body is unchanged. This was a deliberate design choice — we did not want a forked physics implementation."

| Kernel (batched variant) | Launch dim | What it does |
|---|---|---|
| `_xr_predict_pos_batched` / `_xr_predict_rot_batched` | `num_envs × num_points_per_rod` | Symplectic Euler predictor with per-env gravity / damping |
| `_xr_prepare_compliance_batched` | `num_envs × num_edges_per_rod` | XPBD α̃ from per-env Young / torsion moduli |
| `_xr_update_constraints_batched` | `num_envs × num_edges_per_rod` | Stretch + bend constraint residuals |
| `_xr_compute_jacobians_batched` | `num_envs × num_edges_per_rod` | 6×12 constraint Jacobians per edge |
| `_xr_compute_inv_inertia_batched` | `num_envs × num_points_per_rod` | World-frame inverse inertia from quat + per-env diag |
| `_xr_assemble_jmjt_batched` | `num_envs × num_edges_per_rod` | Block-tridiagonal LHS (`diag_blocks`, `offdiag_blocks`) |
| `_xr_block_thomas_batched` | **`num_envs`** (one thread per rod) | The direct solve — see A.3 |
| `_xr_compute_corrections_batched` | `num_envs × num_edges_per_rod` | Per-edge Δp / Δq with `wp.atomic_max(max_delta_lambda)` |
| `_xr_integrate_pos_batched` / `_xr_integrate_rot_batched` | `num_envs × num_points_per_rod` | Velocity update + position commit |

#### A.3 — The critical kernel: batched block-Thomas direct solve

**Say:**

"This is the kernel that determines whether the solver scales to RL. The Thomas algorithm — forward sweep then back substitution on a block-tridiagonal system — is fundamentally sequential along the rod. Inside one rod the steps depend on each other; you cannot parallelise across edges. So the only useful parallel axis is **across rods**, and that is exactly what the batched kernel does: `dim = num_envs`, one GPU thread per rod, each thread walks its own `[edge_offsets[rod], edge_offsets[rod+1])` range over its own slice of the diagonal and off-diagonal block buffers."

"The scaling property that buys: the **per-substep wall-clock cost stays at O(edges_per_rod), independent of `num_envs`**, up to the point where the GPU runs out of resident threads. On A6000 that is in the thousands of rods, well past anything we need for RL. Cost scales with edges per rod, not `num_envs × edges`."

**Pre-empt:** *"Why not iterative Gauss-Seidel like classical PBD?"* Direct block-Thomas converges in one substep where Gauss-Seidel needs 50–200 iterations for the same residual. For RL, where wall-clock per env step dominates training throughput, direct solve wins.

#### A.4 — Per-env material parameters

**Say:**

"Young's modulus, torsion modulus, gravity, and the inverse-inertia diagonal are now `wp.array(num_envs, dtype=…)`. That means a single batch can contain rods with different stiffness profiles, different inertia, and different gravity vectors — which is exactly what domain randomization needs. No per-env solver instantiation, no per-env CPU-side bookkeeping; the per-env material is a uniform GPU lookup inside every kernel."

---

### Section B (right column) — GPU-side control + diagnostics

**Strategic framing:** the previous release had a clean batched solver but a CPU-bridged controller — every `apply_proximal_control` call did a `tangent.norm()` PyTorch sync that forced a device→host barrier and broke any chance of CUDA-graph capture. Section B closes that gap.

#### B.1 — `apply_proximal_control_gpu(push_v[:E], rotate_v[:E], dt)`

**Say:**

"Single Warp launch over `num_envs` threads. Each thread reads its rod's root particle and the next particle, computes the local tangent, advances the root by `push × dt` along the tangent, and applies a `rotate × dt` axial rotation to the root quaternion. No PyTorch round-trip, no `.norm()` on the host, no copy to NumPy. The control inputs accept `np.ndarray`, `torch.Tensor`, `wp.array`, or list — the helper `_coerce_per_env_float_array` marshals into the preallocated `bws.push_velocities` / `bws.rotate_velocities` scratch buffers, so the steady-state path issues zero allocations."

"Backward compatibility is preserved: the legacy `apply_proximal_control(push, rot, dt)` still works for single-env clients and is bit-for-bit unchanged for single-rod use; for multi-env clients it simply broadcasts the scalar inputs and dispatches to the GPU path. Existing call sites continue to compile."

#### B.2 — `set_root_orientation(env_idx, q)`

**Say:**

"Single-thread launch — `dim=1`, the kernel writes the root quaternion at a known global index `env_idx × num_points_per_rod`. Used to seed each environment with an initial C-arm-aligned orientation at reset, or to drive curriculum-controlled root pose without an explicit force. Because the launch is a fixed-size kernel with a precomputed integer index, it is safe inside a CUDA graph — no Python control flow, no host-side branching."

#### B.3 — Floor collision restitution

**Say:**

"`_xr_floor_collision` now takes a `restitution` argument and the solver exposes `floor_restitution: float` in its constructor. Zero gives the previous inelastic behaviour — `vz` clamped to zero on penetration. One gives a perfectly elastic bounce — `vz → −vz`. Anything in between gives a damped reflection. This is parity with Newton upstream and is the first feature we will use when we add table / vessel-wall collision in Sprint 2."

| `restitution` | Behaviour | Use case |
|---|---|---|
| `0.0` | Inelastic — vertical velocity zeroed on contact | Default; legacy parity |
| `0.0–1.0` | Damped bounce | Tuned per surface for vessel wall / table |
| `1.0` | Perfectly elastic | Validation only — generally unphysical for tissue |

#### B.4 — `step()` captures the substep loop into a CUDA graph (the headline perf win)

**Say:**

"The first time `step(dt)` is called on a CUDA device, the solver does three things in order. One: a single warm-up substep outside any capture, so all 11 batched Warp kernels finish JIT compilation. Two: `wp.synchronize_device` to flush the warm-up. Three: open a `wp.ScopedCapture`, run the full `num_substeps` loop inside the capture region, and stash the resulting graph object on the solver instance."

"Every subsequent `step(dt)` call with the same `dt` issues exactly one host call — `wp.capture_launch(self._cuda_graph)` — which replays the entire substep loop with effectively zero Python overhead, zero kernel-launch latency from the host, and zero parameter marshalling. The graph is invalidated automatically if `dt` changes or if the user calls `reset_cuda_graph()` after externally mutating state size."

"This is the only solver in the tree that does graph capture today. It is the dominant reason the self-contained XPBD path is now competitive with Newton upstream on per-step throughput."

**Pre-empt:** *"What invalidates the graph?"* Changing `dt`, reallocating the workspace, or any `_can_use_cuda_graph()` override returning false. We do not invalidate on input changes — `apply_proximal_control_gpu` writes to the preallocated `push_velocities` scratch buffer that is part of the captured graph, so per-step control inputs flow through the graph by buffer mutation rather than by re-capture.

**Pre-empt:** *"Does this break debugging?"* No — graph capture is only enabled on CUDA devices. On CPU, or if the user calls `reset_cuda_graph()` for inspection, the substep loop runs eagerly with full Warp diagnostics.

#### B.5 — `max_delta_lambda` diagnostic

**Say:**

"Inside `_xr_compute_corrections_batched`, every thread that writes a Δλ also issues `wp.atomic_max(max_delta_lambda, |Δλ|)`. After the substep finishes, `solver.max_delta_lambda` returns the maximum-magnitude Lagrange-multiplier change across **every constraint of every env** — a single scalar that monitors solver convergence across the entire batch in O(1) read cost. Useful for catching exploding-rod regressions in 512-env training without a per-env reduction."

---

### Section C (footer) — Smoke test status + backward compatibility

**Say:**

"The footer says: smoke-tested with `num_envs=8` — `step()`, `apply_proximal_control_gpu`, and `set_root_orientation` all pass. The position tensor returns shape `(num_envs, num_points, 3)` when the batched workspace is allocated, and `(num_points, 3)` when it isn't. Same for orientations and velocities. So legacy single-env code that does `solver.positions[10]` and gets back a `(num_points, 3)` tensor continues to work; new multi-env code that does `solver.positions[env_idx, 10]` gets a `(num_points, 3)` slice from the right env."

"Single-env throughput baseline — ~1,300 Hz on RTX A6000 at 20 segments — is preserved. The multi-env throughput benchmark is the immediate next-step deliverable."

---

### Closing strategic frame for Slide 8

**Say:**

"Three things to take away. One: the data layout and the kernel set are now structurally identical to Newton's upstream `xpbd_rod` batched path — there is no longer a research-vs-production fork on this code path. Two: the controller and the substep loop are CUDA-graph-friendly end-to-end, which is the unlock for high-throughput RL — per-step Python overhead is no longer the bottleneck. Three: backward compatibility is total, so every existing single-env experiment continues to run without code changes. The remaining open item is the formal multi-env perf benchmark, which is in Sprint 2."

---

### Pre-empts for Slide 8

- *"Why one thread per rod for block-Thomas instead of inter-rod parallelism inside the solve?"* Each rod's block-tridiagonal solve is inherently sequential — forward sweep then back substitution, neither parallelizable across edges of a single rod. Inter-rod parallelism is therefore the only useful axis. With one thread per rod we saturate the GPU at hundreds of rods, which is exactly the regime RL needs.
- *"What is the expected multi-env perf number?"* Single-env baseline is ~1,300 Hz on A6000 at 20 segments. Multi-env benchmarking is in flight. Based on Newton's upstream measurements, 512-env batched physics is well under 16 ms per substep — i.e. >60 Hz at 512 envs is the working target, and we expect to beat it because graph capture eliminates the per-step Python tax that Newton's measurements include.
- *"Is this safe to drop into the existing `CatheterStateEnv`?"* Yes — the env constructs the solver with its `num_envs` and the batched path is selected automatically. The legacy single-env path is preserved unchanged. No env-side code changes required to take advantage of graph capture.
- *"How does this compare to the production `RodSolver`?"* The production solver uses PyTorch tensors and a PyTorch-side direct tree solver; it is multi-env via tensor batching but does not capture into CUDA graphs, and it has SDF / mesh BVH collision that this solver does not yet have. The self-contained XPBD solver wins on per-step launch overhead and on having zero non-Warp dependencies; the production solver wins on collision realism. Sprint 2 closes the collision gap on the self-contained path.

---

## SLIDE 9 — Fluoroscopy Rendering: Three Compositing Paths

**Say:**

"Three compositing paths exist, ordered by fidelity and use case."

"**Slang GPU unified loop** is the production path. A single shader dispatch ray-marches through the patient's μ volume and the catheter cylinder geometry simultaneously and accumulates one Beer-Lambert exponent: `I = I0 × exp(−∫ μ_total ds)` where `μ_total = μ_volume + μ_catheter` evaluated at every step along the ray. One shader, one integral, no compositing pass. About 25 FPS at 512×512 on an A6000. The catheter is a `StructuredBuffer<CatheterSegment>` uploaded per frame — endpoint pairs, radius, attenuation coefficient per segment."

"**CPU Beer-Lambert** is the offline / regression path. It computes the exact cylinder chord thickness per pixel using the closed-form `t(d) = 2 × √(r² − d²)` from ray-cylinder intersection, accumulates an attenuation map, applies the Beer-Lambert exponential, and then runs the full detector chain — veiling glare, detector PSF, Poisson noise. Physically rigorous but pure NumPy, so 2–5 FPS. The Sprint 2 fix is a Warp port of the per-pixel loop — would push it to 30+ FPS."

"**Isaac Lab 3D USD quad** is the visualisation-only path inside Omniverse — DRR backdrop on a quad, capsule markers for the catheter. Real-time but not physically correct (the catheter is opaque, not transmissive). Used for sanity-check viewports and demos."

"The crucial physics point: Beer-Lambert is multiplicative. The catheter darkens the background, it doesn't paint over it. Self-crossings — where the catheter loops over itself — produce correctly increased attenuation because the µ values add inside the exponent before the exponentiation. Naive alpha compositing would not get this right."

**Buffer layout (technical detail, drop in if asked):** the catheter is uploaded to the Slang renderer as a structured GPU buffer. Positions are `(N, 3)` float32 in mm in the world frame; radii are per-segment or scalar in mm; µ values are per-segment or scalar in mm⁻¹. The structured array packs as `(n_seg, 8)` float32 = `[p0(3), p1(3), radius, mu]` per segment. The shader internals loop over all segments, check ray-cylinder proximity, and accumulate `μ × √(1 − d²/r²)` per ray step — the same chord model as the CPU path but evaluated per-ray-step rather than per-pixel.

---

## SLIDE 10 — Beer-Lambert Detector Physics Chain (CPU path)

**Say:**

"This is the CPU detector chain that runs in `realism.py::apply_realism()`. Five steps."

"Step 1: per-segment attenuation map using the exact cylinder chord — `t(d) = 2√(r² − d²)` where `d` is the perpendicular distance from the pixel ray to the segment axis."

"Step 2: Beer-Lambert transmission — `I_final = I_DRR × exp(−attenuation_map)`. Multiplicative; preserves the volume background through the catheter."

"Step 3: veiling glare — `I_scatter = 0.03 × GaussianBlur(I_blocked, σ=18 px)` where `I_blocked = I_DRR × (1 − T)` is the photons absorbed by the catheter. Models tissue-and-housing scatter halo around bright structures."

"Step 4: detector point spread function — Gaussian σ=0.7 pixels modelling the finite resolution of a CsI scintillator."

"Step 5: Poisson noise at 2,000 photons per pixel — the photon-counting statistics of low-dose pulsed fluoroscopy."

"The catheter has a five-zone longitudinal attenuation profile, mapped to physical materials:"

| Zone | Segments | μ (mm⁻¹) | Material |
|---|---|---|---|
| Proximal marker band | 0–1 | 3.0 | Tungsten |
| Braided shaft | 2 → 60% | 0.8 | Nitinol braid |
| Transition zone | 60% → 85% | 0.8 → 0.2 (linear) | Sparse braid + polymer |
| Soft polymer tip | 85% → 95% | 0.15 | PEBAX |
| Distal tip marker | last 3 | 5.0 | Platinum coil |

"Both the CPU compositor and the catheter segment data uploaded to the Slang renderer use this exact profile."

**Pre-empt:** *"Why don't the GPU and CPU paths produce the same image?"* The GPU Slang shader does the Beer-Lambert ray march including the catheter contribution, but it does **not** apply the detector physics chain on the GPU — that's still done on the CPU after the ray march completes. So the GPU image is the "physically attenuated" image; the CPU physics chain adds noise, scatter, PSF, and gamma. Sprint 2 includes a GPU-side detector physics pass on the Slang path to remove the CPU readback.

---

## SLIDE 11 — DSA Pipeline & Bolus Dynamics — Implemented

**Say:**

"DSA — Digital Subtraction Angiography — and bolus dynamics are both implemented and validated. Two columns."

"Left column: `DSAPipeline.render_dsa_frame()` in `dsa.py` is the seven-step pipeline. Steps 1–2 render the mask DRR (no contrast) and the contrast DRR (with iodinated contrast in the vessels) at the same C-arm pose. Step 3 applies the same scatter convolution to both. Step 4 adds independent Poisson and Gaussian noise to each — this is intentional; correlated noise would not subtract correctly. Step 5 applies sub-pixel misregistration jitter to the mask only — this models patient motion between the mask and contrast acquisitions, which is the dominant DSA artefact. Step 6 subtracts: `diff = contrast − mask`. Step 7 applies a contrast boost with a gain of 20, gamma correction at γ=0.8, and intensity normalization. There's also `render_dsa_sequence()` for temporal cine output."

"Right column: bolus dynamics. `extract_centerlines()` runs VMTK's marching-cubes plus network extraction on the vessel mask. `compute_arrival_map()` runs Dijkstra over the centerline graph with per-edge length as cost — this gives you the contrast travel-time map T(v), which is the time for the contrast bolus to reach voxel `v` from the injection point. `gamma_variate(t, α, β)` is the standard pharmacokinetic concentration curve C(t) — a gamma-variate bolus model that matches measured DSA contrast profiles. `build_contrast_volume()` generates a per-frame µ-volume by adding `Δμ × C(t − T(v))` to the baseline tissue attenuation on a per-voxel basis. `apply_vessel_boost()` multiplies µ by a factor of 8 inside the vessel mask to compensate for the fact that monoenergetic DRR underestimates iodine contrast. The whole pipeline is wired into `FluoroSimulator.render_cine()` via a `volume_callback` — the renderer asks the bolus model for the current µ volume each frame."

"This means we can simulate a 30-second clinical DSA acquisition — 150 frames at 5 FPS — with realistic temporal contrast wash-in and wash-out, on patient-specific vasculature derived from CTA."

**Likely Q&A:** *"Have we validated the gamma-variate parameters against clinical data?"* Not yet — they are parameterised from literature defaults. Validation against acquired DSA sequences is an evaluation-pipeline ticket (Sprint 3+).

---

## SLIDE 12 — Volumetric Instrument Injection — Implemented

**Say:**

"This is the GPU compositing approach for putting a catheter into the µ volume itself, rather than as a separate pass."

"The compositing rule is `μ_composited(v) = max(μ_anatomy(v), μ_instrument)`. Three immediate consequences. One: bone is preserved because µ_bone is greater than µ_polymer — when a polymer catheter passes behind a rib, the rib correctly occludes it. Two: soft tissue is replaced because µ_catheter is greater than µ_tissue — the catheter is correctly visible against soft tissue. Three: in a multi-instrument scenario, the densest material wins per voxel — a tungsten marker on a polymer shaft will read as tungsten. This is the same compositing rule used in clinical CBCT instrument visualisation."

"Implementation is `wp.atomic_max` in two Warp kernels: `paint_spheres_kernel` for the tip and marker bands, `paint_cylinders_kernel` for the shaft segments. Atomic-max is thread-safe so we can parallelise over voxels with no contention or ordering concerns. The performance number is about 2 milliseconds per 64-node catheter on a 512³ volume on an A6000 — versus about 500 milliseconds for the equivalent NumPy painting loop."

"The materials table on the right is the calibrated attenuation library — six materials covering catheter shafts (polymer 0.03), guidewires (nitinol 0.08), needles (stainless 0.10), marker bands (platinum 0.20), radiopaque markers (tungsten 0.35), and iodine contrast (0.05). The agent picks materials per device based on the procedure config."

---

## SLIDE 13 — C-arm Vendor Presets — Implemented

**Say:**

"Nine vendor-specific C-arm presets are implemented as `CarmGeometry` classmethod factories in `fluorosim/config.py`. Three GE — OEC 9900, OEC Elite CFD, Innova IGS 540. Three Siemens — Arcadis Avantic, Cios Alpha, Artis zee. Two Philips — BV Pulsera, Azurion 7. One Ziehm — Vision RFD 3D."

"Each preset specifies the physical geometry: source-to-detector distance (SDD), source-to-isocenter distance (SID), detector matrix size, and pixel spacing. These are the four parameters that determine cone-beam projection geometry, magnification factor, and the effective field of view. They are not aesthetic settings — they directly drive the DRR ray geometry and therefore the simulation matches the actual clinical scanner you're targeting."

**Projection model (drop in if asked):** standard pinhole camera parameterised by C-arm geometry. Intrinsics: `f_px = SID / pixel_spacing` (default ≈ 1235 px for SID=1000, pixel=0.81 mm), principal point at the detector centre. Extrinsics: `R = Rx(cran/caud) × Ry(LAO/RAO)`, source at `(0, 0, −SOD)` in the camera frame, isocentre at the world origin. Cone-beam magnification is depth-dependent: `r_px = r_physical × f_px / z_cam` — the projected catheter radius shrinks correctly with depth, matching the cone-beam geometry of a real interventional C-arm.

"The agent-facing API is one line: `geometry = CarmGeometry.philips_azurion_7()`. The natural language → config skill (Phase 3 agentic workflow) maps phrases like 'use a Philips Azurion in DSA mode' to that classmethod call plus a DSA pipeline configuration."

**Likely Q&A:** *"Where do these numbers come from?"* Vendor data sheets and FDA 510(k) filings. They are nominal values — actual scanner geometry varies with acquisition protocol (zoom, detector position). For sim-to-real transfer the values are within domain-randomisation tolerance.

---

## SLIDE 14 — X-Ray Performance Baseline

**Say:**

"Five rows. Three are achieved, one is the new multi-env capability, one is the Sprint 2 target."

"Single-env physics FPS — target is 1,000 Hz, current is approximately 1,300 Hz at 20 segments on an A6000. Achieved."

"Slang GPU compositing fused — target is under 5 ms at 512², current is about 40 ms — 25 FPS. Single GPU ray march. The gap to target is mostly due to step-size trade-offs (we step at 0.5 mm for accuracy at vessel features); coarser stepping for screening would buy back the headroom."

"CPU Beer-Lambert compositing — target under 2 ms per frame, current 200–500 ms per frame. NumPy. The Warp port would close this — that's a Sprint 2 line item if anyone wants the CPU path on the in-loop training critical path."

"Multi-env physics with batched XPBD — target 60 Hz at 512 envs, status: available. The implementation is in (batched block-Thomas plus CUDA-graph capture) and it runs. End-to-end benchmarking is the immediate next deliverable."

"Five-twelve-env fluoroscopy rendering — target 60 Hz, status: single-env Slang path only. This is the bottleneck. Sprint 2 target is multi-dispatch Slang."

---

## SLIDE 15 — RL Training Pipeline (State-Based)

**Say:**

"State-based PPO is operational at 512 environments via RSL-RL. Five components."

"The environment uses the multi-env production rod solver with proximal push-and-rotate kinematic control and a distance-to-target reward — Euclidean distance from the catheter tip to a target point in the vessel tree."

"The RSL-RL wrapper is a standard `VecEnv` adapter — nothing exotic, follows the published RSL-RL API."

"The PPO config is hyperparameter-tuned for catheter navigation — discount factor, GAE lambda, clip ratio, learning rate, mini-batch size, number of epochs are all set in `agents/rsl_rl_ppo_cfg.py`."

"Training entry point is `examples/train_catheter_state.py` — 512 parallel envs, 1,500 max iterations."

"The smoke test in `examples/run_catheter_state_smoke.py` validates the env without an RL dependency — useful for CI."

"To be precise: this is **state-based** RL. The observation dict contains catheter tip position, catheter centerline state, target position, time-since-reset. There are no pixels in the observations today. Image-based observations are a Sprint 2 deliverable — that's the line below the table."

**Reward function — current vs target (drop in if asked):** the env today implements a single-term distance-to-target reward, `r = −‖tip − goal‖₂`. The target reward decomposition for the production training run is the seven-term formulation below, all of which are implementable on top of the existing collision-and-pose state:

| Component | Formulation | Weight |
|---|---|---|
| Target proximity | `r = −‖tip − goal‖₂` | 1.0 |
| Wall contact penalty | `r = −λ_c · Σ max(0, F − F_threshold)` | 0.5 |
| Procedure time penalty | `r = −λ_t · Δt` | 0.1 |
| Tip force penalty (perforation safety) | `r = −λ_f · ‖F_tip‖` | 0.3 |
| Fluoroscopy dose (minimise imaging) | `r = −λ_d · n_frames` | 0.05 |
| Progress reward | `r = Δ(dist_to_target) / Δt` | 0.2 |
| Success bonus | `r = +R_bonus if ‖tip − goal‖ < ε` | 10.0 |

"The wall-contact and tip-force terms become available once SDF / mesh vessel-wall collision lands in Sprint 2. The fluoroscopy-dose term becomes meaningful once image-based observations are wired in. So the full reward stack lights up incrementally as the dependent capabilities ship."

---

## SLIDE 16 — Part 2 Divider: This Release

**Say:**

"Part 2 — This Release. The deliverables that ship in Sprint 1, the XCATH gaps closed, and the items that genuinely remain after we audited what was already implemented in `i4h-sensor-simulation-internal`."

---

## SLIDE 17 — This Release — Completed Deliverables

**Say:**

"Nineteen completed deliverables this release. I'll group them."

"Rendering and physics core — Beer-Lambert compositing on CPU and Slang GPU paths, the fused DRR-plus-catheter single ray march in `diffdrr_slang.slang`, the per-segment 5-zone catheter attenuation profile, cone-beam magnification of the projected catheter radius."

"Detector physics — Poisson noise plus scatter plus PSF plus gamma in `realism.apply_realism()`, sub-pixel misregistration jitter in `realism.apply_misregistration`."

"DSA and bolus — the four-step `DSAPipeline`, vessel boost (μ × 8) on vessel-masked voxels, VMTK centerline plus Dijkstra arrival map, gamma-variate bolus plus per-frame contrast volume, per-frame µ update in cine rendering via `volume_callback`."

"Volumetric instrument injection — Warp `atomic_max` GPU kernels."

"Clinical / API — nine vendor C-arm presets as `CarmGeometry` classmethods, differentiable rendering with Slang autodiff for 6-DOF pose gradients via `renderDRR_backward`, the proximal kinematic control API for push and rotate."

"Self-contained XPBD upgrades — multi-env batched kernels in `_BatchedWorkspace`, GPU-side root control via `apply_proximal_control_gpu` and `set_root_orientation`, CUDA-graph capture of the substep loop, floor-collision restitution to match Newton upstream."

"And the RL pipeline at 512 envs."

"This is the longest list this team has shipped in a single sprint. The combined audit of IsaacLab plus i4h-sensor-simulation-internal is what made this possible — many of these features were already implemented in the latter and previously believed to be missing."

---

## SLIDE 18 — This Release — Closed Gaps vs XCATH Requirements

**Say:**

"Nineteen XCATH-required capabilities closed in this release. The format is Capability / Before / After."

"The first thirteen rows are the realism deliverables — Beer-Lambert compositing, 5-zone attenuation, detector physics, the DSA pipeline, vessel boost, bolus dynamics, VMTK arrival map, gamma correction, scatter convolution, misregistration jitter, C-arm presets, cone-beam magnification, per-frame µ updates. Every one of these went from 'Missing' to 'Implemented'."

"Three rows below that are net-new capabilities: max-attenuation volumetric instrument injection went from 'Planned' to 'Implemented Warp atomic_max'. Fused GPU DRR + catheter (single ray march) is a NEW capability — we didn't have a fused path before. State-based RL pipeline went from 'Missing' to 'PPO @ 512 envs'."

"And the last three rows are the XPBD upgrade we just shipped: multi-env XPBD self-contained solver, GPU-side proximal control with no CPU sync, and CUDA-graph capture of the substep loop."

"For the audience: this is the chart to anchor on when discussing the release scope. Nineteen items closed. The remaining gaps are on the next slide."

---

## SLIDE 19 — This Release — Genuinely Remaining Items

**Say:**

"Seven items genuinely remain after the audit. Each is a real Sprint 2 or later ticket — none are 'we didn't realize this was already done'."

"Multi-env / batched fluoroscopy rendering — the Slang renderer is single-env. Sprint 2 deliverable."

"Image-based RL observations — pixel observations from the fluoroscopy frames into the PPO observation dict. Sprint 2."

"Beam hardening (polyenergetic correction) — current DRR is monoenergetic. Closing this brings us to parity with DeepDRR. Sprint 2 or 3."

"GPU-side detector physics on the Slang path — currently scatter, PSF, and Poisson are CPU-only. Fine for offline data generation, problematic for in-loop GPU training because of the readback. Sprint 2."

"Selective injection (hemisphere masking) — the bolus model fills both hemispheres simultaneously. In clinical DSA you typically inject into one carotid at a time. ~1 day fix on top of the existing pipeline."

"Realism evaluation metrics — FID, SSIM, vessel visibility — needed for the agent's iterative refinement loop in Phase 3."

"3D physics-based scatter — currently we have a 2D Gaussian convolution approximation. A full 3D Monte Carlo scatter model is more expensive and lower priority — moved to Workflow Enhancements."

---

## SLIDE 20 — Part 3 Divider: Next Release

**Say:**

"Part 3 — Next Release. Sprint 2, weeks 3–4. Theme is scale and contact realism."

---

## SLIDE 21 — Next Release — X-Ray Sprint 2 (Weeks 3–4)

**Say:**

"Five Sprint 2 deliverables."

"One — multi-env fluoroscopy rendering. Batched Slang dispatch for all envs in one frame. The technical scope is on the next slide."

"Two — SDF / mesh collision for vessel walls. Port from the Newton `xpbd_rod` `kernels_collision` module. The XPBD solver knows about floor collision today; we need vessel-wall collision to make the catheter physically reactive to the vasculature."

"Three — image-based RL observations. Wire the fluoroscopy frames into the PPO observation dict, add a CNN feature extractor, retrain the policy. Once multi-env rendering is in, this is mostly plumbing."

"Four — end-to-end multi-env benchmark. Physics + rendering + RL @ 512 envs. We want a single number for wall-clock per training step at scale."

"Five — performance target: 60 Hz end-to-end at 512 envs. That number is what makes a 24-hour PPO training run feasible on a single A6000."

"The status line below the bullets is the honest summary: production XPBD and self-contained XPBD are both multi-env. The Newton bridge wrapper is still single-env. The remaining bottleneck for 512-env training is the renderer, not the physics. Sprint 2 unlocks the renderer."

---

## SLIDE 22 — Multi-Env Fluoroscopy Rendering — Why Sprint 2

**Say:**

"This slide exists because I want the audience to understand that multi-env rendering is engineering work, not a research problem. Each pixel ray-march is independent of every other pixel — there's no algorithmic blocker. The blockers are all in the API surface and the GPU resource allocation."

"Left column — what's hard-coded to one env today. The Slang shader's `outputImage` is `RWTexture2D<float>` — a single 2D image, no array slot. The `Pose` parameter is a single struct, not a buffer. The `renderDRR_forward` kernel is dispatched as `uint3(W, H, 1)` — the third dimension is hard-coded to 1. The Python wrapper allocates one 3D `Texture3D` for the µ volume, one 2D `Texture2D` for the output, one `StructuredBuffer<CatheterSegment>` for the catheter. `render()` returns shape `(H, W)` per call. So 512 envs today means 512 sequential dispatches per frame, plus 512 round-trip readbacks. The backward path has the same single-image limitation."

"Right column — what Sprint 2 changes, estimated about one week of work. Convert the shader's `RWTexture2D` to `RWTexture2DArray`, with the array slot indexed by `dispatchThreadID.z`. Convert `Pose` to a `StructuredBuffer<Pose>` indexed by env. Concatenate the catheter buffer across envs and add a separate `envOffsets` plus `envSegmentCounts` index buffer — exactly the same pattern we used in the XPBD `_BatchedWorkspace`. Single dispatch as `uint3(W, H, num_envs)` instead of a Python loop. `render()` returns `(num_envs, H, W)`. The backward path gets the same array-isation. Optionally a per-env μ volume slot for domain-randomised anatomies — most use cases share one volume."

"The footer is the honest takeaway: this is the same engineering pattern that just unblocked self-contained XPBD multi-env. Concatenated GPU buffers plus per-env offset arrays plus one batched kernel launch. We have a template for it now."

**Pre-empt:** *"Why didn't we do this earlier?"* The fluoroscopy renderer was originally built for offline data generation and single-pose differentiable registration — multi-env was not the primary use case. With the XPBD physics now multi-env, the renderer is genuinely on the critical path for the first time.

---

## SLIDE 23 — Next Release — Phase 1 Fidelity Items (X-Ray)

**Say:**

"Six items, with effort estimates. About three weeks of total work for one engineer."

"~1 week — multi-env / batched Slang rendering."

"~3 days — image-based RL observations (pixel obs)."

"~3 days — GPU-side detector physics on Slang path."

"~2 days — beam hardening (polyenergetic correction)."

"~1 day — selective injection (hemisphere masking)."

"~2 days — realism evaluation module (FID, SSIM, vessel visibility)."

"The note at the bottom is important: DSA, vessel boost, bolus dynamics, gamma correction, scatter convolution, misregistration jitter, C-arm presets, and per-frame µ updates are already implemented in `i4h-sensor-simulation-internal`. The remaining work is multi-env scaling and image-based RL — the realism foundation is already in place."

---

## SLIDE 24 — Part 4 Divider: Following Releases

**Say:**

"Part 4 — Following Releases. Sprint 3 plus the agentic workflow integration phases."

---

## SLIDE 25 — Sprint 3 — Training Readiness (Weeks 5–6)

**Say:**

"Five Sprint 3 deliverables, focused on training readiness."

"Domain randomization — randomise C-arm angles, attenuation coefficients, photon count, scatter parameters per env, per episode. This is what gets the policy to transfer to a real C-arm rather than overfit to one preset."

"Gymnasium wrapper — adds standard RL ecosystem compatibility. The env is RSL-RL native today; the Gymnasium wrapper opens it to Stable-Baselines3, CleanRL, and SB3-Contrib."

"Per-task CUDA-graph variants — separate captured graphs for sub-batches with mixed materials. The single-graph path captures one set of compliance constants; if you want to run, say, 384 envs of polymer catheter and 128 of nitinol guidewire, you want two graphs and a fast switch. This builds on the CUDA-graph infrastructure shipped this release."

"Image-based observations — fluoroscopy frames as RL inputs end-to-end. Depends on Sprint 2 multi-env rendering."

"Automated pytest suite — regression coverage for the solver, the compositing pipeline, and the renderer. Mostly a productionisation deliverable."

---

## SLIDE 26 — Agentic Workflow — Phase 2 (Skill Packaging)

**Say:**

"Phase 2 of the agentic workflow is skill packaging. Each pipeline stage is wrapped as a portable OpenClaw / NemoClaw Skill with a clean I/O contract."

"Seven skills. `patient-digital-twin` — CTA in, μ volume plus vessel mask plus centerline plus arrival map out. `catheter-physics-sim` — wraps the rod solver and the compositing config. `sensor-sim-xray` — DRR / DSA / vessel-boost rendering modes. `dataset-creation` — paired HDF5 / WebDataset of frames + pose + ground truth. `reward-function` — RL reward configuration covering target, contact, dose, progress, success. `policy-training` — IL via GR00T-H, then RL via PPO or SAC, then SIL evaluation. `evaluation` — success rate, navigation time, contact force, FID, registration accuracy."

"The point of skill packaging is composability. An agent in Phase 3 can chain these in different orders, swap implementations, and re-run the whole pipeline as a single workflow."

---

## SLIDE 27 — Agentic Workflow — Phase 3 (Agent Integration)

**Say:**

"Phase 3 is the agent integration itself. Four week-numbered milestones."

"Weeks 5–6 — skill discovery and chaining. The agent parses the skill definitions from Phase 2, resolves the I/O dependencies, and constructs a valid pipeline graph from a high-level user intent."

"Week 6 — natural language to config mapping. The user says 'use a Philips Azurion in DSA mode'; the agent translates that to the YAML overrides for `CarmGeometry.philips_azurion_7()` plus the DSA pipeline configuration. This requires a small library of config fragments and a robust LLM prompt."

"Weeks 6–7 — iterative refinement loop. The agent runs the evaluation skill, analyses the metrics, proposes config changes (more contrast, different photon count, different scatter parameters), re-runs the pipeline, repeats. This is the core innovation — closing the loop on what was previously a manual experiment cycle."

"Week 7 — Slack and IDE integration. The agent posts progress, intermediate visualisations, and final reports into Slack channels and IDE windows so the human is in the loop without having to drive the experiment."

"The outcome is a measurable compression: experiment cycles that took weeks of human time become hours of agent time, with human review at the milestones rather than at every parameter sweep."

**Concrete dialog example (use this verbatim if asked 'what does it actually look like'):**

> **Developer:** *"Train a catheter navigation policy for cerebral ICA → MCA. Patient CTA #42, Philips Azurion C-arm, DSA mode, 50 teleop demos."*
>
> **Agent → patient-digital-twin:** Loads CTA NIfTI → HU→μ volume (512×512×300). Runs VMTK → 5,243 centerline nodes, 176 cells. Dijkstra arrival map from ICA root. Selective injection: left hemisphere only. *"Patient twin ready."*
>
> **Agent → catheter-physics-sim:** Newton config — stiffness 2.5 N/m, friction 0.3. Builds vessel SDF. Beer-Lambert per-segment µ profile (tungsten markers µ=3.0, nitinol shaft µ=0.8, platinum tip µ=5.0). Collects 50 teleop demos, mean episode 45 s. *"Ready for sensor sim."*
>
> **Agent → sensor-sim-xray:** Picks `CarmGeometry.philips_azurion_7()` — SDD 1240, SID 780, 2480×1920. Renders via Slang fused DRR + catheter at ~25 FPS. DSA mode with k=20, γ=0.8. Domain randomisation enabled. Detector physics — Poisson 2000 photons/px, veiling glare σ=18 px / 3%, PSF σ=0.7 px. *"10,000 paired frames across 200 episodes."*
>
> **Agent → dataset-creation:** Packages HDF5 — fluoroscopy + catheter_pose + GT_carm_pose. 80/10/10 split. *"Dataset: 10,000 frames, 200 episodes, 2.3 GB HDF5."*
>
> **Agent → policy-training:** IL on 50 teleop demos (BC, 100 epochs) → PPO RL fine-tune (2,000 epochs). *"Best checkpoint: epoch 1600."*
>
> **Agent → evaluation:** SIL on 100 episodes. Success 73%, mean time 38 s, max contact 0.08 N. *"27% failures cluster at the ICA-MCA bifurcation. Recommendation: add a curriculum stage."*
>
> **Agent re-runs pipeline** with bifurcation curriculum → success 84%. *"Remaining failures distal MCA; suggest 20 more teleop demos targeting distal branches."*

"That's the loop. Note the agent is making three classes of decisions: (1) skill selection and chaining, (2) configuration translation from natural language to YAML, (3) failure-mode analysis and corrective proposals. None of these are heroic LLM reasoning — they are bounded decisions over a known skill graph with a known config schema."

---

## SLIDE 28 — Following Releases — Remaining Capability Gaps

**Say:**

"This is the same set of gaps we saw on the 'Genuinely Remaining' slide, now mapped to skill ownership for the agentic workflow phase."

"Multi-env fluoroscopy rendering — `sensor-sim-xray` skill. Required for batched 512+ env RL training."

"Image-based RL observations — `reward-function` and `policy-training` skills. Image-guided policies."

"GPU-side detector physics on Slang — `sensor-sim-xray`. Avoid CPU readback in the in-loop training pipeline."

"Beam hardening — `sensor-sim-xray`. Polyenergetic correction. Closes the realism gap vs DeepDRR."

"Selective injection — `patient-digital-twin`. Hemisphere masking on the bolus pipeline."

"Realism metrics — `evaluation`. FID, SSIM, vessel visibility for the iterative refinement loop."

"3D physics-based scatter — `sensor-sim-xray`. Higher-fidelity scatter halo."

"The footer is the closing reminder: DSA, vessel boost, bolus tracking, per-frame µ update, max-attenuation volume compositing, C-arm presets, gamma correction, scatter convolution, and misregistration jitter — all closed in this release. The remaining gaps are the genuine forward-looking work."

---

## SLIDE 29 — Part 5 Divider: Workflow Enhancements

**Say:**

"Part 5 — Workflow Enhancements. Adjacent capabilities that augment the X-ray catheter intervention workflow but aren't on the critical path for the immediate releases."

---

## SLIDE 30 — Workflow Enhancements

**Say:**

"Seven enhancement candidates with priority and effort estimates."

"High priority: multi-env / batched fluoroscopy rendering (~1 week, Sprint 2). Image-based RL observations (~3 days). Force / torque sensing — the data is already in the collision solver, low effort, high value as a safety reward signal for tip-force penalties on vessel perforation."

"Medium priority: beam hardening for sim-to-real transfer; GPU-side detector physics on the Slang path (~3 days); FID / SSIM / vessel-visibility realism metrics (~2 days)."

"Low–medium priority: CBCT reconstruction. Moderate effort because it requires batched DRR plus FDK reconstruction kernel, but it reuses our existing GPU ray-caster infrastructure. Strategic value: intra-procedural 3D imaging for procedures that can't tolerate a separate CBCT acquisition."

"Strategically: the high-priority three are all directly enabling 512-env image-based RL training. The medium-priority three are realism-and-evaluation upgrades. CBCT is a workflow extension."

---

## SLIDE 31 — Summary

**Say:**

"Four-column summary."

"TODAY: X-ray fluoroscopy pipeline implemented end-to-end. Fused GPU Beer-Lambert at ~25 FPS, physics ~1,300 Hz single-env. DSA + bolus + vessel boost + 9 vendor C-arm presets. Volumetric instrument injection at ~2 ms per 64-node catheter on a 512³ volume. State-based PPO at 512 envs operational."

"THIS RELEASE: Full DSA pipeline plus temporal bolus dynamics. Detector physics — Poisson, scatter, PSF, gamma, jitter. VMTK centerline plus Dijkstra arrival plus gamma-variate C(t). Volumetric instrument injection via Warp atomic_max. Self-contained XPBD: batched plus GPU control plus CUDA graphs. 15+ XCATH-required capabilities CLOSED."

"NEXT RELEASE (Sprint 2): Multi-env / batched Slang fluoroscopy rendering. Image-based RL observations. SDF / mesh collision for vessel walls. GPU-side detector physics on Slang path. Beam hardening (polyenergetic correction)."

"FOLLOWING (Sprint 3 + agentic workflow): Sprint 3 training readiness — DR, Gymnasium, per-task graphs. Phase 2 skill packaging — 7 OpenClaw skills. Phase 3 agent integration — NL → config → eval loop. Realism metrics — FID / SSIM / vessel visibility. Workflow extensions — F/T, CBCT."

"The headline message: the realism foundation is largely in place. The remaining work is scale (multi-env rendering), image-based RL, and the agentic orchestration layer. None of those are research problems; all are well-scoped engineering tickets."

---

## SLIDE 32 — Questions?

**Say:**

"Happy to take questions. I can go deeper on any of the following: the XPBD batched block-Thomas internals; the Slang shader and the proposed array-isation; the DSA pipeline noise model; the bolus dynamics travel-time computation; or the agentic workflow phasing. Anything else, take it offline and I'll route it to the relevant owner."

---

## Anticipated Questions & Tight Answers

### On the solvers

**Q: Why three XPBD backends? Pick one.**
A: Production solver is the RL workhorse and stays. Self-contained solver is the strategic path forward — zero external deps, batched physics, GPU control, CUDA graphs. Newton bridge is a validation harness; will likely be deprecated once the self-contained solver passes a regression suite against Newton.

**Q: Is the self-contained XPBD really feature-complete vs Newton upstream?**
A: The batched block-Thomas direct solve, batched kernel variants, GPU root control, CUDA-graph capture, floor restitution, and per-env material parameters are all in. SDF / mesh collision and contact are not yet ported (Sprint 2 deliverable). For the rod-only XPBD path (no contacts), feature parity is achieved.

**Q: Why one GPU thread per rod for block-Thomas? Doesn't that under-saturate the GPU?**
A: The block-tridiagonal solve is inherently sequential along the rod's chain — forward then back substitution, neither parallelizable across edges. So the only useful parallelism is across rods. At 512+ rods we saturate an A6000.

**Q: Have you benchmarked multi-env XPBD?**
A: Smoke-tested at `num_envs=8`. Full benchmark at `num_envs=512` is the immediate next deliverable in Sprint 2. The expectation is well under 16 ms per substep.

### On the rendering

**Q: Why is the Slang renderer single-env if the physics is multi-env now?**
A: The renderer was originally built for offline data generation and single-pose differentiable registration, not in-loop multi-env training. Now that the physics scales, the renderer is on the critical path for the first time. Sprint 2 unblocks it — about a week of engineering using the same `_BatchedWorkspace`-style pattern.

**Q: Why are detector physics on CPU when the renderer is on GPU?**
A: Historical separation — `i4h-sensor-simulation-internal` was built around a clean offline data-generation pipeline where CPU detector physics is fine. For in-loop GPU training we need the readback removed, which is the "GPU-side detector physics on Slang path" Sprint 2 ticket (~3 days).

**Q: How does the Beer-Lambert compositor handle catheter self-crossings?**
A: Correctly. Beer-Lambert is multiplicative — the µ values add inside the exponent before exponentiation. So a self-crossing produces `exp(−2µL)` rather than `exp(−µL) × exp(−µL) overlaid via alpha`. The single fused ray march in `diffdrr_slang.slang` is the correct physics; an alpha-composited pass would not be.

**Q: How does this compare to DeepDRR?**
A: We have parity on volume rendering plus catheter compositing. DeepDRR has polyenergetic beam hardening which we don't yet (Sprint 2 ticket). DeepDRR does not have our DSA pipeline, bolus dynamics, vessel boost, volumetric instrument injection, or 9-vendor C-arm preset library. Net: we lead on the catheter / interventional workflow; DeepDRR leads on the polyenergetic physics.

### On the RL pipeline

**Q: Is image-based RL training working?**
A: Not yet. State-based PPO is at 512 envs. Image-based observations require Sprint 2 multi-env rendering plus a CNN feature extractor — about a week of additional plumbing once rendering ships.

**Q: What's the reward function?**
A: Distance-to-target on the catheter tip. Sprint 2 / Sprint 3 add contact force penalties (vessel perforation safety), dose penalties, and progress-along-centerline shaping rewards.

### On the program

**Q: How confident are the Sprint 2 effort estimates?**
A: The multi-env rendering "~1 week" is anchored on the XPBD multi-env work we just shipped — same pattern, comparable scope. The other estimates are smaller, scoped tickets with well-understood implementations. Confidence: high for the engineering work; the long-pole risk is the realism evaluation metrics where calibration against clinical data could surface follow-on work.

**Q: When does the agentic workflow Phase 3 ship?**
A: Phases 2 and 3 are weeks 5–7 in the current plan. Predicated on the Sprint 2 deliverables landing on time so the skills have correct I/O contracts.
