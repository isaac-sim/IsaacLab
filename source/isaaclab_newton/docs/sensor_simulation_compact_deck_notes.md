# Presentation Notes — Sensor Simulation Compact Release Deck
**X-Ray–Guided Robotic Catheter Intervention · NVIDIA Healthcare / Holoscan Team**
*Notes for `sensor_simulation_release_deck_compact.pptx` (17 slides)*

---

## Slide 1 — Title: X-Ray–Guided Robotic Catheter Intervention

**Talking Points**

This deck covers the full simulation workflow for X-ray–guided robotic catheter intervention, developed as part of the Isaac for Healthcare (i4h) initiative. The workflow spans four tightly coupled subsystems: (1) catheter physics simulation using extended position-based dynamics (XPBD) Cosserat rod models running on NVIDIA GPUs; (2) X-ray fluoroscopy rendering via a Slang-based differentiable DRR (digitally reconstructed radiograph) pipeline; (3) a DSA (digital subtraction angiography) post-processing chain with clinical-grade detector physics; and (4) reinforcement learning training via RSL-RL PPO across 512 parallel environments.

The goal of this release is to demonstrate that all four subsystems are integrated, validated end-to-end, and that we have closed the majority of capability gaps identified by our clinical partner XCATH Robotics. Key new additions in this sprint include vessel mesh collision via the ported `XCathRodSolver`, multi-environment Slang rendering with batched GPU dispatch, and a fully validated catheter insertion workflow generating X-ray frames with correct Beer-Lambert attenuation polarity.

**Key Message for Audience**
> We are not building independent tools. We are building a full closed-loop simulation stack: physics → X-ray render → RL training, all running on GPU, all validated against clinical partner requirements.

---

## Slide 2 — Architecture Overview

**Talking Points**

The system is organized as a linear pipeline with four stages, each consuming the output of the previous.

**Stage 1: CT Ingestion**
Starting from a DICOM or NIfTI CT volume, we convert Hounsfield Units (HU) to linear attenuation coefficients (μ, in cm⁻¹) using a piecewise linear mapping calibrated against tissue μ values at 70 keV. The volume is stored as a 512³ float32 GPU tensor. Optionally, vessel segmentation masks are applied to produce a vessel-only μ volume. VMTK (Vascular Modeling Toolkit) extracts the centerline graph, and Dijkstra's algorithm computes a per-voxel arrival time map for bolus simulation. Signed-distance field (SDF) and mesh representations of the vessel lumen are generated and exported as USD assets for downstream use.

**Stage 2: Physics Simulation**
Catheter mechanics are modeled as a Cosserat rod: a 1D deformable body with position, orientation (quaternion), linear velocity, and angular velocity per segment. Three solver backends are available:
- **Production solver** (`rod_solver.py`): PyTorch-based, uses Newton's block-tridiagonal JMJT solver. Full mesh BVH collision support.
- **Self-contained XPBD** (`xpbd_rod_solver.py`): Pure Warp implementation. Batched block-Thomas O(n) direct solve. Runs 1 GPU thread per rod with zero external dependency. CUDA-graph captured.
- **Newton bridge** (`newton_xpbd_rod_wrapper.py`): Thin wrapper delegating to Newton's `_BatchedRodWorkspace`. Multi-environment now unlocked.

All three are multi-env capable. The self-contained solver runs at ~1,300 Hz (20 segments, NVIDIA A6000) single-env.

**Stage 3: X-Ray Rendering**
The Slang-based DRR renderer (`diffdrr_slang.slang`) performs a fused volume + catheter ray march in a single GPU kernel dispatch. For each pixel, a ray is cast from the X-ray source through the CT volume, accumulating Beer-Lambert attenuation (I = I₀ · exp(−∫μ ds)), then the catheter's 5-zone attenuation profile is composited on top using depth-correct blending. The batched path (`renderDRR_forward_batched`) dispatches `envIdx` as `dispatchThreadID.z`, enabling N environments to render simultaneously with separate pose and catheter buffers per env. DSA post-processing applies a 4-step pipeline: contrast DRR → mask DRR with misregistration jitter → subtraction → Poisson noise + scatter + gamma correction.

**Stage 4: RL Training**
The RL environment wraps the production XPBD solver in an RSL-RL VecEnv adapter. State observations include rod particle positions, tip orientation, and distance-to-target. The PPO policy is trained across 512 parallel environments at 1,500 maximum iterations. CUDA-graph capture eliminates per-substep CPU overhead.

**Sprint 2 connection**: fluoroscopy frames from Stage 3 will be wired into Stage 4 as pixel observations, closing the loop for image-based policy learning.

---

## Slide 3 — X-Ray Fluoroscopy Pipeline — At a Glance

**Talking Points**

This table provides a canonical status snapshot of every major system component. Walk the audience through the Multi-Env column carefully — this is the dimension that controls whether we can achieve the 512-env training target.

**Row by row**:

- **Catheter physics solver**: All three XPBD backends are now multi-env. The self-contained Warp solver uses `_BatchedWorkspace`, a concatenated GPU buffer layout with index arrays (`rod_offsets`, `edge_offsets`, `particle_rod_id`, `edge_rod_id`) that allow a single kernel launch to process all rods in a batch without per-env CPU dispatch.

- **DRR volume rendering**: The Slang `renderDRR_forward_batched` kernel dispatches a 3D threadgroup where the Z dimension is `envIdx`. Each env reads its pose from `StructuredBuffer<Pose> poses[envIdx]` and its catheter from a global flat buffer sliced by `StructuredBuffer<int> offsets/counts`.

- **Catheter Beer-Lambert compositing**: Both paths share the same physical model. The GPU path (Slang) is the primary path for training. The CPU path (NumPy) is available for offline analysis and validation.

- **Volumetric instrument injection**: Uses Warp's `wp.atomic_max` to write catheter segment attenuation into the CT volume buffer. This enables the catheter to be rendered as part of the CT volume itself (rather than composited on top), which is more physically accurate for metallic implants that block X-rays completely. ~2 ms per 64 node segments on a 512³ volume.

- **DSA pipeline**: The 4-step pipeline is: (1) render DRR with contrast agent (iodine μ boost on vessel voxels), (2) render mask DRR with vessels replaced by tissue μ and misregistration jitter added, (3) subtract mask from contrast to isolate vessel signal, (4) amplify × 20, apply gamma (γ=0.8), add Poisson noise, convolve scatter kernel. This exactly mirrors the clinical DSA acquisition protocol.

- **Bolus dynamics**: VMTK extracts the centerline graph (nodes, edges, radii). Dijkstra computes arrival time T(v) at each graph node. Per-voxel T is computed via KDTree nearest-node lookup. The gamma-variate function C(t) = A·(t−T)^α·exp(−(t−T)/β) at each voxel modulates μ over time, driving a per-frame volume update passed via a `volume_callback` to `render_cine`.

- **Vessel mesh collision**: `XCathRodSolver` (new in this sprint). The SDF path uses `wp.mesh_query_point_sign_normal` which queries Warp's internal BVH (bounding volume hierarchy) for the signed distance from a particle to the nearest triangle surface. The AABB/edge path adds broadphase AABB filtering and explicit vertex-vs-triangle + rod-segment-vs-mesh-edge contact resolution.

- **Multi-env Slang rendering**: Currently limited to N≤4 without cache degradation due to the `RWStructuredBuffer<float>` output format. For N>8, a `Texture2DArray` would allow the GPU texture cache to coalesce reads across envs. This is the primary Sprint 2 item for the rendering path.

- **Image-based RL**: Currently blocked on connecting the Slang output tensor to the RL observation dict. The multi-env renderer is already producing N frames per call; the wire-up to RSL-RL obs space is the remaining work.

---

## Slide 4 — XCATH Robotics — Partner Requirements

**Talking Points**

XCATH Robotics is our primary clinical partner. This slide establishes *why* we are building this simulator — specifically to solve their paired data bottleneck — and confirms that our simulator has already satisfied their initial set of requirements.

**The Clinical Problem**
XCATH is building a real-time neurointerventional monitoring system: an AI pipeline that overlays a 3D vessel model onto live fluoroscopy during catheter procedures. This requires two AI components to work in real time at the point of care: (1) vessel segmentation on intra-operative fluoroscopy, and (2) 2D/3D image registration to align the pre-operative CTA coordinate frame with the intra-operative X-ray frame.

Training either model requires paired data: for every training example, you need a CTA image, a fluoroscopy image taken at the same time, and the ground-truth 3D-to-2D projection matrix. This data essentially does not exist in practice: CTA and fluoroscopy are acquired at different sessions, with different patient positioning, on different C-arm geometries. You cannot buy or collect it at scale.

**Our Solution**
The simulator generates unlimited synthetic paired data on demand. Given a CTA volume, vessel mask, and a sampled C-arm pose, the simulator outputs a synthetic fluoroscopy or DSA image plus the exact camera matrix. This is the fundamental value proposition: replace scarce, label-free real data with unlimited, perfectly labeled synthetic data.

**What They Built (March 2026)**
XCATH completed 5 simulation methods in our environment. The key validated methods are: base DRR (fluoroscopy from CT), vessel boost (amplifying vessel visibility), the 4-step DSA pipeline, and bolus centerline simulation (contrast propagation via VMTK + Dijkstra + gamma-variate model). These are now canonical parts of our simulator.

**Next Steps**
They need: (1) a large-scale paired dataset generator (batch mode, many poses/patients), (2) a trained 2D/3D registration model, and (3) realism metrics. They have 50 internal patient cases with both CTA and clinical DSA — these will be used for registration validation only (not shared externally).

**Key Message for Audience**
> XCATH has validated the core simulator. The question is no longer "does the simulator produce realistic images" — SSIM=0.964 vs DeepDRR answers that. The question is: does it generate enough variety of paired data, fast enough, for registration training?

---

## Slide 5 — XCATH — April 2026 Progress & Validated Metrics

**Talking Points**

This slide shows the quantitative progress XCATH made between March and April 2026. It is important to present this as evidence of *simulator fidelity* — every metric here validates a specific physical model.

**Blood Flow Model Upgrade**
The March 2026 allometric scaling law (v ∝ r^0.5) was based on Murray's law, which predicts how vessel radius scales with volumetric flow. However, it produced distal velocities that were too fast: distal MCA velocity was 71% of ICA velocity, compared to a physiological target of ~25%. The April upgrade adopts Hagen-Poiseuille (v ∝ r²), which gives the correct parabolic flow velocity for laminar flow in cylindrical vessels. This reduced the bolus delay mismatch from 2.6× to 89% match with real DSA timing (1.90 s simulated vs 2.13 s real).

**Selective Injection via Feeding Trunk Excision (F-8)**
The March simulator injected contrast from 4 roots simultaneously — the entire cerebral vasculature enhanced at once. Real DSA is always selective: the clinician injects from a single catheter tip position, so only the ipsilateral hemisphere fills first. The F-8 algorithm solves this by: (1) running multi-source Dijkstra to partition the vessel graph into Voronoi territories, (2) identifying the feeding arteries for non-selected roots (ICA/VA proximal segments), (3) excising those trunks from the graph to model the antegrade pressure barrier, then (4) running standard Dijkstra from the selected root. This reduced territory coverage from 100% (4-root) to 43.1% (single root), matching clinical selective injection.

**Dispersion Correction**
The gamma-variate model C(t) with fixed β produces identical bolus shapes everywhere in the vessel tree, which is unrealistic — in reality, distal vessels show broader, more dispersed contrast curves due to mixing and residence time effects. The correction uses: β_eff = β₀ + k·T(voxel) + α_res/r(voxel). With k=0.15 and α_res=0.1, this produces a 2.55× asymmetry between proximal FWHM (~1.1 s) and distal FWHM (~2.8 s), matching the physiological range.

**FluoroSim vs DeepDRR (SSIM=0.964)**
This is the key realism validation result. DeepDRR is a polyenergetic X-ray renderer (90 kVp spectrum, material decomposition, Compton scatter energy dependence). FluoroSim uses monoenergetic rendering at 70 keV. The critical question for DSA was whether monoenergetic rendering produces structurally equivalent images. Static DRR comparison: SSIM=0.964, MAE=9.3%, Pearson=0.981. The 9.3% difference is localized to bone edges (where beam hardening occurs in real polyenergetic X-rays). For DSA specifically, this difference cancels out: beam hardening affects mask and contrast DRRs equally, so it vanishes in the subtraction step. Conclusion: monoenergetic rendering is practically sufficient for cerebral DSA.

**Remaining Gaps**
The four gaps XCATH identified are genuine physical limitations of monoenergetic rendering that cannot be closed by parameter tuning:
1. **Beam hardening noise mismatch**: noise CV sim 0.47 vs real 1.46 (3.1×). This arises because real X-ray beams preferentially lose low-energy photons as they pass through tissue, changing the effective energy of transmitted photons. Monoenergetic rendering treats μ as energy-independent.
2. **Iodine K-edge**: at 33.2 keV, iodine's μ jumps by 2.5×. A 70 keV monoenergetic beam misses this discontinuity, underestimating vessel contrast relative to a clinical spectrum.
3. **Focal-spot blur**: clinical C-arms use extended X-ray sources (typically 0.3–1.2 mm focal spot), producing penumbra-softened vessel edges. Our point-source model produces sharper vessel boundaries than real clinical images. Multi-source integration is the mitigation.
4. **Finer distal branches**: annotation density in clinical CTA limits how finely the vessel mask captures small cortical branches.

---

## Slide 6 — PART 1: Current Status (Section Divider)

**Talking Points**

Brief transition. The next two slides present measured performance numbers. Everything on these slides is from actual runs — not projections.

---

## Slide 7 — X-Ray Performance Baseline

**Talking Points**

This table presents the measured performance of each subsystem against target latency/throughput goals.

**Single-env physics FPS (~1,300 Hz)**
The self-contained XPBD solver (`xpbd_rod_solver.py`) runs at approximately 1,300 Hz for a 20-segment rod on an NVIDIA A6000. This is the single-environment substep throughput. The solver uses a block-Thomas tridiagonal direct solve: for a rod with N particles, the system decomposes into N−1 2×2 block equations that can be solved with forward/backward substitution in O(N) time. One GPU thread handles the entire solve for a single rod. At 512 envs, the batched path processes all rods simultaneously with a single Warp kernel launch.

**Slang GPU compositing (~40 ms, ~25 FPS)**
The Slang fused DRR render (`renderDRR_forward_batched`) currently achieves ~40 ms per frame for a 512×512 image. This includes: (1) full volume ray march through the 512³ CT volume (Beer-Lambert integration), (2) catheter segment intersection test for each ray, (3) depth-correct catheter attenuation compositing. The 40 ms figure is for a single environment (N=1). Performance degrades gracefully to N≤4 without significant overhead, because the `StructuredBuffer` buffers are small. At N>8, cache thrashing on the `RWStructuredBuffer<float>` output becomes the bottleneck.

**CPU Beer-Lambert compositing (~200–500 ms)**
The CPU/NumPy Beer-Lambert path is used for offline validation and reference rendering. It is not in the training loop. NumPy ray-marching through a 512³ volume is inherently slow because it cannot exploit GPU parallelism. The plan is to port this to a Warp kernel, which should bring it to >30 FPS by running one GPU thread per output pixel.

**Multi-env physics (batched XPBD)**
The batched XPBD path is available and smoke-tested at `num_envs=8`. `apply_proximal_control_gpu` and `set_root_orientation` are confirmed CUDA-graph capturable (no CPU synchronization points). The CUDA graph is captured on the first call to `step()` and replayed on all subsequent calls, eliminating per-substep Python/CUDA launch overhead. At 512 envs, the CPU overhead is effectively zero.

**Multi-env Slang rendering (◐ partial)**
The RWStructuredBuffer-based batched rendering path is live but has a known cache efficiency limitation at N>8. The Texture2DArray upgrade (Sprint 2) changes the output from a flat 1D buffer indexed by `envIdx * H * W + y * W + x` to a true 3D texture where the GPU L1/L2 cache can coalesce reads across neighboring pixels of the same env.

---

## Slide 8 — RL Training Pipeline (State-Based)

**Talking Points**

This slide describes the current state of the RL training integration. The key word is *state-based* — the policy observes direct physics state variables, not camera images.

**Environment**
The RL environment wraps the production XPBD rod solver. Each of the 512 parallel environments has its own rod instance (batched `_BatchedWorkspace`). At each step, the solver advances the rod physics by `substeps × dt` seconds. The environment applies a proximal control action — push velocity (m/s) and rotate velocity (rad/s) — via `apply_proximal_control_gpu`, which sets the root constraint directly on the GPU without round-tripping to CPU.

**Observations**
Current state observation includes: Cartesian positions of all rod particles (N×3 tensor), root orientation quaternion, tip position, and scalar distance-to-target. This observation vector is constructed entirely on GPU and passed to RSL-RL.

**RSL-RL PPO**
RSL-RL is NVIDIA's reinforcement learning library optimized for GPU-based physics simulation. The VecEnv adapter wraps the IsaacLab environment into the RSL-RL interface. The PPO configuration uses: learning rate 1e-3, clip range 0.2, entropy coefficient 0.01, 5 update epochs per rollout, 24-step rollout horizon. Training runs for 1,500 maximum iterations.

**Smoke test**
A smoke test validates the environment without requiring RL dependencies — it steps the physics, applies random actions, and verifies observation/reward shapes. This is critical for CI integration.

**Sprint 2 transition**
The pixel observation path requires: (1) running `renderDRR_forward_batched` after each physics step to generate N fluoroscopy frames, (2) normalizing and reshaping the output tensor to (N, 1, H, W), (3) encoding via CNN or ViT encoder, (4) concatenating with state obs before passing to the policy. The rendering and physics steps are both GPU-resident, so this should add only ~40 ms render latency per step at N=1.

---

## Slide 9 — PART 2: This Release (Section Divider)

**Talking Points**

This sprint is the completion of Sprint 1 — which focused on getting the full pipeline from CT ingestion to RL training running end-to-end on GPU. The next three slides document what was actually built and where it lives.

---

## Slide 10 — This Release — Stage 1 & 2 (Overview)

**Talking Points**

This slide is a high-level bridge between the section divider and the detailed deliverables table. It confirms the scope of Sprint 1 covered both Stage 1 (CT ingestion + rendering) and Stage 2 (physics solver multi-env + RL pipeline). Use this as a transition — the detailed breakdowns follow on the next two slides.

---

## Slide 11 — This Release — Completed Deliverables

**Talking Points**

This is the canonical reference slide for Sprint 1 deliverables. Walk through the key groups:

**Fluoroscopy rendering (rows 1–4)**
- **Beer-Lambert compositing**: Implements I_out = I₀ · exp(−∫μ(s)·ds) along each ray. The GPU path is in `diffdrr_slang.slang`; the CPU path uses NumPy integration. Both paths produce identical results (validated with synthetic phantoms).
- **Slang fused DRR + catheter**: A single ray-march handles both the CT volume attenuation and the catheter overlay. This avoids the cost of two separate render passes and ensures depth-correct blending — catheter segments behind the patient obstruct correctly.
- **Per-segment attenuation profile (5-zone)**: The catheter is modeled with 5 material zones — tungsten tip marker, nitinol hypotube, polymer jacket, platinum radiopaque band, and lumen (air/saline). Each zone has a calibrated μ value at 70 keV. Zone boundaries are defined by segment index ranges.
- **Cone-beam magnification**: The projected width of a catheter segment is magnified by the ratio SDD/SOD (source-to-detector distance / source-to-object distance), following standard radiographic geometry. This corrects for parallax at different catheter depths.

**Detector physics (rows 5–6)**
- **Poisson noise**: I_detected ~ Poisson(λ = I · N_photons). With N_photons set to the clinical count (~10⁵ photons/mm²), the resulting noise standard deviation matches clinical quantum noise.
- **Scatter**: A Gaussian kernel convolution G(σ_scatter) * I modeled as veiling glare — low-frequency background haze.
- **PSF (detector blur)**: A second convolution simulates the detector point spread function, primarily due to the scintillator layer thickness and optical coupling. PSF σ is configurable per C-arm preset.
- **Gamma correction**: I_display = I_linear^(1/γ) where γ=0.8. This matches the display transfer function of clinical PACS monitors.
- **Misregistration jitter**: Applied to the mask DRR before DSA subtraction. Small random rotation (σ=0.05°) and translation (σ=0.1 mm) simulate sub-pixel patient motion between the mask and contrast acquisition.

**DSA pipeline (rows 7–11)**
The complete 4-step DSA chain (`dsa.py DSAPipeline`): (1) contrast DRR with μ_vessel boosted by iodine attenuation, (2) mask DRR with vessels mapped back to tissue μ + jitter, (3) DSA = I_mask − I_contrast to isolate vessel signal, (4) post-processing: scale × 20, gamma, scatter, Poisson noise. Vessel boost applies a multiplicative factor A=8 to vessel-masked voxels.

**Bolus dynamics (rows 9–11)**
VMTK centerline extraction (`extract_centerlines`) builds a graph with ~5,000+ nodes for a cerebral vasculature. Dijkstra on edge weights w(i,j) = dist(i,j)/v(r̄) gives per-node arrival times T. A KDTree maps CT voxels to nearest graph nodes for their arrival time. `build_contrast_volume` applies C(t−T(v)) per voxel per frame. `render_cine` accepts a `volume_callback(t)` hook to update the μ volume before each frame render.

**Physics solver (rows 16–19)**
- **Batched XPBD**: `_BatchedWorkspace` concatenates all rod state tensors into single flat GPU buffers. `rod_offsets[i]` gives the start index for rod i's particles. All 11 batched kernel variants process every rod in a single launch.
- **GPU root control**: `apply_proximal_control_gpu(push_v, rotate_v, dt)` is a Warp kernel that updates the root particle position and orientation in-place without CPU involvement. `set_root_orientation(env_idx, q)` uses a single-thread kernel launch — safe to call inside a captured CUDA graph.
- **CUDA-graph capture**: `step()` wraps the substep loop in `wp.ScopedCapture`. On first call, the graph is captured. On subsequent calls, `wp.capture_launch(graph)` replays it. This eliminates Python interpreter and CUDA launch overhead, reducing per-step CPU time from ~1 ms to ~10 µs.
- **Floor collision restitution**: The floor constraint now supports a configurable restitution coefficient (0 = perfectly inelastic, 1 = perfectly elastic). Implementation: when a particle penetrates the floor plane, the normal velocity component is reversed and scaled by the restitution coefficient.

**RL pipeline (rows 20+)**
Multi-env Slang renderer and Newton wrapper multi-env are described in Slides 12 and the Closed Gaps slide respectively.

---

## Slide 12 — This Release — Closed Gaps vs XCATH Requirements

**Talking Points**

This slide demonstrates alignment with the XCATH partner requirements. Every row is a capability XCATH either explicitly requested or that was identified as a gap during their March–April 2026 review sessions.

**Key rows to highlight**:

**Multi-env XPBD (self-contained solver)**
Before this sprint, `xpbd_rod_solver.py` was single-environment only — it allocated separate `Workspace` objects per environment with individual Warp array allocations. This was N × overhead. The new `_BatchedWorkspace` (PR-equivalent) allocates once: `positions = wp.zeros((total_particles, 3), ...)` where `total_particles = sum(particles_per_rod)`. Each rod's data is located at `rod_offsets[i]:rod_offsets[i+1]`. The 11 batched kernels (`predict_positions_batched`, `update_constraints_batched`, etc.) iterate over this flat layout using the offset arrays.

**Multi-env Slang fluoroscopy renderer**
The critical change in `diffdrr_slang.slang` was changing the kernel signature from `[shader]` to `[CudaKernel]` and adding `uint3 dispatchThreadID : SV_DispatchThreadID`. The Z component is `envIdx`. The Python driver (`diffdrr_slang_renderer.py`) calls `kernel.dispatch([W, H, N])` where N = num_envs. Each env reads its own pose from `StructuredBuffer<Pose> poses[envIdx]` and its own catheter segments from the flat `StructuredBuffer<CatheterSegment>` sliced by `StructuredBuffer<int> catheter_offsets[envIdx]`.

**Newton XPBD wrapper multi-env**
`newton_xpbd_rod_wrapper.py` previously hard-coded `num_envs=1` and raised `NotImplementedError` for batched calls. The fix delegated to Newton's own `_BatchedRodWorkspace` (Newton PR #1981), which supports arbitrary `num_envs`. The wrapper now constructs `_BatchedRodWorkspace(num_rods=num_envs, ...)` and calls Newton's batched `step()`. Note: unlike the self-contained solver, the Newton wrapper does not support CUDA-graph capture because Newton's substep loop has CPU synchronization points.

**E2E catheter motion in renders**
This was the key end-to-end validation. The workflow: (1) initialize 20-segment nitinol rod at origin, (2) apply proximal push at 5 mm/s for 7 seconds, (3) capture Slang DRR frames at keyframes. The catheter tip advances ~35 mm in the cranial AP direction. Beer-Lambert polarity was confirmed correct: catheter appears as darker attenuation (not white enhancement) on the raw DRR, consistent with X-ray physics (metallic catheter attenuates more than tissue). The 5-zone μ_profile with nitinol values produces ~3× more attenuation than soft tissue, matching clinical catheter appearance.

**Vessel mesh collision (XCathRodSolver)**
New in this sprint. Full details on Slide 13.

---

## Slide 13 — XCathRodSolver — Vessel Mesh Collision COMPLETED

**Talking Points**

This slide documents the `XCathRodSolver` — the most technically novel feature of this sprint. It was ported from Newton's internal `xpbd_rods_solver_integr` branch (authored by Przemek Korzeniowski) into IsaacLab's self-contained solver framework.

**Design Philosophy**
The key insight is that `XCathRodSolver` is a *subclass* of `XPBDRodSolver`. It does not modify any of the XPBD constraint projection logic. Instead, it hooks into two new extension points added to the base class:
- `_pre_constraints_hook(ws, dt, device)`: called after position prediction but before constraint projection.
- `_post_constraints_hook(ws, dt, device)`: called after constraint projection, before velocity update.

These hooks are injected into both `_substep` (single-env) and `_substep_batched` (multi-env), meaning any subclass can add collision or other forces at either point in the substep loop.

**Mesh Representation**
The collision mesh is a `wp.Mesh` object, which stores vertex positions and triangle indices on GPU and auto-constructs a BVH (bounding volume hierarchy) for spatial queries. Any closed triangle mesh can be used — the E2E validation used a procedurally generated S-bend tube (13 mm radius, ~500 triangles) representing a stylized aortic arch. In production, the mesh would come from vessel segmentation (CT → vessel mask → marching cubes → smoothed mesh → `wp.Mesh`).

**SDF Path (default)**
Implemented in `_project_vessel_containment_kernel`. For each rod particle position `p`:
1. Call `wp.mesh_query_point_sign_normal(mesh.id, p, max_dist)` → returns `(hit, sign, position_on_surface, normal, face_id)`.
2. Compute `phi = sign * ||p - position_on_surface||`. With outward-pointing mesh normals and `sign_scale=1.0`: phi < 0 means inside the vessel, phi ≥ 0 means outside (in the wall or beyond).
3. If `phi > target_phi` (particle is outside the desired clearance region), compute correction: `dp = -(phi - target_phi) * normal`, clamp by `max_delta_lambda`.
4. Apply `p += dp`.

The `target_phi` parameter sets the desired clearance from the wall. With outward normals, `target_phi = -0.001` (−1 mm) means particles are held 1 mm inside the vessel wall.

**AABB / Edge Path**
For vessels with tight curvature where the SDF may not resolve contact correctly:
1. `_project_mesh_vertex_collision_kernel_averaged`: uses `wp.mesh_query_aabb(mesh.id, aabb_lower, aabb_upper)` to find all triangles whose bounding box overlaps a sphere of radius `particle_radius` around the particle. For each candidate triangle, computes the closest point on the triangle (`_closest_point_triangle`) and a correction proportional to penetration depth.
2. `_project_mesh_edge_collision_kernel_averaged`: for each rod segment (edge), finds candidate mesh edges via AABB query, then computes the closest point between the rod segment and the mesh edge (`_segment_segment_barycentric`), and adds a correction.
3. Corrections are accumulated atomically and flushed at the end of each Gauss-Seidel iteration.

**Track-Guided Insertion**
`_track_sliding_kernel` implements the catheter insertion axis constraint. Given a track defined by `(track_start, track_dir, track_length)`, all rod particles except the distal `tip_num_edges` segments are projected onto this axis. This models the fact that the proximal catheter shaft is constrained by the introducer sheath or guide catheter — only the steerable tip is free to deflect.

**E2E Validation Result**
A synthetic S-bend tube mesh (300 mm long, 13 mm radius, Z-undulating centerline with ±5 mm excursion) was used. The solver was run for 4 seconds of simulated time with a 3 mm/s proximal push. Without collision, the catheter falls straight under gravity. With `XCathRodSolver` active, the catheter tip follows the vessel Z-undulation: Δz = +2.7 mm over a 40 mm push, consistent with the vessel geometry. Output frames are in `docs/e2e_vessel/`.

**Future Work**
The `_substep_batched` hook is already wired; `XCathRodSolver._pre/_post_constraints_hook_batched` just need to be implemented to accept a per-env mesh index, enabling one mesh per environment at 512+ envs. This is the primary Sprint 2 physics item.

---

## Slide 14 — PART 3: Next Release (Section Divider)

**Talking Points**

Sprint 2 targets four areas: (1) GPU rendering efficiency via Texture2DArray, (2) multi-env vessel collision, (3) image-based RL observations, and (4) GPU-side detector physics. The goal is to close the remaining gap to a fully GPU-resident, 512-env training loop with fluoroscopy image observations.

---

## Slide 15 — Next Release — X-Ray Sprint 2 (Weeks 3–4)

**Talking Points**

Two items on this slide are already completed from this sprint — surface them clearly before discussing what remains.

**[OK] Multi-env Slang renderer (COMPLETED)**
`renderDRR_forward_batched` is live. The batched dispatch with `dispatchThreadID.z = envIdx` is implemented and smoke-tested. The remaining item is the `Texture2DArray` upgrade to improve cache efficiency at N>8.

**[OK] Vessel mesh collision (COMPLETED)**
`XCathRodSolver` is implemented, smoke-tested, and E2E validated. The remaining item is extending the batched hook to support per-env meshes.

**Texture2DArray upgrade**
Current output: `RWStructuredBuffer<float>` of size `[N, H, W]` indexed as `envIdx * H * W + y * W + x`. Problem: for a given warp of 32 threads rendering adjacent pixels, some pixels are in different envs, causing L1 cache line thrashing on the output write. Fix: change to `RWTexture2DArray<float>` indexed as `[x, y, envIdx]`. The GPU texture cache is specifically optimized for 2D spatial locality — adjacent pixels in the same env will hit the same cache lines.

**Multi-env vessel collision**
Extend `XCathRodSolver._post_constraints_hook_batched` to:
1. Accept a `wp.array(dtype=wp.uint64)` of mesh IDs, one per env.
2. Launch the containment kernel with env-specific mesh ID indexed by `env_id`.
3. Requires each env to have its own `wp.Mesh` BVH — acceptable since BVH construction is a one-time cost at initialization.

**Image-based RL observations**
Wire-up path:
1. After `env.step()`, call `renderer.render_batched(rod_positions)` → returns `(N, H, W)` float tensor.
2. Normalize to [0, 1], reshape to `(N, 1, H, W)`.
3. Add `image_obs` key to the observation dict with shape `(N, 1, H, W)`.
4. Configure RSL-RL PPO to use a CNN encoder (e.g., NatureCNN: 3 conv layers → 512-dim feature) or ViT.
5. Total observation = CNN(image_obs) concat state_obs.

**GPU-side detector physics**
Currently, `realism.py` applies Poisson noise, scatter convolution, and PSF in NumPy. This runs on CPU and takes ~5 ms per frame (fast for offline use, too slow for 512-env training loops where we need ≤1 ms). Port plan: Poisson noise → Warp kernel using `wp.rand_poisson`; scatter convolution → Slang separable Gaussian kernel; PSF → same. These can be fused into the existing `renderDRR_forward_batched` call.

---

## Slide 16 — Summary

**Talking Points**

Use this slide as the closing synthesis. Emphasize the progression from TODAY → THIS RELEASE → NEXT → FOLLOWING.

**TODAY (steady-state baseline)**
- The X-ray fluoroscopy pipeline is implemented end-to-end: physics → render → DSA → RL. Every component runs on GPU (except the CPU detector physics path).
- Performance: physics ~1,300 Hz (single-env), rendering ~25 FPS (single-env). These are not targets — they are measured numbers.
- DSA, bolus dynamics, vessel boost, and 9 vendor C-arm presets are all operational.
- State-based PPO training at 512 envs is running.

**THIS RELEASE (Sprint 1 deliverables)**
- The full DSA pipeline with temporal bolus dynamics, detector physics, and misregistration jitter is implemented and validated.
- The self-contained XPBD solver is now multi-env with GPU root control and CUDA-graph capture.
- The Slang renderer supports batched multi-env dispatch.
- The Newton XPBD wrapper multi-env restriction is removed.
- E2E validation: catheter traverses 35 mm in Slang renders with correct Beer-Lambert attenuation polarity.
- **Vessel mesh collision** (`XCathRodSolver`): SDF BVH + AABB/edge + track-guided insertion — ported from Newton, smoke-tested, E2E validated.
- **21+ XCATH-required capabilities closed**: this includes every item from the March 2026 XCATH requirements review.

**NEXT RELEASE (Sprint 2)**
Four priorities: Texture2DArray rendering efficiency at N>8, multi-env vessel collision, image-based RL observations, and GPU-side detector physics.

**FOLLOWING (Sprint 3 + phases)**
- Sprint 3: full training readiness with domain randomization, Gymnasium wrappers, and per-task CUDA graphs.
- Phase 2: 7 OpenClaw skill packages (each skill = CT ingestion, physics sim, X-ray render, dataset creation, reward function, policy training, evaluation).
- Phase 3: agentic workflow integration — natural language → config → iterative refinement via NemoClaw/OpenClaw orchestration.
- Realism metrics (FID, SSIM, vessel visibility) for simulation-to-real gap quantification.
- Workflow extensions: force/torque sensor integration, CBCT (cone-beam CT) rendering.

**Closing statement for audience**
> Sprint 1 proves the stack is real — every number on the performance slide is measured, every deliverable on the completed slide has a file path. Sprint 2 closes the remaining gap to full GPU-resident, image-based RL training at 512 environments.

---

## Slide 17 — Questions?

**Talking Points**

Anticipate these likely questions:

**Q: Why not just use DeepDRR for rendering?**
A: DeepDRR is polyenergetic and physically more accurate, but it is not differentiable, not GPU-CUDA-graph capturable, and runs at ~2 FPS (vs our 25 FPS). For DSA specifically, the SSIM=0.964 result shows monoenergetic is practically sufficient. For future beam-hardening closure, we can extend Slang with a spectral model without sacrificing differentiability.

**Q: Is the catheter physics validated against real data?**
A: The Cosserat rod model is the industry-standard model for flexible medical instruments. The production solver (`rod_solver.py`) uses Newton's validated block-tridiagonal JMJT formulation. The self-contained XPBD solver has been verified to produce equivalent trajectories to the production solver in smoke tests.

**Q: How does the vessel collision scale to complex anatomies?**
A: The SDF path uses Warp's BVH — query complexity is O(log N) per particle, regardless of mesh triangle count. For a realistic aortic arch with ~50,000 triangles, this is still sub-millisecond per particle. The AABB/edge path adds broadphase filtering to remain efficient for tight curvatures.

**Q: What is the expected timeline for image-based RL observations?**
A: The rendering path (multi-env Slang) is already implemented. The remaining work is the wire-up to RSL-RL obs dict and the CNN encoder configuration — estimated 1–2 weeks. The GPU detector physics port is a further 1 week. So image-based RL training should be available in Sprint 2 (Weeks 3–4).

**Q: What is the role of XCATH in this collaboration?**
A: XCATH provides clinical requirements, validation data (50 patient CTA+DSA pairs), and real-time feedback on simulator realism. We provide the simulation infrastructure. Their feedback drives prioritization — for example, the F-8 selective injection, dispersion correction, and bolus timing calibration all came from their April 2026 review.

---

*Document generated from: `sensor_simulation_release_deck_compact.pptx` (17 slides)*
*Implementation references: `source/isaaclab_newton/isaaclab_newton/solvers/`, `i4h-sensor-simulation-internal/`*
*Last updated: May 2026*
