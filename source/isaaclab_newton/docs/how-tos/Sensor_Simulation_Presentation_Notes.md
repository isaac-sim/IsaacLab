# Sensor Simulation — Presentation Notes

**Prepared for:** Technical Review  
**Date:** 2026-04-09  
**Document Reference:** Sensor_Simulation.docx  
**Codebase References:** `isaaclab_newton/solvers/` (X-ray), `ultrasound-raytracing/raysim/slang/` (Ultrasound)

---

## SLIDE: Executive Summary

**Speaker notes:**

We have two sensor modalities implemented and running on GPU — X-ray fluoroscopy and ultrasound B-mode. Both share a common architectural pattern: a physics-based rendering kernel on GPU feeding into a post-processing pipeline, with the output packaged as a tensor for downstream consumption. The architecture is designed to be extensible — the document covers five additional sensor types (endoscopy, CBCT, force/torque, IVUS, hemodynamics) as future additions.

Key point for the audience: X-ray uses DiffDRR+Slang for pre-baked DRR generation with a C-arm pinhole projection model for catheter compositing. Ultrasound uses a real-time Slang/CUDA volumetric ray-marcher with BVH-accelerated geometry intersection. The ultrasound path is fully differentiable with respect to 6-DOF probe pose for gradient-based optimization.

---

## SLIDE: Sensor Status Table

**Speaker notes:**

Walk through each row:

- **X-Ray Fluoroscopy** — Implemented. Rendering backend is DiffDRR + Slang, meaning we pre-generate digitally reconstructed radiographs offline from CT volumes and load them as PNG backgrounds at runtime. Catheter compositing happens either in 3D via Omniverse's renderer (USD textured quad backdrop + capsule markers) or in 2D via OpenCV (C-arm pinhole projection + polyline overlay). Not differentiable today. Multi-env is single-env only on the XPBD solver; the production RodSolver supports batched environments.

- **Ultrasound B-Mode** — Implemented. Rendering backend is a custom Slang/CUDA volumetric ray-marcher with BVH-accelerated ray-triangle intersection using Möller-Trumbore. Differentiable with respect to 6 DOF — three for probe position, three for probe Euler angles (ZXY convention). The backward pass is implemented in Slang via `bwd_diff(computeElementWeightedScattering)` and covers the Phase 1 volumetric scattering path. Phase 2 (specular geometry) blocks gradient flow because BVH traversal and branch-heavy surface interaction logic are not marked `[Differentiable]` in the Slang source.

- **Endoscopy / RGB** — Planned. Leverage Isaac Lab's existing `Camera` sensor class with the Omniverse RTX renderer. The main work is realistic vessel interior USD assets.

- **CT / CBCT** — Planned. Extends the fluoroscopy GPU ray-caster to perform a rotational sweep (200-400 projections) with FDK reconstruction.

- **Force/Torque** — Planned. Not a visual sensor — derived directly from the XPBD collision solver's contact response. Contact forces are already computed during position-level projection (SDF gradient times penetration depth times contact stiffness). Warp autodiff propagates through the contact solver, making this differentiable.

---

# PART 1: X-RAY FLUOROSCOPY

---

## SLIDE: Catheter Physics Solvers — RodSolver (Production)

**Speaker notes:**

The production solver lives in `rod_solver.py` (1,122 lines) with Warp GPU kernels in `rod_kernels.py` (1,249 lines). It implements an XPBD Cosserat rod model following Deul et al.

The algorithm runs in two modes:

**Direct solver mode** (`use_direct_solver=True`, the default): For each substep, we predict positions and orientations via explicit Euler with gravity and damping, then run Newton iterations (default 4 per substep). Each iteration evaluates constraint residuals (stretch and Darboux), builds a 6x6 system per constraint (3 translational + 3 rotational DOFs), and solves each block independently via `torch.linalg.solve`. Important technical detail: the current direct path builds a block-diagonal approximation — it does not fill the off-diagonal coupling blocks between adjacent constraints. The docstring references a "linear-time Thomas" solve, but the implementation independently solves each 6x6 block. This is mathematically equivalent to a single Jacobi iteration on the block system, not a full coupled solve.

**Gauss-Seidel mode** (`use_direct_solver=False`): Uses Warp kernels directly — `solve_stretch_bend_fused_kernel` handles stretch and bend/twist constraints in a single kernel launch when `use_fused_kernel=True`. Shear constraints are handled in a separate pass via `solve_shear_constraints_kernel` when `shear_stiffness > 0`.

Constraint types:
- **Stretch:** Enforces coincidence between attachment points at segment boundaries — the vector between the end of the parent segment frame and the start of the child segment frame should be zero.
- **Bend/twist (Darboux):** Computes the relative rotation between adjacent frames via `compute_darboux_vector` and penalizes deviation from `rest_darboux`. Tip shaping uses `RodTipConfig` to set non-zero rest Darboux vectors on the distal segments, producing J-tip or Simmons-style curves.
- **Shear:** Optional; activated when `shear_stiffness > 0`. Penalizes lateral displacement of the segment midpoint relative to the connecting line.

Collision handling:
- **Ground plane** (`solve_ground_collision_kernel`): Horizontal plane at `ground_height`; corrects segment center Y coordinate accounting for radius.
- **Mesh BVH** (`solve_mesh_collision_kernel`): Uses Warp's `mesh_query_point` BVH closest-point query. Penetrating particles are projected along the surface normal. Contact normals and depths are stored for downstream friction computation.
- **Self-collision** (`solve_self_collision_kernel`): O(n-squared) pairwise sphere-sphere test over non-adjacent segments with inverse-mass weighted separation.
- **Friction:** Applied after mesh collision using stored contact normals/depths. Coulomb friction kernel applies tangential impulse proportional to normal force times friction coefficient. Viscous friction applies velocity-proportional damping. Static-dynamic model uses both.

Multi-env is supported — all workspace arrays carry a leading environment dimension; kernel launches are indexed by `(env_id, particle_id)`.

---

## SLIDE: Catheter Physics Solvers — XPBDRodSolver (Self-Contained)

**Speaker notes:**

This is a self-contained port of Newton's `SolverXPBDRod`, living entirely in `xpbd_rod_solver.py` (1,168 lines) with all 16 Warp kernels and 30+ helper functions embedded. Zero external dependency — no Newton package required.

The critical technical distinction from `RodSolver`: this solver assembles and solves the **full coupled block-tridiagonal system** via a block-Thomas algorithm, rather than solving each constraint independently. This means constraint coupling between adjacent edges is captured in a single linear solve.

**Algorithm per substep:**
1. Predict positions (`_xr_predict_pos`) and rotations (`_xr_predict_rot`) via explicit Euler with gravity and per-mode damping.
2. Zero the Lagrange multiplier accumulators and compute compliance values (`_xr_prepare_compliance`). Stretch compliance is set near-zero (1e-10, effectively rigid). Bend compliance is `1 / (E * I * L * dt^2)` where E is Young's modulus, I is the cross-section second moment, and L is the rest length. Twist compliance uses the torsion modulus instead of Young's.
3. Evaluate constraint errors (`_xr_update_constraints`): 6 scalar equations per edge — 3 stretch (coincidence of attachment points using half-length offsets along local Z axis) and 3 Darboux (imaginary part of `conj(q0)*q1` minus rest Darboux).
4. Compute 6x6 Jacobians (`_xr_compute_jacobians`) for both position and rotation DOFs.
5. Compute world-frame inverse inertia (`_xr_compute_inv_inertia`) from quaternion-rotated local diagonal inertia.
6. Assemble the block-tridiagonal JMJT matrix (`_xr_assemble_jmjt`): diagonal blocks are 6x6, coupling arises because adjacent edges share a particle. Off-diagonal blocks capture the shared-particle mass and inertia contributions.
7. Build the right-hand side (`_xr_build_rhs`): `-c - alpha * lambda_sum` where c is the constraint error vector and alpha is the compliance-scaled accumulated multiplier.
8. Solve via block-Thomas (`_xr_block_thomas`): This is the core — a single GPU thread executes the forward elimination and backward substitution sweeps over n_edges 6x6 blocks. Each elimination step uses a 6x6 Schur complement via four 3x3 Cholesky factorizations. Complexity is O(n) in the number of edges with constant 6x6 block work per step.
9. Compute corrections (`_xr_compute_corrections`): J-transpose times delta-lambda, scaled by inverse mass and inverse inertia, accumulated via `wp.atomic_add` into per-particle correction buffers.
10. Apply corrections (`_xr_apply_corrections`): Add position corrections directly; apply rotation corrections via quaternion exponential map (`_xr_qcorr`).
11. Optional floor collision (`_xr_floor_collision`): Clamp predicted Z to `floor_z`, zero downward velocity.
12. Integrate (`_xr_integrate_pos`, `_xr_integrate_rot`): Derive velocities from position/orientation change, commit predicted state.

**Current limitations:** Single environment only (`num_envs != 1` raises `NotImplementedError`). Floor collision only — no mesh BVH, no self-collision, no friction. Tip shaping not implemented (rest Darboux fixed at zero). No external force/torque API exposed on the public class (the workspace arrays exist internally but have no setter).

---

## SLIDE: Catheter Physics Solvers — NewtonXPBDRodSolver (External Bridge)

**Speaker notes:**

This is a thin wrapper in `newton_xpbd_rod_wrapper.py` (242 lines) that bridges Isaac Lab's `RodConfig` dataclass to the upstream Newton package's `SolverXPBDRod` class from PR #1981. It translates our config parameters to Newton's `ModelBuilder` API, calls `xpbd_rod.add_elastic_rod`, and delegates the step call to Newton's solver.

The value here is access to Newton's multiple backend solvers: `block_thomas`, `split_thomas`, `block_jacobi`, and `banded_cholesky`. The split-Thomas variant decomposes the forward sweep into two independent half-sweeps from each end, meeting in the middle — this doubles throughput by using two threads. The banded Cholesky backend treats the full system as a banded matrix and factors it directly.

This solver requires the external `newton` package with XPBD rod support. Single environment only.

---

## SLIDE: Fluoroscopy Rendering

**Speaker notes:**

Two compositing paths exist:

**3D Isaac Lab compositing** (`visualize_rod_fluoro_isaaclab.py`): We create a USD `UsdGeom.Mesh` quad in the XZ plane at Y=-0.5, apply the DRR as a `UsdUVTexture` with `UsdPreviewSurface` using `emissiveColor` so the image is self-illuminated (not affected by scene lighting). The catheter is rendered as `VisualizationMarkers` with capsule geometry — fixed/shaft/tip segments get different materials. Compositing happens naturally through the Omniverse renderer's depth ordering.

**2D OpenCV compositing** (`visualize_rod_fluoroscopy.py`, ~800 lines): This is the primary development visualization. The pipeline:
1. Pre-generated DRR PNGs are loaded from a directory (default path points to DiffDRR+Slang output). Each PNG corresponds to a specific C-arm angle.
2. C-arm intrinsics: `make_carm_intrinsics(sid, pixel_spacing, image_size)` computes focal lengths as `f_x = f_y = SID / pixel_spacing` with the principal point at image center. This is a standard pinhole camera model where SID (source-to-image distance) is the focal length in physical units and pixel_spacing converts to pixel units.
3. C-arm extrinsics: `make_carm_extrinsics(sod, carm_angle_deg, cran_caud_deg)` builds the rotation matrix as `R = R_x(cranial/caudal) * R_y(LAO/RAO)`. The camera center is positioned so the iso-center is at the world origin, with the default source at `(0, 0, -SOD)` looking along +Z.
4. Projection: Standard `P = K * [R|t]`, homogeneous divide to get pixel coordinates `(u, v)`.
5. Catheter overlay: Currently draws an anti-aliased polyline with Gaussian blur "glow" on top of the DRR background. Segment centers are drawn as colored dots. The scale factor is 1000x (solver uses metres, fluoroscopy uses mm).

**Current limitation — technically important:** The catheter is rendered as a bright opaque line painted on top of the X-ray. This is physically wrong. X-ray imaging is transmissive — the catheter absorbs photons, so it should darken the background image, not brighten it. The overlay completely obscures background anatomy, produces hard edges with no partial transparency, and cannot model self-crossing darkening or material-dependent opacity variation (platinum markers should be much darker than polymer segments).

Interactive controls: C-arm angle adjustment selects the nearest pre-computed DRR; supports AP, LAO 30, lateral, RAO 30 views. DICOM loading with window/level is supported via `load_fluoroscopy_background`.

---

## SLIDE: X-Ray Performance Baseline

**Speaker notes:**

Walk through each metric:

- **Single-env physics at >1,000 Hz — achieved at ~1,300 Hz** on A6000 with a 20-segment rod. This is the XPBD solver running 2 substeps per physics frame. The per-substep time is dominated by the 15 kernel launches (predict, compliance, constraints, Jacobians, inertia, assembly, RHS, Thomas, corrections, apply, integrate).

- **Single-env compositing at <2 ms — within budget.** The current polyline overlay is under 1 ms. Beer-Lambert compositing is estimated at ~2 ms because it requires per-pixel distance computation to the projected catheter centerline and exponential evaluation, but this only touches pixels within r_max of the centerline (a few thousand pixels per frame, not the full image).

- **512-env physics at >60 Hz — not available.** The XPBDRodSolver is single-env only. This is the Sprint 2 target. The block-Thomas kernel already runs as a single thread per rod, so multi-env parallelism is embarrassingly parallel across rods — we launch `dim=num_envs` threads, each solving its own block-tridiagonal system.

- **GPU memory at 4096 envs — estimated ~200 MB (physics only).** A 20-segment rod uses approximately 50 KB of workspace arrays. At 4096 environments, that is 200 MB, well within the A6000's 48 GB.

---

# PART 1: ULTRASOUND

---

## SLIDE: Slang Ultrasound Renderer

**Speaker notes:**

The ultrasound renderer lives in `ultrasound_slang_renderer.py` (1,328 lines of Python) and `ultrasound_slang.slang` (1,417 lines of Slang shader code). It implements a volumetric ray-marching algorithm with BVH-accelerated ray-triangle intersection.

**Architecture overview:** The renderer loads two shader entry points from the Slang file — `renderUltrasound_forward` and `renderUltrasound_backward`. The forward pass produces a depth-resolved RF signal per transducer element. The backward pass computes gradients of the output with respect to 6-DOF probe pose.

**Input data:** Two 3D float32 textures — a density volume and an amplitude/scattering volume — uploaded as GPU 3D textures in ZYX memory order with XYZ extent metadata passed via a `VolumeInfo` struct. These are typically derived from CT data (Hounsfield units mapped to acoustic properties).

**Configuration (`SlangUltrasoundConfig`):** Default step size is 0.5 mm, maximum depth 200 mm, depth buffer size 2048 samples, 128 transducer elements, 1 elevational sample. The step size is fixed — no adaptive stepping.

---

## SLIDE: Ultrasound Physics — Phase 1 (Volumetric)

**Speaker notes:**

Phase 1 handles volumetric scattering and Beer-Lambert attenuation — this is where tissue texture (speckle, organ parenchyma) comes from.

**Ray marching algorithm:** Fixed-step march with step size `stepMM` (default 0.5 mm). Each step: compute world position along the ray, sample the scattering texture via trilinear interpolation, apply attenuation, accumulate into a depth-resolved buffer indexed by `depthIdx = int((t / maxDepth) * (bufferSize - 1))`.

**Beer-Lambert attenuation (exact implementation from the Slang source):**

The `applyAttenuation` function computes amplitude scaling as `intensity * 10^(-alpha * f * d_cm / 20)` where:
- `alpha` is the material attenuation coefficient in dB per cm per MHz
- `f` is the transducer frequency in MHz
- `d_cm` is the distance converted from mm to cm (multiply by 0.1)
- The division by 20 converts from dB to amplitude ratio (not power — amplitude is voltage, so it is 20*log10, not 10*log10)

This follows the standard ultrasound attenuation model where tissue attenuation scales linearly with frequency and depth.

**Scattering model:** The scattering value at each voxel is computed from the amplitude texture weighted by an effective scattering coefficient: `sigmaEff = sigma * (f / f_ref)^n`, where `sigma` is the base scattering coefficient, `f` is the frequency, `f_ref` is a reference frequency (defaults floored at 0.001 MHz), and `n` is the frequency exponent. When `n = 0`, scattering is frequency-independent. The density texture acts as a threshold gate — scattering only occurs where `density <= mu0` (the material's density threshold parameter). This allows different tissue types to produce different speckle patterns.

**Depth-resolved output:** The march writes contributions to a 1D depth buffer per transducer element. The contribution at each step is `scattering * intensity * stepMM` — the step size acts as an integration weight. The depth buffer has `buffer_size` bins (default 2048), uniformly spaced over `max_depth_mm`.

All functions in the Phase 1 path are marked `[Differentiable]` in Slang, enabling gradient computation for probe pose optimization.

---

## SLIDE: Ultrasound Physics — Phase 2 (Geometry / Specular)

**Speaker notes:**

Phase 2 handles specular reflection and refraction at tissue boundaries — this is where organ edges, bone surfaces, and interface echoes come from.

**Ray-triangle intersection:** Möller-Trumbore algorithm implemented in Slang as `rayTriangleIntersect`. Standard cross-product formulation with epsilon tolerance of 1e-12 for degenerate triangles. Returns hit distance `t`, barycentric coordinates, and face normal.

**BVH acceleration structure:**
- **Build (CPU):** Median split on triangle centroids along the longest axis of the bounding box. Leaf termination at 8 triangles or degenerate extent. The Python `_build_bvh()` method constructs the tree and reports node count, leaf count, and estimated depth.
- **GPU layout:** Three flat arrays: `bvhBounds` (6 floats per node — AABB min/max), `bvhInfo` (4 ints per node — left child, right child, triangle start, triangle count — leaf when triCount > 0), `bvhTriOrder` (permuted triangle indices).
- **Traversal:** Stack-based DFS with an explicit `int stack[32]` in the Slang shader. Ray-AABB slab test with precomputed `invDir`. Stack depth capped at 30 when pushing children. Iteration limit `[MaxIters(1024)]` on the traversal loop for the Slang compiler. Falls back to brute-force when `numBVHNodes == 0`.

**Impedance mismatch reflection coefficient:** `R = ((Z2 * cos(theta) - Z1) / (Z2 * cos(theta) + Z1))^2` where Z1 and Z2 are the acoustic impedances (in MRayl) of the incident and transmitted media, and theta is the angle of incidence computed from `acos(|dot(rayDir, surfaceNormal)|)`. This is the intensity reflection coefficient for a planar acoustic interface.

**Refraction (Snell's law):** `sin_t = (c1/c2) * sin_i` where c1 and c2 are the speeds of sound in the two media. Total internal reflection occurs when `sin_t >= 1`. The refracted direction is computed as `(c1/c2)*I + ((c1/c2)*cos_i - cos_t)*N` after aligning the normal to oppose the incident direction.

**Multi-bounce handling:** Up to 3 bounces (the loop runs `bounce <= MAX_BOUNCES` with `MAX_BOUNCES = 3`, so 4 surface interactions are possible). Each segment between bounces contributes volumetric scattering via `writeSegmentScattering` with accumulated path length offset and combined transmission intensity. At each hit: specular intensity is written to the depth bin at `tAccumulated + tGeomHit`, coherence attenuation decays as `0.3^bounce`, and the ray continues along the refracted direction with intensity scaled by `(1 - R)`. Self-intersection avoidance uses a 0.01 mm offset along the new ray direction. Early termination when `intensity < MIN_INTENSITY` (0.001) or total internal reflection.

**Differentiability limitation:** Phase 2 is not differentiable. The BVH traversal, branch-heavy surface interaction logic, and `writeSegmentScattering` are not marked `[Differentiable]` in the Slang source. Gradients only flow through Phase 1 (volumetric path).

---

## SLIDE: Post-Processing Pipeline

**Speaker notes:**

All six stages run on GPU via PyTorch CUDA tensors — no CPU roundtrips in the pipeline. The entry point is `post_process_bmode_gpu` in `postprocessing.py`.

**Stage 1 — PSF Convolution:** Separable 1D convolution. Axial PSF is a Gaussian envelope modulated by a cosine at the spatial frequency `f_spatial = frequency_mhz / c_mm_us` where `c_mm_us = 1.54` mm/us (speed of sound in tissue). The Gaussian sigma is derived from the acoustic wavelength. Lateral PSF is a pure Gaussian with sigma approximately `6 * wavelength`, with spacing determined by `probe_width / (num_elements - 1)`. Both kernels are L1-normalized before convolution.

**Stage 2 — Time-Gain Compensation (TGC):** Applies depth-dependent gain to compensate for attenuation: `gain = 10^(tgc_db * depth_norm / 20)` where `depth_norm` is a linear ramp from 0 to 1 over the depth axis. The `tgc_db` parameter controls the total compensation in dB. Skipped if `tgc_db <= 0`.

**Stage 3 — Hilbert Envelope Detection:** Computes the analytic signal via FFT along the depth axis: zero the negative frequency components, inverse FFT, take the magnitude. This extracts the envelope of the RF signal, removing the carrier frequency oscillation. Handles both even and odd buffer lengths.

**Stage 4 — Log Compression:** Normalize by the 99.9th percentile of positive samples, then compute `20 * log10(envelope + epsilon)`. This compresses the ~60 dB dynamic range of ultrasound echo amplitudes into a displayable range. The `db_range` parameter (default 60 dB) controls the visible dynamic range.

**Stage 5 — Median Clip Filter:** Artifact removal for salt-and-pepper noise. Computes a sliding median along the element (lateral) axis using `unfold`. Each pixel is clipped to `[max(median, db_min), min(median, db_max)]`. This suppresses isolated bright or dark pixels while preserving structural features.

**Stage 6 — Scan Conversion:** Transforms the element-by-depth RF data to a displayable Cartesian image using PyTorch's `grid_sample` with bicubic interpolation. Three probe types supported:
- **Curvilinear:** Polar-to-Cartesian using probe radius, sector angle, and max depth. Maps `(distance, angle)` to `(depth_fraction, element_fraction)`.
- **Phased array:** Fan geometry from a point source. Maps `(theta, depth)` from `atan2` and `hypot`.
- **Linear array:** Rectilinear mapping with aspect ratio handling based on physical probe width and max depth.

---

## SLIDE: Rendering Modes

**Speaker notes:**

Four modes selectable at render time:

- **Volumetric only:** Phase 1 active, Phase 2 disabled (`enable_geometry=False`). Produces soft-tissue texture and speckle. Fully differentiable for probe pose optimization.

- **Geometry only:** Phase 2 active, Phase 1 scattering coefficient set to zero (`sigma=0`). Produces bone and organ boundary echoes only. Not differentiable.

- **Combined:** Both phases active. Produces the most realistic multi-tissue image. Not differentiable because Phase 2 blocks gradient flow.

- **Gradient mode:** Phase 1 only with backward pass enabled. The backward kernel `bwd_diff(computeElementWeightedScattering)` computes `dL/d_probePosition` (3 DOF) and `dL/d_probeEuler` (3 DOF, ZXY convention) by summing per-element gradient textures on CPU. Total: 6 differentiable DOF.

Important technical detail: the `trilinearSample` function wraps texture reads in `no_diff(volume.Sample(...))` — gradients flow through the interpolation weights but not through the voxel values themselves. This means we can optimize probe pose but cannot back-propagate into the volume data.

---

## SLIDE: Verification Status

**Speaker notes:**

Five verification scenarios documented in `DESIGN_REVIEW_VALIDATION_RESULTS.md` (February 2026, RTX A6000, slangpy 0.40.1):

1. **Attenuation test:** Constant-texture volume with known attenuation coefficient. Measured envelope decay matches the expected -3.5 dB/cm slope from the Beer-Lambert model. This validates the `applyAttenuation` function's dB-to-amplitude conversion and distance scaling.

2. **Specular reflection test:** Ray-triangle intersection on a sphere with zero volumetric scattering (`sigma=0`). Measured reflection coefficient R approximately 0.43 at the water-bone boundary, consistent with the impedance mismatch formula using Z_water = 1.48 MRayl and Z_bone = 7.38 MRayl. Multi-bounce paths produce visible secondary reflections.

3. **Scan conversion test:** Curved arc patterns in the element-depth domain transform to straight lines after polar-to-Cartesian conversion. Validates the coordinate mapping geometry for curvilinear probes.

4. **Multi-material test:** Per-triangle material lookup with different impedances produces correct acoustic shadowing. Measured 82.8 dB attenuation through bone, consistent with clinical observations of 40-80 dB/cm at diagnostic frequencies through cortical bone.

5. **Scattering spheres test:** Density threshold gating produces volumetric filling with correct speckle statistics. Depth attenuation visible as decreasing brightness with distance. Validates the interaction between the density texture, `mu0` threshold, and `sigma` scattering coefficient.

---

## SLIDE: Ultrasound Performance Baseline

**Speaker notes:**

- **Single-env render (Phase 1) at <5 ms — estimated 3-8 ms.** The cost scales with `num_elements * num_elevational_samples * num_steps * (Phase 2: bounces * BVH traversal cost)`. BVH build is one-time on CPU.

- **Post-processing at <2 ms — estimated 1-3 ms on GPU.** This covers PSF convolution, TGC, Hilbert envelope (two FFTs), log compression, median clip, and scan conversion. All operations use PyTorch CUDA kernels.

- **End-to-end frame at <10 ms (100 Hz) — estimated 8-20 ms.** Includes render + post-processing + GPU-to-CPU transfer. The variance depends on Phase 2 being enabled (BVH traversal is the main variable cost).

- **512-env batch at <50 ms (20 Hz) — not available.** Currently single environment. Sprint 2 target. The batched path would extend the Slang kernel dispatch to `(num_elements, num_elevational, num_envs)` with shared scattering volume and geometry, varying only probe pose per environment.

- **GPU memory (single env) at <500 MB — approximately 200 MB.** Dominated by the 3D volume textures. At 256-cubed float32, each volume is 64 MB.

- **Backward pass (6 DOF) at <5 ms — reported 1-2 ms.** Phase 1 only. Slang's auto-differentiation generates the backward kernel from the `[Differentiable]`-annotated functions.

---

# PART 2: NEXT STEPS

---

## SLIDE: X-Ray Sprint 1 — Close the Simulation Loop

**Speaker notes:**

Three deliverables in weeks 1-2:

**Beer-Lambert Compositing:** Replace `draw_rod_overlay()` in `visualize_rod_fluoroscopy.py` with physically-correct transmission imaging. The implementation builds a per-pixel attenuation map: for each projected segment, compute the perpendicular distance from each nearby pixel to the projected centerline, evaluate the cylinder chord-length `t(d) = 2*sqrt(r^2 - d^2)` for the projected thickness, multiply by the segment's attenuation coefficient mu, and accumulate into the attenuation map. Apply as `I_final = I_DRR * exp(-attenuation_map)`. Per-segment material variation: tungsten markers (mu*2r ~ 3.0), nitinol braid (0.8), polymer tip (0.2), platinum coil (5.0). Cone-beam radius magnification: `r_px = r_phys * SID / z_cam / pixel_spacing`. The computation only touches pixels within `r_max` of the projected centerline — order of thousands of pixels per frame, not full-image.

**Proximal Kinematic Control API:** Add `apply_proximal_control(push_velocity, rotate_velocity)` to `XPBDRodSolver`. This modifies the root particle's position (translate along insertion axis) and orientation (apply torsion) at the start of each substep, before the predict phase. The root particle already has `inv_masses[0] = 0` and `quat_inv_masses[0] = 0` (locked), so control is kinematic — we directly set position/orientation rather than applying forces.

**Imaging Realism:** Poisson quantum noise at 500-5000 photons/pixel (fluoroscopy is inherently photon-starved), veiling glare as a large-kernel Gaussian blur (sigma ~15-20 px) of the blocked-intensity map re-added at 2-5% amplitude, detector PSF as a small Gaussian (sigma ~0.5-1.0 px), and beam hardening via multi-energy-bin Beer-Lambert summation (5-10 bins spanning 20-80 keV).

---

## SLIDE: X-Ray Sprint 2 — Multi-Environment + Collision

**Speaker notes:**

**Multi-Env XPBDRodSolver:** Extend all `_Workspace` arrays with a leading environment dimension. Most kernels are embarrassingly parallel across environments — add `env_id` to thread indexing. The block-Thomas kernel is the key case: launch `dim=num_envs`, each thread solves its own independent block-tridiagonal system. Memory layout: flat arrays with `env_id * per_env_size + local_offset` indexing to avoid warp divergence in the Thomas solver. At 20 segments per rod, each environment uses approximately 50 KB of workspace. Target: 512 environments on single GPU.

**SDF/Mesh Collision:** Port the collision detection path from `RodSolver` (BVH mesh query via Warp's `mesh_query_point` + contact response) into the XPBD substep, inserted between constraint projection and integration. Position-level projection: if a predicted particle penetrates the SDF, project it to the surface along the SDF gradient.

**Observation Dict + Reward Signals:** Structured observation tensor dict: `fluoroscopy_frame` (num_envs, H, W), `positions` (num_envs, num_points, 3), `orientations` (num_envs, num_points, 4), `velocities` (num_envs, num_points, 3), `tip_contact_force` (num_envs, 3), `tip_position` (num_envs, 3). Reward components: distance-to-target, wall contact penalty (force magnitude), procedure time penalty, buckling detection (segment compression beyond threshold).

---

## SLIDE: Ultrasound Sprint 1 — Isaac Lab Sensor Module

**Speaker notes:**

**UltrasoundSensor Wrapper:** Create an `UltrasoundSensor` class that wraps `SlangUltrasoundRenderer` with an Isaac Lab-compatible interface. Accept probe pose from the USD scene (position + orientation of a virtual probe mesh), render B-mode via the Slang forward kernel, post-process via the GPU pipeline, and return the result as an observation tensor.

**CT-to-Volume Pipeline:** Formalize the DICOM-to-volume preprocessing as a one-time pipeline step. DICOM Hounsfield units are mapped to acoustic properties (impedance, attenuation, scattering) via tissue-specific lookup tables. Output is cached `.npy` files for fast loading at training time. This replaces the current manual notebook-style scripts.

**Observation Dict Integration:** Wire `ultrasound_frame` (B, H, W), `probe_position` (B, 3), and `probe_rotation` (B, 3, 3) into Isaac Lab's structured observation dictionary.

---

## SLIDE: Ultrasound Sprint 2 — Multi-Environment + Performance

**Speaker notes:**

**Batched Rendering:** Extend the Slang kernel dispatch dimensions to `(num_elements, num_elevational, num_envs)`. The scattering volume and triangle geometry are shared across environments — only the probe pose varies per environment. This avoids duplicating the 200+ MB volume data.

**GPU Memory Pipeline (Zero-Copy):** Eliminate the GPU-to-CPU-to-GPU roundtrip between Slang rendering and PyTorch post-processing. Currently, the Slang kernel outputs are transferred to numpy on CPU, then loaded back to GPU as PyTorch tensors. Fix via CUDA pointer sharing or DLPack interop — both Slang and PyTorch support CUDA device pointers. This saves approximately 4 GB/s of bandwidth at 4096 environments.

**Pre-Computed Scan Conversion Grid:** The polar-to-Cartesian coordinate mapping is probe-geometry-dependent but constant across frames. Cache the `grid_sample` coordinates per probe type, eliminating the `atan2`/`hypot`/normalization computation per frame. Estimated 30% reduction in scan conversion time.

---

## SLIDE: Shared Solver Optimizations

**Speaker notes:**

These optimizations benefit both sensor pipelines because both use the XPBD catheter solver:

**CUDA Graph Capture:** Each XPBD substep consists of approximately 15 sequential `wp.launch()` calls. Each launch incurs about 5 microseconds of CPU-side overhead for argument marshaling and kernel dispatch. At 2 substeps per frame, that is roughly 150 microseconds of launch overhead per frame. Warp supports CUDA graph capture via `wp.capture_begin()` / `wp.capture_end()` / `wp.capture_launch()`. All argument shapes are fixed after initialization (num_points and num_edges are constant), so graph replay is valid. Expected impact: reduce per-substep CPU overhead from approximately 75 microseconds to approximately 5 microseconds (single graph launch). Most significant at high environment counts where launch overhead scales linearly.

**Kernel Fusion — 4 fusible pairs:**
1. `_xr_predict_pos` + `_xr_predict_rot` — same thread count (num_points), no data dependency
2. `_xr_integrate_pos` + `_xr_integrate_rot` — same thread count, no dependency
3. Two `_xr_zero_v3` calls for pos_corrections and rot_corrections — same thread count
4. `_xr_compute_jacobians` + `_xr_assemble_jmjt` — highest value fusion; avoids writing and re-reading `num_edges * 36` floats of Jacobian data to global memory

Expected impact: reduce kernel launch count from 15 to 11 per substep. Jacobian+JMJT fusion reduces global memory traffic by approximately 30% for the constraint phase.

**Multi-Env Batching:** Leading environment dimension on all workspace arrays. Block-Thomas parallelism across rods (each environment is an independent block-tridiagonal system). Linear throughput scaling with environment count until GPU occupancy saturates.

**Thomas Solver Optimization:** The block-Thomas kernel currently runs as a single thread. Opportunities: warp-level parallelism within the 6x6 block operations using shuffle instructions, register pressure reduction via shared memory for temporaries, and the split-Thomas variant that decomposes the sweep into two independent half-sweeps from each end (available in Newton's solver as the `split_thomas` backend).

**Memory Aliasing:** Workspace arrays that are never live simultaneously can share storage. For example, `predicted_positions` and `pos_corrections` are used in different substep phases. `rhs` and `delta_lambda` are the same size and output overwrites input. Aliasing reduces per-environment workspace from approximately 50 KB to approximately 35 KB.

---

# PART 3: FUTURE SENSORS

---

## SLIDE: Endoscopy / RGB Camera

**Speaker notes:**

Leverage Omniverse RTX renderer — mount a virtual camera on the catheter tip or a separate endoscope instrument. Isaac Lab already provides `CameraCfg` and `Camera` sensor classes, so the integration path is straightforward. The main technical work is authoring realistic vessel interior USD assets with appropriate materials — wet tissue requires subsurface scattering, specular reflections from fluid, and realistic endoscopic lighting (ring light or fiber-optic illumination). Additional challenges: barrel distortion and vignetting to match real endoscope optics, narrow-band imaging simulation for tissue classification. Not natively differentiable, but depth maps and segmentation masks from RTX are available for reward computation. Medium priority — not required for initial catheter navigation RL but valuable for downstream procedures.

---

## SLIDE: Cone-Beam CT (CBCT)

**Speaker notes:**

Extend the fluoroscopy GPU ray-caster to perform a full rotational sweep — 200 to 400 projections over 180 to 240 degrees — then reconstruct via the Feldkamp-Davis-Kress (FDK) algorithm. The DiffDRR+Slang stack already computes single projections; CBCT is a batched version. FDK reconstruction is well-suited to GPU parallelization (one thread per voxel). Target: under 1 second for a 256-cubed volume from 200 projections. Key challenges: metal artifact reduction (the catheter creates streak artifacts in reconstructed volumes), truncation and scatter correction. Differentiable through the existing Slang differentiable DRR path — gradients would flow from reconstruction loss back through projections to catheter positions. Low-medium priority — CBCT is used intermittently during procedures, not every frame.

---

## SLIDE: Force / Torque Sensing

**Speaker notes:**

This is not a visual sensor — it is derived directly from the XPBD collision solver output. Contact forces are already computed during the position-level projection step: SDF gradient times penetration depth times contact stiffness. The `RodSolver` already tracks contact points and normals via `contact_normals` and `contact_depths` arrays populated by `solve_mesh_collision_kernel`. Integration path: expose `tip_contact_force` as a `(num_envs, 3)` tensor in the observation dict. Add `wall_contact_force_magnitude` as a reward signal component. Calibration challenge: matching contact stiffness to real catheter-vessel interaction forces (typically 0.1-2.0 N for guidewire, up to 5 N for microcatheter). Differentiable via Warp autodiff through the contact solver. High priority — force sensing is a key reward signal for safe navigation policies.

---

## SLIDE: Intravascular Ultrasound (IVUS)

**Speaker notes:**

Adapt the existing Slang ultrasound renderer with a radial transducer model: 360-degree rotational scan from a point source on the catheter tip. Replace the linear/curvilinear/phased probe geometry with a rotating single-element model. The core ray-marching and scattering physics are identical to external ultrasound — `applyAttenuation`, `getScatteringValue`, BVH traversal, multi-bounce specular all carry over. The main changes are probe geometry (radial fan) and scan conversion (polar-to-Cartesian with 360-degree sector). The IVUS transducer pose tracks the catheter tip position from the rod solver output. Technical challenges: IVUS operates at 20-60 MHz versus 2-10 MHz for external ultrasound, requiring finer step sizes and higher scattering texture resolution. NURD artifacts (non-uniform rotational distortion). Vessel wall layering (intima/media/adventitia) at sub-millimeter resolution requires high-resolution meshes not yet available in the CT preprocessing pipeline.

---

## SLIDE: Hemodynamic Pressure and Flow

**Speaker notes:**

Requires coupling with a 1D or reduced-order hemodynamic solver — for example, a lumped-parameter Windkessel model or 1D Navier-Stokes along the vessel centerline. The catheter's presence in the vessel modifies the effective cross-section, altering pressure drop. Use case: fractional flow reserve (FFR) measurement and catheter-based flow assessment. Integration is complex — requires a hemodynamic solver module (not currently in scope), vessel centerline extraction from CT, and cross-section area computation at catheter contact points. Cardiac cycle boundary conditions and peripheral resistance modeling add physiological complexity. Potentially differentiable if using a differentiable 1D fluid solver (JAX-based, for example). Low priority — represents significant new solver integration for a specific clinical use case.

---

## SLIDE: Gap Analysis — X-Ray Critical Gaps

**Speaker notes:**

Five items blocking RL training:

1. **512-4096 parallel environments at 60 Hz** — XPBDRodSolver is single-env only. Resolution: Sprint 2, multi-env batching with block-Thomas parallelism across rods.

2. **Vessel collision (rod vs SDF/mesh)** — XPBD solver has floor collision only. The production RodSolver has mesh BVH collision. Resolution: Sprint 2, port BVH collision path into XPBD substep.

3. **Policy control input (push/pull/rotate)** — No control API on XPBD solver. The workspace arrays `forces` and `torques` exist internally but have no public setter. Resolution: Sprint 1, `apply_proximal_control()` method.

4. **Batched GPU state extraction** — No structured observation pipeline. Each solver exposes raw tensors but there is no unified observation dict. Resolution: Sprint 2.

5. **Physically-correct fluoroscopy** — Current compositing is an opaque bright polyline (Level 0). Resolution: Sprint 1, Beer-Lambert compositing.

---

## SLIDE: Gap Analysis — Ultrasound Critical Gaps

**Speaker notes:**

Five items:

1. **Isaac Lab sensor wrapper** — No integration module exists. The Slang renderer is standalone. Resolution: Sprint 1, create `UltrasoundSensor` wrapper class.

2. **Multi-environment batching** — Single render call per pose. Resolution: Sprint 2, extend Slang dispatch to `(num_elements, num_elevational, num_envs)`.

3. **GPU memory pipeline (no CPU roundtrip)** — Current path is Slang output to numpy on CPU, then reload to PyTorch on GPU. Resolution: Sprint 2, CUDA pointer sharing or DLPack interop.

4. **Batched observation dict** — No structured observation. Resolution: Sprint 1.

5. **CT preprocessing pipeline** — Manual notebook-style scripts. Resolution: Sprint 1, formalize as a one-time pipeline.

---

## SLIDE: Dependencies and Risks

**Speaker notes:**

**X-Ray risks:**
- Block-Thomas single-threaded bottleneck: Medium severity. The multi-env architecture parallelizes across rods, so this is a within-rod concern. For n < 100 segments, the O(n) sequential sweep completes in microseconds — acceptable.
- Beer-Lambert insufficient for sim-to-real transfer: Medium. Mitigated by domain randomization on noise, scatter, and dose parameters. Validation against real fluoroscopy images from clinical partner data.
- CUDA graph capture and dynamic shapes: Low. All shapes are fixed after initialization. Re-capture only triggered on configuration change.
- SDF resolution vs collision fidelity: Medium. Target 0.25-0.35 mm voxel size per XCath specification. Narrow-band SDF to reduce memory footprint.

**Ultrasound risks:**
- slangpy version compatibility: Medium. Slang is under active development. Pin to slangpy >= 0.40.0 and test against nightly builds. Breaking API changes have occurred historically.
- GPU-to-CPU-to-GPU transfer bottleneck: High severity. This is the single largest performance risk for multi-env scaling. Sprint 2 zero-copy pipeline is critical path.
- Phase 2 not differentiable: Medium. Phase 1 is sufficient for gradient-based probe pose optimization. Phase 2 contributes boundary detail but is not required for the optimization loop.
- Scattering volume memory at 256-cubed+: Medium. Each volume is 64 MB at float32. Use 128-cubed (8 MB) for training, 256-cubed for validation. Shared across environments to avoid duplication.
- Ray-march accuracy vs step size: Medium. Default 0.15-0.5 mm step size is adequate for 3-7 MHz diagnostic frequencies. Higher frequencies (IVUS at 20+ MHz) would require finer steps.

---

## SLIDE: Material Model Reference

**Speaker notes (for Q&A):**

The ultrasound renderer uses 8 floats per material:

1. `impedance` — Acoustic impedance in MRayl (megaRayleigh). Water: 1.48, soft tissue: 1.6-1.7, bone: 7.38, air: 0.0004.
2. `speed_of_sound` — in m/s. Water: 1480, soft tissue: 1540, bone: 3500.
3. `attenuation` — in dB/(cm*MHz). Soft tissue: 0.5-1.0, bone: 10-20, blood: 0.2.
4. `mu0` — Density threshold for scattering gating. Voxels with density <= mu0 produce scatter.
5. `sigma` — Base scattering coefficient. Controls speckle intensity.
6. `specularity` — Exponent for specular intensity falloff. Higher values produce sharper reflections.
7. `scattering_freq_exponent` (`n`) — Controls frequency dependence of scattering: `sigma_eff = sigma * (f/f_ref)^n`.
8. `scattering_ref_freq` (`f_ref`) — Reference frequency in MHz for frequency-dependent scattering.

Per-triangle material IDs map each mesh face to a row in this table, enabling multi-tissue simulation from a single segmented mesh.

---

## SLIDE: Hemodynamic Pressure & Flow Sensing

**Speaker notes:**

This is a non-imaging sensor that measures intravascular pressure and flow — the quantities that clinicians use to determine whether a coronary stenosis is hemodynamically significant. The key clinical metric is fractional flow reserve (FFR): the ratio of distal-to-proximal pressure across a stenosis during maximal hyperemia. An FFR below 0.80 indicates a functionally significant lesion warranting intervention. Pressure wire readings and catheter-based flow assessment (e.g., instantaneous wave-free ratio, iFR) are standard-of-care measurements performed during diagnostic catheterization.

**Why this matters for simulation:** A navigation policy that can position a pressure wire correctly, hold it stable during measurement, and interpret the hemodynamic response would cover a significant portion of the diagnostic catheterization workflow. Pressure is also a natural reward signal — policies can be penalized for creating excessive pressure drops (vessel occlusion by the catheter itself).

**Rendering approach — this is a computed sensor, not a visual one:** It requires coupling with a 1D or reduced-order hemodynamic solver. Two candidate architectures:

1. **Lumped-parameter Windkessel model:** The vascular tree is decomposed into resistive-capacitive-inductive (RCL) circuit elements. Each vessel segment has a resistance proportional to `8 * mu * L / (pi * r^4)` (Poiseuille flow), a compliance (capacitance) from vessel wall elasticity, and an inertance (inductance) from blood mass. The catheter's presence modifies the effective radius `r` at each contact point, increasing resistance. This is computationally cheap — a system of ODEs solvable at sub-millisecond timescales — but sacrifices spatial resolution along the vessel.

2. **1D Navier-Stokes along the vessel centerline:** The governing equations are cross-section-averaged continuity and momentum equations: `dA/dt + d(AU)/dx = 0` and `dU/dt + U*dU/dx + (1/rho)*dP/dx = -(8*pi*mu*U)/(rho*A) + f_ext`, where `A(x,t)` is the cross-sectional area, `U(x,t)` is the mean velocity, `P(x,t)` is pressure, and `rho`, `mu` are blood density and viscosity. The tube law `P = P_ext + beta*(sqrt(A) - sqrt(A_0))` with `beta = (sqrt(pi)*E*h)/(1 - nu^2)*A_0` closes the system, where `E` is the vessel wall Young's modulus, `h` is wall thickness, and `A_0` is the unstressed area. This gives spatial pressure and flow profiles along the vessel at the cost of solving a hyperbolic PDE system.

**Catheter-fluid coupling:** The catheter's physical presence reduces the effective lumen area at each segment where it contacts or passes through the vessel. At a given axial position `x`, the effective area becomes `A_eff(x) = A_vessel(x) - A_catheter(x)`, where `A_catheter` is the catheter's cross-sectional area (pi * r_catheter^2). This modified area feeds back into both the Poiseuille resistance (Windkessel) or the 1D momentum equation. The coupling is bidirectional in principle — fluid drag acts on the catheter — but for initial implementation, one-way coupling (catheter position affects flow, flow does not affect catheter motion) is sufficient and avoids stability issues.

**Integration path:** Complex. Requires:
- A hemodynamic solver module (neither Windkessel nor 1D NS is currently implemented)
- Vessel centerline extraction from CT segmentation — the centerline defines the 1D computational domain
- Cross-section area computation at each discretization point — derived from the segmented vessel mesh
- Branching topology — the coronary tree has bifurcations; the solver needs a graph structure, not a single line
- Boundary conditions: inlet flow waveform from the cardiac cycle (typically a time-varying flow rate at the aortic root), outlet Windkessel models representing the downstream microvascular beds (three-element Windkessel: characteristic impedance, peripheral resistance, total arterial compliance)

**Real-time performance for RL:** The Windkessel model is fast — a 50-segment coronary tree is a 150-variable ODE system solvable in microseconds with explicit Euler or RK4. The 1D Navier-Stokes is more expensive but tractable: a 500-cell finite-volume discretization with CFL-limited time stepping runs at sub-millisecond per cardiac cycle on GPU. Both are compatible with the 60 Hz real-time target.

**Observation tensor:** `pressure_proximal` (num_envs, 1), `pressure_distal` (num_envs, 1), `ffr` (num_envs, 1) computed as `pressure_distal / pressure_proximal` during hyperemia, `flow_velocity` (num_envs, 1) at the catheter tip. Optionally: full pressure profile `pressure_field` (num_envs, num_centerline_points) for visualization.

**Differentiability:** Possible if the hemodynamic solver is implemented in a differentiable framework. A JAX-based 1D solver would provide automatic differentiation of pressure with respect to catheter position, vessel geometry, and boundary conditions. Alternatively, Warp's autodiff could handle a Warp-native solver. The gradient `dFFR/d_catheter_position` is clinically meaningful — it tells the policy how small position changes affect the measurement, enabling precise positioning.

**Key technical challenges:**
- Catheter-flow coupling stability: the catheter can nearly occlude small vessels, creating stiff resistance terms. Implicit time integration or regularization of the area term (floor at 5-10% of vessel area) prevents numerical blowup.
- Physiological boundary conditions: the cardiac cycle produces pulsatile flow with a 60-100 bpm heart rate. The solver must run at temporal resolution sufficient to capture the diastolic pressure nadir (where FFR is typically measured). At 120 Hz physics and ~1000 Hz hemodynamic solver internal stepping, this is achievable.
- Hyperemia simulation: FFR is measured during pharmacologically-induced maximal vasodilation (adenosine). This reduces peripheral resistance by 3-5x. The policy needs to learn to trigger or wait for hyperemia — this is a state transition in the environment.
- Validation: comparing simulated FFR values against published in-vivo data for known stenosis geometries (e.g., the FAME trial benchmarks). Target accuracy: FFR within ±0.05 of ground truth for 70-90% area stenoses.

**Priority:** Low. FFR-specific training is a niche clinical application. The solver integration effort is significant — estimated 3-4 weeks for a Windkessel model, 6-8 weeks for a full 1D NS solver. However, the observation tensor is simple (scalar pressure values), and the coupling to the catheter solver is mechanically straightforward (just area reduction at contact points). If the clinical partner prioritizes FFR training, the Windkessel path provides a fast minimum viable implementation.
