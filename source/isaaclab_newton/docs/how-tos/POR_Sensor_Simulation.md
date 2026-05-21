# Plan of Record: Sensor Simulation & Catheter Physics

## X-Ray–Guided Robotic Catheter Interventional System — Isaac for Healthcare

**Document Owner:** Simulation Engineering  
**Last Updated:** 2026-04-09  
**Status:** Draft  
**Scope:** Simulation Environment (Architecture Section 2) — Fluoroscopy Renderer, Newton Catheter Physics, Unified Sim Loop  

---

## 1. Current State

### 1.1 Catheter Physics Solvers

Three solver implementations exist in `isaaclab_newton/solvers/`:

#### RodSolver (Production)

| Property | Value |
|---|---|
| Source | `rod_solver.py` (1,122 lines) + `rod_kernels.py` (1,249 lines) |
| Algorithm | XPBD Cosserat rod — Deul et al. direct solver or Gauss-Seidel |
| Constraints | Stretch, bend/twist (Darboux), shear |
| Multi-env | Supported (`num_envs >= 1`) |
| Collisions | Ground plane, mesh BVH, self-collision |
| Friction | Coulomb, viscous, static/dynamic |
| Tip shaping | Via `RodTipConfig` rest Darboux vectors (J-tip, Simmons curves) |
| Control hooks | `set_external_force_callback` |
| Observability | `positions`, `orientations`, `velocities`, `get_energy()` |

#### XPBDRodSolver (Self-Contained)

| Property | Value |
|---|---|
| Source | `xpbd_rod_solver.py` (1,168 lines) |
| Algorithm | Block-Thomas O(n) direct solve on 6×6 block-tridiagonal JMJT |
| Constraints | Stretch + Darboux only (no shear) |
| Multi-env | Single environment only |
| Collisions | Floor plane only |
| Friction | None |
| Tip shaping | Not implemented (`rest_darboux` fixed at zero) |
| Control hooks | None |
| Dependency | Zero external dependency — all Warp kernels embedded |

#### NewtonXPBDRodSolver (External Bridge)

| Property | Value |
|---|---|
| Source | `newton_xpbd_rod_wrapper.py` (242 lines) |
| Algorithm | Delegates to Newton `SolverXPBDRod` (PR #1981) |
| Multi-env | Single environment only |
| Backends | `block_thomas`, `split_thomas`, `block_jacobi`, `banded_cholesky` |
| Dependency | Requires external `newton` package with XPBD rod support |

### 1.2 Fluoroscopy Rendering

Two compositing paths exist:

**3D Isaac Lab compositing** (`visualize_rod_fluoro_isaaclab.py`):
DRR PNG loaded as a USD textured quad backdrop; catheter rendered as 3D capsule markers via `VisualizationMarkers`; composited by the Omniverse renderer's depth ordering.

**2D OpenCV compositing** (`visualize_rod_fluoroscopy.py`):
Pre-generated DRR PNGs loaded as backgrounds; C-arm pinhole projection maps 3D rod positions to 2D pixel coordinates; catheter drawn as an opaque bright polyline with Gaussian glow overlay. Supports AP, LAO 30, Lateral, RAO 30 views with interactive C-arm angle control and DICOM ingestion.

**Current compositing limitation:** The catheter is rendered as a bright opaque line painted on top of the X-ray background. This is physically incorrect — X-ray imaging is transmissive, and a catheter should darken the background, not brighten it. The overlay completely obscures background anatomy, produces hard edges, and cannot model self-crossing darkening or material-dependent opacity.

### 1.3 Configuration System

`RodConfig` dataclass hierarchy (715 lines in `rod_data.py`) covers material properties (Young's modulus, shear modulus, density, damping, per-mode stiffness), geometry (segment count, segment length, per-segment radius, cross-section, tip config), solver parameters (timestep, substeps, iterations, direct vs iterative), friction (method, coefficients, stiction velocity), and collision mesh (path, BVH, contact stiffness/damping).

### 1.4 Test Coverage

| Test | Lines | Scope |
|---|---|---|
| `test_rod_solver.py` | 362 | Init, step, fixed root, gravity, constraints, direct vs GS, cantilever, reset |
| `test_newton_xpbd_rod_optional.py` | 62 | Conditional smoke test (requires Newton) |
| `run_xpbd_rod_smoke.py` | 99 | Headless XPBDRodSolver gravity validation (example, not pytest) |

No automated pytest coverage exists for `XPBDRodSolver`.

---

## 2. Fluoroscopy Renderer Integration

### 2.1 Beer-Lambert Catheter Compositing

**Objective:** Replace the opaque polyline overlay with physically-correct transmission imaging.

**Physics:** X-ray image formation follows Beer-Lambert attenuation. The detector pixel intensity after passing through both anatomy and catheter is:

    I_final(u,v) = I_DRR(u,v) × exp( -Σ μ_i × t_i(u,v) )

where `I_DRR` is the pre-computed anatomy DRR, `μ_i` is the linear attenuation coefficient of segment `i`, and `t_i(u,v)` is the projected thickness of that segment's cylindrical cross-section at detector pixel `(u,v)`.

**Per-pixel projected thickness:** For a cylinder of radius `r`, the chord length at perpendicular distance `d` from the centerline is `t(d) = 2√(r² - d²)` for `|d| < r`, zero otherwise. This produces the characteristic dark-center, soft-edge profile.

**Per-segment material variation:** Attenuation coefficients vary along the rod to match real device construction:

| Region | Segments | μ × 2r | Material |
|---|---|---|---|
| Proximal marker | First 1–2 | 3.0 | Tungsten band |
| Braided shaft | 0–60% | 0.8 | Nitinol braid |
| Transition | 60–85% | 0.4 | Sparse braid + polymer |
| Soft tip | 85–95% | 0.2 | Polymer |
| Tip marker | Last 2–3 | 5.0 | Platinum coil |

**Depth-dependent magnification:** Each segment's pixel-space radius is magnified by the cone-beam geometry: `r_px = r_phys × SID / z_cam / pixel_spacing`, where `z_cam` is the segment's depth in the camera frame.

**Implementation:** Build a per-pixel attenuation map by iterating over segments, computing perpendicular distance from each nearby pixel to the segment line, evaluating the cylinder chord length, and accumulating `μ × t`. Apply as `background × exp(-attenuation_map)`. The computation touches only pixels within `r_max` of the projected centerline — a few thousand pixels per frame.

### 2.2 Imaging Realism Enhancements

**Poisson quantum noise:** Fluoroscopy operates at low photon counts (500–5000 photons/pixel). The catheter reduces the count further, producing elevated mottle behind dense segments. Simulate via `np.random.poisson(composited × dose_level) / dose_level`. The `dose_level` parameter maps directly to tube current (mA) and serves as a domain-randomization parameter.

**Veiling glare / scatter:** X-rays scattered by the patient and detector create a low-frequency haze around dense objects. Approximate as a large-kernel (σ ≈ 15–20 px) Gaussian blur of the blocked-intensity map, re-added at 2–5% amplitude. Produces the subtle bright halo around platinum markers visible on real fluoroscopy.

**Detector blur (PSF):** Apply spatial convolution with a 2D Gaussian (σ ≈ 0.5–1.0 px) to the final image to model the detector's finite point-spread function.

**Beam hardening:** For polychromatic X-ray simulation, sum Beer-Lambert attenuation across discrete energy bins (5–10 bins spanning 20–80 keV) with energy-dependent attenuation coefficients and spectral weights. Produces characteristic cupping artifacts through dense structures.

### 2.3 GPU Ray-Cast Renderer (Future)

The current system loads pre-baked DRR PNGs generated offline by DiffDRR+Slang. For full training pipeline integration, a real-time GPU ray-caster through the CT-derived μ volume is needed. This would:

- Support arbitrary C-arm poses without pre-rendering
- Enable dynamic contrast agent injection (catheter lumen fills with iodine)
- Allow real-time μ-volume updates (e.g., tool insertion modifies the volume)
- Target: single-frame rendering at <5 ms for 512×512 detector at training time

---

## 3. Solver Optimizations

### 3.1 CUDA Graph Capture

**Current state:** Each XPBD substep consists of ~15 sequential `wp.launch()` calls. Each launch incurs CPU-side overhead for argument marshaling and kernel dispatch (~5 μs per launch). At 2 substeps per frame, this is ~150 μs of launch overhead per frame.

**Optimization:** Capture the entire substep sequence as a CUDA graph on the first invocation, then replay the graph on subsequent frames. This eliminates per-launch overhead and reduces CPU-GPU synchronization to a single graph launch per substep.

**Requirements:**
- All kernel argument shapes must be fixed (already the case — `num_points`, `num_edges` are constant after init)
- Warp supports CUDA graph capture via `wp.capture_begin()` / `wp.capture_end()` / `wp.capture_launch()`
- Dynamic values (`dt`, `gravity`) passed as kernel arguments must use Warp constant arrays or be baked into the graph with re-capture on change

**Expected impact:** Reduce per-substep CPU overhead from ~75 μs to ~5 μs (single graph launch). Most significant at high env counts where launch overhead scales with number of graphs.

### 3.2 Kernel Fusion

Several kernel pairs always execute sequentially with identical thread counts and no intervening data dependency:

| Fusible Pair | Current Launches | After Fusion |
|---|---|---|
| `_xr_predict_pos` + `_xr_predict_rot` | 2 | 1 |
| `_xr_integrate_pos` + `_xr_integrate_rot` | 2 | 1 |
| `_xr_zero_v3(pos_corrections)` + `_xr_zero_v3(rot_corrections)` | 2 | 1 |
| `_xr_compute_jacobians` + `_xr_assemble_jmjt` | 2 | 1 |

Each fusion eliminates one kernel launch, one implicit synchronization point, and one global memory round-trip for intermediate data. The Jacobian + JMJT fusion is the highest-value: it avoids writing and re-reading `num_edges × 36` floats of Jacobian data to global memory.

**Expected impact:** Reduce kernel launch count from ~15 to ~11 per substep. Jacobian + JMJT fusion reduces global memory traffic by ~30% for the constraint phase.

### 3.3 Multi-Environment Batching

**Current state:** `XPBDRodSolver` supports `num_envs == 1` only.

**Design:** Extend all workspace arrays with a leading environment dimension. Most kernels (prediction, integration, constraint evaluation, Jacobians, JMJT assembly, corrections) are embarrassingly parallel across environments — add an `env_id` dimension to thread indexing.

The block-Thomas kernel is the critical case. It runs sequentially (forward + backward sweep) within a single rod but is **independent across environments**. The multi-env version launches `dim=num_envs` threads, each solving its own block-tridiagonal system. This is a natural 1D parallelism that maps well to GPU occupancy.

**Memory layout:** Flat arrays with `env_id * per_env_size + local_offset` indexing, or structured arrays with `(num_envs, per_env_elements)` shape. The flat layout avoids warp divergence in the Thomas solver.

**Expected scaling:** Linear throughput with environment count until GPU occupancy saturates. For a 20-segment rod, each environment uses ~50 KB of workspace. At 4096 envs: ~200 MB — well within A6000's 48 GB.

### 3.4 Thomas Solver Optimization

**Current state:** The block-Thomas kernel runs as a single-threaded sequential loop on the GPU, performing 6×6 block operations via four 3×3 sub-block Cholesky solves per elimination step.

**Opportunities:**

- **Warp parallelism within blocks:** Each 6×6 block solve involves six independent 6-vector solves (one per column of the `C` matrix). These can be parallelized within a warp using shuffle instructions or thread-block cooperation.
- **Register pressure reduction:** The current kernel holds ~36 floats per 6×6 block in registers. Pre-loading frequently-accessed sub-blocks and using shared memory for temporaries would reduce register spilling.
- **Split-Thomas variant:** Decompose the forward sweep into two independent half-sweeps (one from each end), meeting in the middle. Each half runs on a separate thread, doubling throughput. This is the `split_thomas` backend available in Newton's solver.

### 3.5 Memory Optimization

The workspace allocates several arrays that could be aliased (reused) since they're never live simultaneously:

| Array | Size | Can Alias With |
|---|---|---|
| `predicted_positions` | `N × vec3` | `pos_corrections` (used in different phases) |
| `rhs` | `E × 6` | `delta_lambda` (output overwrites, same size) |
| `c_blocks` | `E × 36` | `offdiag_blocks` (consumed before `c_blocks` written) |

Aliasing would reduce per-environment workspace from ~50 KB to ~35 KB, enabling higher env counts within the same GPU memory budget.

---

## 4. Feature Integration Roadmap

### 4.1 Sprint 1 — Close the Simulation Loop (Weeks 1–2)

#### 4.1.1 Beer-Lambert Compositing

Replace `draw_rod_overlay()` in `visualize_rod_fluoroscopy.py` with `composite_catheter_beer_lambert()`. Implement per-segment attenuation, cylinder chord-length projection, cone-beam radius magnification, and multiplicative darkening.

**Deliverable:** Catheter appears as a dark band on the DRR with smooth edges, anatomy visible through the device, and additive darkening at self-crossings.

#### 4.1.2 Proximal Kinematic Control API

Add to `XPBDRodSolver`:

```python
def apply_proximal_control(self, push_velocity: float, rotate_velocity: float) -> None:
```

This modifies the root particle's position (translate along insertion axis) and orientation (apply torsion) at the start of each substep, before the predict phase. Maps directly to cassette motor simulation (XCath Task N10).

**Deliverable:** Policy can push, pull, and rotate the catheter through a single method call per frame.

#### 4.1.3 XPBD Solver in Fluoroscopy Script

Add `--use-xpbd` flag to `visualize_rod_fluoroscopy.py`. Adapt position extraction from `solver.data.positions[0]` to `solver.positions`. Adjust coordinate frame mapping (solver uses metres, fluoroscopy uses mm).

**Deliverable:** XPBD solver drives the catheter in the 2D fluoroscopy overlay.

### 4.2 Sprint 2 — Multi-Environment + Collision (Weeks 3–4)

#### 4.2.1 Multi-Env XPBDRodSolver

Extend `_Workspace` arrays with environment dimension. Modify all kernels to accept `env_id` via 2D thread indexing (`env, particle` or `env, edge`). Launch block-Thomas with `dim=num_envs`.

**Deliverable:** `XPBDRodSolver(config, num_envs=512)` runs 512 independent rods in parallel on a single GPU.

#### 4.2.2 SDF/Mesh Collision for XPBDRodSolver

Port the collision detection path from `RodSolver` (BVH mesh query + contact response) into the XPBD substep, inserted between the constraint projection and integration phases. Implement as position-level projection: if a predicted particle penetrates the SDF, project it to the surface along the SDF gradient.

**Deliverable:** Catheter interacts with vessel wall geometry. No tunneling at 2–4 substeps.

#### 4.2.3 Observation Dict + Reward Signals

Define a structured observation:

```python
@dataclass
class CatheterObservation:
    fluoroscopy_frame: torch.Tensor    # (num_envs, H, W) or (num_envs, H, W, 3)
    positions: torch.Tensor            # (num_envs, num_points, 3)
    orientations: torch.Tensor         # (num_envs, num_points, 4)
    velocities: torch.Tensor           # (num_envs, num_points, 3)
    tip_contact_force: torch.Tensor    # (num_envs, 3)
    tip_position: torch.Tensor         # (num_envs, 3)
```

Define reward components: distance-to-target, wall contact penalty (contact force magnitude), procedure time penalty, buckling detection (segment compression beyond threshold).

**Deliverable:** Policy receives structured batched observations; reward signals enable RL training.

### 4.3 Sprint 3 — Training Readiness (Weeks 5–6)

#### 4.3.1 Domain Randomization

Implement per-episode randomization of:
- C-arm angles: uniform sampling of LAO/RAO and CRAN/CAUD within clinical ranges
- X-ray dose: `dose_level` parameter randomized between 200–5000 photons/pixel
- Scatter magnitude: 1–5% coefficient
- Catheter stiffness: Young's modulus within ±20% of nominal
- Vessel anatomy: random selection from a library of CT-derived SDFs

**Deliverable:** Each parallel environment sees a different imaging condition and anatomy, improving policy generalization.

#### 4.3.2 Gymnasium Environment Wrapper

Wrap the unified sim loop (Newton step → fluoroscopy render → observation packaging) as a `gymnasium.Env` with:
- `action_space`: `Box(2)` — push velocity + rotate velocity
- `observation_space`: `Dict` containing fluoroscopy frame and catheter state
- `step()`: advance physics, render fluoroscopy, compute reward, check termination
- `reset()`: randomize anatomy, C-arm pose, catheter initial state

**Deliverable:** Standard Gym interface for GR00T-H or any RL/IL framework.

#### 4.3.3 CUDA Graph Integration

Capture the full substep kernel sequence as a CUDA graph. Implement re-capture on configuration change (dt, gravity). Profile end-to-end frame time with and without graph capture at 512 and 4096 environments.

**Deliverable:** Measurable latency reduction; performance benchmark report.

#### 4.3.4 Automated Test Coverage

Add `test_xpbd_rod_solver.py` covering:
- Gravity drop (tip descends)
- Constraint preservation (segment lengths maintained)
- Floor collision (particles stay above floor_z)
- Multi-env consistency (all envs produce identical results from identical initial conditions)
- Proximal control (push advances tip position)
- Performance regression (assert >1000 fps for 20-segment single-env on A6000)

**Deliverable:** CI-ready pytest suite for the XPBD solver.

---

## 5. Gap Analysis vs XCath Requirements

### 5.1 Critical Gaps (Blocking RL Training)

| XCath Requirement | Current State | Resolution Sprint |
|---|---|---|
| 512–4096 parallel envs at 60 Hz | `XPBDRodSolver`: single env only | Sprint 2 |
| Vessel collision (rod vs SDF/mesh) | XPBD: floor only; `RodSolver`: mesh BVH | Sprint 2 |
| Policy control input (push/pull/rotate) | No control API on XPBD solver | Sprint 1 |
| Batched GPU state extraction | No structured observation pipeline | Sprint 2 |
| Physically-correct fluoroscopy | Opaque polyline (Level 0) | Sprint 1 |

### 5.2 Medium Gaps (Degraded Fidelity)

| XCath Requirement | Current State | Resolution |
|---|---|---|
| Coaxial multi-instrument (GW inside MC inside IC) | Single rod only | Future: sliding joint constraints between concentric rods |
| Hydroelastic vessel contacts | Rigid collisions only | Future: requires MuJoCo-Warp solver integration |
| Variable stiffness along rod | Uniform material; per-segment radius supported | Sprint 3: per-segment Young's modulus in `_xr_prepare_compliance` |
| Tip shaping on XPBDRodSolver | `rest_darboux` fixed at zero | Sprint 2: apply `RodTipConfig` to `rest_darboux` array at init |
| Friction on XPBDRodSolver | None | Sprint 2: post-collision friction impulse kernel |
| GPU fluoroscopy renderer | Pre-baked DRR PNGs | Future: Warp ray-cast through μ volume |

### 5.3 Lower Priority Gaps

| XCath Requirement | Current State | Notes |
|---|---|---|
| Blood flow drag | Not modeled | Phase D; acceptable for initial training |
| Anisotropic bending stiffness | Isotropic only | Braided catheter approximation; low impact on navigation behavior |
| Dynamic friction coefficients | Static config | Hydrophilic coating wear model; future enhancement |
| Per-contact-pair friction | Global friction config | Gripper vs vessel differentiation; future enhancement |
| Differentiable simulation | Warp kernels are differentiable in principle | Parameter calibration pipeline not built |

---

## 6. Performance Targets

| Metric | Target | Current | Notes |
|---|---|---|---|
| Single-env physics FPS | >1,000 Hz | ~1,300 Hz (20 segments, A6000) | Achieved |
| Single-env compositing | <2 ms/frame | <1 ms (polyline); ~2 ms (Beer-Lambert estimate) | Beer-Lambert within budget |
| 512-env physics FPS | >60 Hz | N/A (single env only) | Sprint 2 target |
| 4096-env physics FPS | >60 Hz | N/A | Sprint 3 target with CUDA graphs |
| Fluoroscopy frame render | <5 ms at 512×512 | ~0.5 ms (PNG load + overlay) | Acceptable for pre-baked; GPU ray-cast target for future |
| GPU memory at 4096 envs | <8 GB | ~200 MB estimated (physics only) | Well within A6000 budget |
| End-to-end frame (physics + render + obs) | <16 ms (60 Hz) | N/A | Sprint 3 profiling target |

---

## 7. Dependencies and Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Block-Thomas single-threaded bottleneck at high env counts | Medium | Multi-env parallelism is across rods (embarrassingly parallel); bottleneck is within-rod sequential sweep, which is O(n_segments) — acceptable for n < 100 |
| Beer-Lambert compositing insufficient for sim-to-real | Medium | Domain randomization on noise, scatter, dose; validate against real fluoroscopy images from XCath |
| CUDA graph capture incompatible with dynamic shapes | Low | All shapes are fixed after init; re-capture only on config change |
| SDF resolution vs collision fidelity trade-off | Medium | Target 0.25–0.35 mm voxel size per XCath spec; narrow-band SDF to reduce memory |
| Warp JIT compilation latency on first run | Low | Kernel cache persists across runs; first-run compilation is one-time (~2 s) |

---

## Appendix A: File Inventory

| File | Lines | Purpose |
|---|---|---|
| `solvers/rod_data.py` | 715 | Config dataclasses + runtime state (`RodConfig`, `RodData`) |
| `solvers/rod_solver.py` | 1,122 | Production XPBD Cosserat rod solver |
| `solvers/rod_kernels.py` | 1,249 | Warp kernels for `RodSolver` |
| `solvers/xpbd_rod_solver.py` | 1,168 | Self-contained XPBD solver (all kernels embedded) |
| `solvers/newton_xpbd_rod_wrapper.py` | 242 | Bridge to external Newton `SolverXPBDRod` |
| `solvers/__init__.py` | 54 | Package exports |
| `examples/visualize_rod_fluoroscopy.py` | 800 | 2D OpenCV fluoroscopy overlay |
| `examples/visualize_rod_fluoro_isaaclab.py` | 420 | 3D Isaac Lab fluoroscopy backdrop |
| `examples/visualize_rod_simple.py` | 322 | Simple Isaac Lab debug visualization |
| `examples/run_xpbd_rod_smoke.py` | 99 | Headless XPBD solver smoke test |
| `test/solvers/test_rod_solver.py` | 362 | RodSolver pytest suite |
| `test/solvers/test_newton_xpbd_rod_optional.py` | 62 | Optional Newton wrapper test |
| `docs/BYOS_Tutorial.md` | 1,402 | Bring-your-own-solver tutorial |

## Appendix B: Kernel Inventory (XPBDRodSolver)

| Kernel | Type | Parallelism | Purpose |
|---|---|---|---|
| `_xr_predict_pos` | `@wp.kernel` | Per particle | Explicit Euler position prediction with gravity + damping |
| `_xr_predict_rot` | `@wp.kernel` | Per particle | Quaternion rotation prediction |
| `_xr_prepare_compliance` | `@wp.kernel` | Per edge | Compute XPBD compliance from material stiffness |
| `_xr_update_constraints` | `@wp.kernel` | Per edge | Evaluate stretch + Darboux constraint errors |
| `_xr_compute_jacobians` | `@wp.kernel` | Per edge | 6×6 position + rotation Jacobians |
| `_xr_compute_inv_inertia` | `@wp.kernel` | Per particle | World-frame inverse inertia from quaternion |
| `_xr_assemble_jmjt` | `@wp.kernel` | Per edge | Block-tridiagonal JMJT diagonal + off-diagonal blocks |
| `_xr_build_rhs` | `@wp.kernel` | Per DOF | Right-hand side: `-c - α·Σλ` |
| `_xr_block_thomas` | `@wp.kernel` | Single thread | O(n) block-Thomas forward + backward sweep |
| `_xr_compute_corrections` | `@wp.kernel` | Per edge | J^T·Δλ position + rotation corrections with atomic accumulation |
| `_xr_apply_corrections` | `@wp.kernel` | Per particle | Add corrections to predicted state |
| `_xr_integrate_pos` | `@wp.kernel` | Per particle | Velocity derivation + position commit |
| `_xr_integrate_rot` | `@wp.kernel` | Per particle | Angular velocity derivation + orientation commit |
| `_xr_floor_collision` | `@wp.kernel` | Per particle | Position-level floor projection |
| `_xr_zero_f` | `@wp.kernel` | Per element | Zero float array |
| `_xr_zero_v3` | `@wp.kernel` | Per element | Zero vec3 array |

## Appendix C: Helper Function Inventory (XPBDRodSolver)

| Function | Category | Purpose |
|---|---|---|
| `_xr_qmul`, `_xr_qconj`, `_xr_qnorm`, `_xr_qrot`, `_xr_qcorr` | Quaternion | Hamilton product, conjugate, normalize, rotate vector, apply angular correction |
| `_xr_m33_add`, `_xr_m33_sub`, `_xr_m33_mul`, `_xr_m33v`, `_xr_m33_t` | 3×3 matrix | Element-wise add/sub, multiply, matrix-vector, transpose |
| `_xr_chol`, `_xr_solvL`, `_xr_solvU`, `_xr_cholsol` | 3×3 linear algebra | Cholesky factorization, forward/backward substitution, full solve |
| `_xr_invI_v` | Inertia | Multiply flattened 3×3 inverse inertia by vector |
| `_xr_load_blk`, `_xr_store_blk` | 6×6 block I/O | Load/store 6×6 block as four 3×3 sub-matrices |
| `_xr_load_v6`, `_xr_store_v6` | 6-vector I/O | Load/store 6-vector as two vec3 |
| `_xr_blk_col`, `_xr_blk_set_col` | 6×6 column ops | Get/set a single column of a 6×6 block |
| `_xr_blk_mul`, `_xr_blk_sub`, `_xr_blk_mv`, `_xr_blk_solve` | 6×6 block algebra | Block multiply, subtract, matrix-vector, full 6×6 Cholesky solve |
| `_xr_jidx`, `_xr_bidx` | Indexing | Flat-array index for edge×6×6 Jacobian and block storage |
| `_xr_jdot` | Jacobian | Dot product of Jacobian column with 6-vector (for corrections) |
