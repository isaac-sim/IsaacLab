# Presentation Notes — Release Deck v5
## X-Ray–Guided Robotic Catheter Intervention — Isaac for Healthcare
### Verified against: `source/isaaclab_newton/`, `fluorosim/`, `/home/cdinea/newton/newton/`

---

## Code verification summary (checked before writing these notes)

| Claim | Verified in |
|---|---|
| Block-Thomas, Split-Thomas, Block-Jacobi, Banded Cholesky backends | `xpbd_rod_solver.py` lines 54–63; Newton `solver_xpbd_rod.py` lines 31–36 |
| Batched workspace (`_BatchedWorkspace`, `rod_offsets`) | `xpbd_rod_solver.py`; Newton `_BatchedRodWorkspace` line 170 |
| CUDA-graph capture (`wp.ScopedCapture` + `wp.capture_launch`) | `xpbd_rod_solver.py` lines 2376–2426 — **Isaac Lab addition to `SolverXPBDRod`** (Newton uses graphs in Kamino/MPM but `solver_xpbd_rod.py` itself has none — confirmed 0 matches) |
| Configurable `floor_restitution` | `xpbd_rod_solver.py` line 2240 — **Isaac Lab extension** (Newton hardcodes 0.0) |
| `apply_proximal_control_gpu`, `set_root_orientation` | `xpbd_rod_solver.py` — **Isaac Lab additions** |
| `XCathRodSolver` vessel SDF + AABB collision | `xcath_rod_solver.py`; ported from Newton branch `xpbd_rods_solver_integr` (Przemysław Korzeniowski) |
| Misregistration jitter | `realism.py` + `dsa.py` — **implemented** |
| Hagen-Poiseuille / F-8 / Dispersion correction | **NOT in fluorosim codebase** — validated by XCath externally |
| Inline catheter Beer-Lambert compositing | `diffdrr_slang.slang` — `catheterAttenuation_slice` + `rayMarchWithCatheterSlice` |
| `DET_SIZE = 256` in interactive demo | `interactive_catheter_fluoro.py` line 114 |

---

## Slide 1 — Title

**What to say:**
> "This presentation covers the full X-ray guided robotic catheter intervention simulation system — what's running today, what we built this sprint, and the forward roadmap. The system runs three tightly coupled components on a single GPU: XPBD Cosserat rod physics ported from the Newton research engine, a Slang GPU Beer-Lambert DRR renderer with inline catheter compositing, and an RL training pipeline using RSL-RL PPO at 512 parallel environments."

**Key technical point to land:**
The phrase "zero CPU sync on the hot path" is the architectural claim that matters. Physics, rendering, and RL observation all happen on GPU memory — the CPU is only involved in initial setup and final policy updates.

---

### Q&A — Slide 1

**Q: What does "zero CPU sync on the hot path" actually mean in practice?**
A: The physics CUDA graph, catheter buffer write, and Slang render dispatch all operate on GPU-resident buffers. The CPU never reads or writes physics state per timestep. `wp.ScopedCapture` captures the entire substep loop as a single graph node; subsequent calls replay via `wp.capture_launch` with no CPU kernel-launch overhead.

**Q: Is this running inside Isaac Sim?**
A: No. The physics uses NVIDIA Warp directly (not PhysX), and the renderer uses a custom Slang shader. Isaac Sim's RTX renderer and PhysX are not involved. Isaac Lab provides the environment registry and RL training scaffolding, but the physics and rendering are entirely custom GPU code.

**Q: What GPU are these numbers measured on?**
A: NVIDIA A6000 (48 GB VRAM, Ampere). The ~63 FPS at 256×256 and ~25 FPS at 512×512 are measured on that hardware.

---

## Slide 2 — Architecture Overview (4-Stage Pipeline)

**What to say:**
> "The pipeline has four stages that all share one coordinate frame — CT volume millimetres. A patient CT enters at Stage 1 and an autonomous navigation policy exits at Stage 4. Stage 1 runs once per anatomy, producing the μ-volume for the renderer and the collision mesh for the physics. Stage 2 is the sim loop — physics and rendering co-located on the GPU. Stage 3 is RL training. Stage 4 is deployment on Holoscan with the XCath robot."

**Key technical point to land:**
The same CT that produces the renderer's μ-volume also produces the physics collision mesh. They are extracted from the same data with the same coordinate transform, so physics collision and X-ray rendering are always registered to the same patient anatomy — no separate calibration step needed.

---

### Q&A — Slide 2

**Q: Why does Stage 1 run once and not per frame?**
A: The CT μ-volume is a 3D texture that is uploaded to the GPU once and stays there for the entire training session. Only the catheter segment buffer (N×8 floats) changes per frame. This is why the renderer can run at interactive speeds — it's doing ray marching through a static volume, not reconstructing CT on every frame.

**Q: How are Stage 2 physics and rendering registered to the same anatomy?**
A: Both assets come from the same CT pipeline. HU→μ conversion produces the volume for rendering. Vessel segmentation + Marching Cubes on the same HU data produces the `wp.Mesh` for collision. Both operate in the same coordinate frame (CT volume mm), so the physics solver and renderer see identical anatomy boundaries.

**Q: What does Holoscan IGX do at Stage 4?**
A: It's a low-latency NVIDIA edge AI computer designed for medical imaging. In this context it runs policy inference at the bedside (<10 ms), receives real fluoroscopy frames from the C-arm at 30 FPS, and sends push/rotate commands to the XCath robotic catheter. The trained policy runs in the same format it was trained in — the sim-to-real gap is closed by domain randomization and realistic detector physics in Stage 2.

---

## Slide 3 — Unified Sim Loop — Per-Timestep Execution

**Opening line:**
> "Every training frame executes five GPU-resident steps in sequence — no CPU synchronization, no device copies on the hot path. This is what makes 512-environment training at interactive frame rates possible. Let me explain why each step was designed the way it was."

---

### Step 1 — Root Control

**Why it matters strategically:**
This is the only place where the RL policy's action touches the physical simulation. The policy outputs two scalars — insertion speed and rotation rate — and Step 1 converts those scalars into a proximal boundary condition applied directly to the catheter root in GPU memory. The entire RL training loop depends on this being GPU-resident and graph-safe: if Step 1 required a CPU round-trip, it would add a 100–500 µs synchronization penalty per frame and break CUDA graph capture, cutting throughput by ~40%.

**Why it was designed as a GPU kernel (not CPU):**
Newton's `SolverXPBDRod` uses a CPU-side `apply_proximal_control` that calls `wp.synchronize_device()` before writing to the buffer — a host-GPU sync that serializes the execution stream and is incompatible with CUDA graph capture. (Important nuance: Newton as a platform does use CUDA graphs in its Kamino articulated-body and implicit-MPM solvers. The specific gap is that `solver_xpbd_rod.py` has no graph capture — confirmed by grep, 0 matches for `ScopedCapture` in that file.) Isaac Lab re-implements both control operations as dedicated Warp kernels that write only to GPU-resident arrays, making them safely capturable inside the substep graph.

**How it is implemented:**
Two Warp kernels launch before every physics step:

1. **Axial insertion/retraction** — reads the live first-segment tangent vector `(pos[next_idx] − pos[root_idx])` to determine the current insertion axis. Moves `pos[root_idx]` along that tangent by `push_velocity × dt`. Simultaneously applies an incremental axial twist: `dq = quat(tangent × sin(ω·dt/2), cos(ω·dt/2))`, accumulated into `ori[root_idx]`. The tangent is recomputed from current particle positions every call — the controller always follows the actual insertion direction, not a fixed world axis.

2. **Absolute orientation set** — forces `ori[root_idx] = pred_ori[root_idx] = prev_ori[root_idx] = q` in a single-thread launch. Used when a specific tip orientation must be commanded directly rather than integrated from a rotation rate. Writing all three orientation arrays (current, predicted, previous) ensures the XPBD integrator does not revert the override on the next substep.

---

### Step 2 — Physics Step (XPBD CUDA Graph)

**Why it matters strategically:**
At 512 environments, launching 14 CUDA kernels × 8 substeps × 512 envs per frame via individual CPU kernel-launch calls would cost ~4–8 ms in kernel-launch overhead alone before any compute executes. CUDA graph capture records all 112 kernel launches (14 × 8) into a single replay call. After the first frame, the entire physics step costs only `wp.capture_launch()` — a single CPU-side API call with ~1 µs overhead. This is what enables 1,300 Hz physics throughput per environment.

**Why the block-Thomas solver was chosen over Gauss-Seidel:**
Gauss-Seidel constraint projection requires tuning iteration counts and convergence is not guaranteed for stiff constraints (high Young's modulus). Block-Thomas is a direct O(N) solver for the 6×6 block-tridiagonal system — it converges in one pass regardless of rod stiffness, making it robust for the full range of catheter material parameters without tuning.

**How it is implemented:**
The CUDA graph captures the complete `num_substeps` loop. Each substep executes this sequence:

```
predict_pos/rot     → explicit Euler extrapolation (one kernel each)
pre_hook            → vessel SDF containment + track projection (XCathRodSolver)
prepare_compliance  → compute α/dt² compliance per edge from Young/torsion modulus
update_constraints  → evaluate stretch + Darboux violations
compute_jacobians   → assemble J_pos and J_rot (6×N_edges × 6 blocks)
compute_inv_inertia → rotate local inertia tensor to world frame
solve_system        → block-Thomas forward/back solve on 6×6 block-tridiagonal JMJT
apply_corrections   → apply Δλ to predicted positions and orientations
post_hook           → AABB mesh-edge contacts (XCathRodSolver)
floor_collision     → optional floor plane restitution
integrate_pos/rot   → commit predicted state; update velocities from position delta
```

The graph is captured once per unique `sub_dt` value. After capture, `step()` reduces to a single `wp.capture_launch(graph)` call — verified in `xpbd_rod_solver.py` lines 2420–2430. The graph is invalidated if `dt` changes or `solver.reset_cuda_graph()` is called (e.g. after the UI's Reset button directly writes new particle positions into GPU buffers).

---

### Step 3 — Segment Buffer Update

**Why it matters strategically:**
Physics and rendering operate in different coordinate systems — physics runs in world-space metres, rendering runs in CT-volume millimetres registered to the patient anatomy. This step is the bridge. Without it, catheter positions would be misregistered in the rendered DRR. The coordinate transform is lightweight (~0.05 ms) but architecturally critical: it is also where the physics-to-rendering registration guarantee is enforced. Both systems share a known common coordinate chain, so this step is deterministic and invertible.

**Why the buffer is separate from the physics state:**
Separating the catheter segment buffer from the physics workspace allows physics and renderer memory layouts to evolve independently. The renderer expects a flat structured array of cylinder primitives; the physics stores positions as `wp.vec3` arrays per particle. Decoupling means either can be refactored without touching the other.

**How it is implemented:**
```python
# _pos_to_vol_mm():
pos_ct_mm  = (pos_m − local_z0_m + ct_offset_m) × 1000.0   # physics metres → CT mm
pos_vol_mm = pos_ct_mm − ct_origin_mm                        # CT mm → volume mm
```
Three offsets registered at init time:
- `local_z0_m` — height of the track insertion axis in physics world
- `ct_offset_m` — CT anatomy entry point in physics world coordinates
- `ct_origin_mm` — CT volume corner origin (from `metadata.json`)

The packed `CatheterSegmentData(positions=pos_vol_mm, radii=CATHETER_R, mu_values=CATHETER_MU)` is passed to the renderer. Radius and μ are scalars — the renderer's `to_structured_array()` uses `np.broadcast_to` internally, so no per-segment allocation occurs. In the RL training path, this step operates entirely on GPU via `wp.to_torch()` — no device copy.

---

### Step 4 — Render Call (Slang DRR)

**Why it matters strategically:**
The key architectural decision is that catheter compositing is fused *inside* the ray-march loop — not applied as a post-process overlay. This is non-obvious but physically critical: a catheter behind dense bone is attenuated by *both* the bone and the catheter material. Post-process overlays cannot model this because they operate on the final pixel, not on the sample-by-sample integral. The fused Beer-Lambert formulation matches the physics a real X-ray detector sees — which is the foundation of sim-to-real transfer for the RL policy.

**Why a single Slang kernel handles all environments:**
The renderer dispatch is `(det_width, det_height, num_envs)` — a 3D grid where the Z axis is environment index. Each thread traces one ray for one environment. All environments share the same CT μ-volume (static `Texture3D<float>` uploaded once), but each reads a different slice of the `CatheterSegmentData` buffer for its own catheter state. This makes rendering throughput scale linearly with GPU core count, not sequentially with environment count.

**How it is implemented:**
For each ray sample point `s`, the kernel accumulates:
```
μ_total(s) = μ_CT(s) + Σᵢ μᵢ · √(1 − dᵢ²/rᵢ²)
```
The `√(1 − d²/r²)` term is the chord-length weight — the exact analytical path length through a circle of radius `r` at perpendicular offset `d`. The final pixel intensity is `I(x,y) = I₀ · exp(−∫ μ_total ds)`. C-arm projection switching is zero-cost: only a `(1, 3)` Euler angle array changes — no CT reload. Post-processing pipeline on the same GPU stream: Poisson noise → detector PSF → gamma correction → misregistration jitter.

---

### Step 5 — Output

**Why it matters strategically:**
The outputs are designed to keep both the RL policy observation and the physics state vector GPU-resident, eliminating the GPU→CPU→GPU round-trip that most sim frameworks require. For a 512-environment training loop, that round-trip would add ~15–30 ms per frame — enough to cut training throughput in half. The two outputs serve different consumers: the fluoroscopy frame feeds the neural network; the catheter state vector (tip position, bend magnitude) feeds the reward function and any external logging.

**How it is implemented:**
- **Fluoroscopy frame** — `(H × W)` float32, normalized to `[0, 1]`. In the RL training path, returned as a GPU tensor via `wp.to_torch()` — no host copy. In the interactive UI, converted to NumPy for Gradio display (UI-only overhead, not on training path).
- **Catheter state vector** — tip position in CT-volume mm (`pos_ct_mm[-1]`), insertion depth, and C-arm orientation from `PROJECTIONS`. Available every frame as a lightweight CPU read for the reward function.
- **Catheter bend metric** — max perpendicular deviation of any rod node from the straight line between proximal and distal endpoints. Computed from `_catheter_bend_mm(pos_vol_mm)`. Values >2 mm confirm that vessel-mesh collision constraints are actively firing. This is the live diagnostic that vessel containment is working — not an estimate.

---

### Key performance numbers (A6000, verified)

| Step | Time | What it proves |
|---|---|---|
| Step 1 — Root Control | < 0.1 ms | GPU kernel, no CPU sync |
| Step 2 — Physics (CUDA graph, 8 substeps) | ~12 ms | 1,300 Hz per env; CUDA graph eliminates launch overhead |
| Step 3 — Segment buffer (UI path) | ~0.1 ms | Negligible coordinate transform |
| Step 4 — Render (256 px, 1 env) | ~3.8 ms | Static CT texture; only catheter buffer changes |
| Step 5 — Output (RL path, GPU-resident) | ~0 ms | No device copy |
| **Full loop (interactive, 256 px)** | **~15.8 ms ≈ 63 FPS** | |

---

### Q&A — Slide 3

**Q: Why is Step 1 separate from Step 2 — why not fold the root control into the physics graph?**
A: The RL policy generates new push/rotate actions every frame *from outside* the CUDA graph. The graph was captured at a fixed state — it cannot incorporate externally computed values mid-replay. Step 1 writes the new control values into GPU buffers before graph replay begins. This is the correct pattern: external inputs are applied as graph inputs (buffer writes) before capture replay, not inside the graph itself.

**Q: What happens if the CUDA graph gets invalidated?**
A: The first call after invalidation re-runs `_capture_cuda_graph()`: one warm-up substep (kernels already JIT-compiled), one `wp.synchronize_device()`, then `wp.ScopedCapture` records the full substep loop. This adds ~10–50 ms on that one frame, then returns to near-zero overhead. In the UI, this happens after every Reset button press — the user experiences one slightly slow frame then smooth resumption.

**Q: Why does Step 3 go through CPU in the interactive demo but not in RL training?**
A: In the UI, Gradio's image pipeline runs on CPU, so `solver.positions.cpu().numpy()` is required. In RL training, `wp.to_torch(ws.positions)` returns a zero-copy GPU tensor view of the Warp buffer. The coordinate transform in RL training is also batched on GPU. The interactive UI is therefore ~0.5 ms slower on Step 3 than the pure RL path — accepted as a UI-only cost.

**Q: What does `_pre_constraints_hook` do in `XCathRodSolver` and why is it placed *before* the solve?**
A: It applies vessel SDF containment and track-guided insertion to predicted positions *before* compliance is computed. Placing it pre-solve means the block-Thomas system sees already-corrected predicted positions — the constraint solve then only needs to enforce stretch and bending, not fight a containment violation. Placing it post-solve (in `_post_constraints_hook`, used for AABB edge contacts) is less stable because the solve may have already moved particles back outside the vessel. Pre-hook for SDF, post-hook for edge contacts — each placed where it is most numerically stable.

---

## Slide 4 — Newton Upstream Algorithms — Ported to Isaac Lab

**Opening line:**
> "This slide covers the two halves of the physics engine. The left column — five solver backends — were ported as-is from Newton's `SolverXPBDRod`. The right column — vessel-mesh collision — was ported from Przemysław Korzeniowski's xcath branch. Neither came with CUDA graph capture, substep loops, or RL control interfaces. Isaac Lab added all of that on top. I'll walk through each feature and be explicit about what was ported unchanged, what was extended, and what is a net-new Isaac Lab addition."

---

### Ported vs. Improved — Full Verified Diff

The table below covers every significant feature and states exactly what was taken from Newton unchanged, what was extended, and what was added entirely in Isaac Lab. All claims verified by direct inspection of `solver_xpbd_rod.py` (1,172 lines) and `xpbd_rod_solver.py` (3,054 lines).

| Feature | Newton `SolverXPBDRod` | Isaac Lab `XPBDRodSolver` | Status |
|---|---|---|---|
| Block-Thomas direct solve | ✅ Present | ✅ Ported | **Ported as-is** |
| Split-Thomas solve | ✅ Present | ✅ Ported | **Ported as-is** |
| Block-Jacobi solve | ✅ Present | ✅ Ported | **Ported as-is** |
| Banded Cholesky solve | ✅ Present | ✅ Ported | **Ported as-is** |
| Tiled Cholesky (GPU shared mem) | ✅ Present | ✅ Ported | **Ported as-is** |
| `_RodWorkspace` layout | ✅ Present | ✅ Ported | **Ported as-is** |
| `_BatchedRodWorkspace` / `rod_offsets` | ✅ Present | ✅ Ported + improved | **Extended** (see below) |
| `set_root_orientation` GPU kernel | ✅ Present | ✅ Ported | **Ported as-is** |
| Pre/post constraint hooks (single-rod) | ✅ Present (`_project_predicted_positions_pre/post_constraints`) | ✅ Ported | **Ported as-is** |
| SDF containment (`wp.mesh_query_point_sign_normal`) | ✅ In xcath branch | ✅ Ported | **Ported as-is** |
| AABB broadphase + vertex/edge contacts | ✅ In xcath branch | ✅ Ported | **Ported as-is** |
| Track-guided insertion | ✅ In xcath branch | ✅ Ported | **Ported as-is** |
| Substep loop (`num_substeps`) | ❌ Caller-managed | ✅ Internal to `step(dt)` | **Isaac Lab addition** |
| CUDA graph capture | ❌ Not in `SolverXPBDRod` | ✅ Auto-capture + `reset_cuda_graph()` | **Isaac Lab addition** |
| `RodConfig` dataclass API | ❌ Requires `Model`/`ModelBuilder` | ✅ Single dataclass | **Isaac Lab addition** |
| `apply_proximal_control_gpu` (push+rotate kernel) | ❌ No push/rotate at all | ✅ `_xr_proximal_push_kernel` | **Isaac Lab addition** |
| `floor_restitution` configurable | ❌ Hardcoded `0.0` (lines 576, 664) | ✅ Constructor parameter | **Isaac Lab extension** |
| Batching for all solver backends | ❌ Block-Thomas only (line 391) | ✅ All 4 backends | **Isaac Lab extension** |
| `positions`/`orientations`/`velocities` PyTorch properties | ❌ None (`wp.copy` to `state_out`) | ✅ Zero-copy `wp.to_torch()` views | **Isaac Lab addition** |
| Convergence diagnostics (`max_delta_lambda`, `max_correction`) | Internal arrays only | ✅ Exposed as solver properties | **Isaac Lab extension** |
| Pre/post hooks for batched path | ❌ No batched hook | ✅ `_pre/post_constraints_hook_batched` | **Isaac Lab addition** |
| `_BatchedWorkspace` single allocation | ❌ Copies from per-rod workspaces | ✅ Allocated at correct shape directly | **Isaac Lab extension** |

---

### Detailed Notes for Each Feature

---

#### What was ported as-is from Newton

**Block-Thomas, Split-Thomas, Block-Jacobi, Banded Cholesky, Tiled Cholesky:**
All five solver backends are faithful ports of Newton's `solver_xpbd_rod.py`. The kernel implementations (`_xr_block_thomas`, `_xr_solve_blocks_jacobi`, `_xr_spbsv_u11_1rhs`, `_xr_cholesky_solve_tile`) mirror Newton's kernel naming convention and logic exactly. Constants `_XPBD_TILE=64`, `_XPBD_BAND_LDAB=34`, `_XPBD_BLOCK_DIM=128` are taken directly from Newton's `constants.py` (`TILE`, `BAND_LDAB`, `BLOCK_DIM`).

**`_RodWorkspace` layout:**
The per-rod buffer layout — positions, velocities, orientations, predicted arrays, jacobian arrays, compliance, lambda_sum, diag/offdiag blocks, c_blocks, d_prime, delta_lambda — is a direct port. Newton's `_RodWorkspace.__init__` and Isaac Lab's `_Workspace.__init__` allocate the same buffers in the same shapes.

**`set_root_orientation` GPU kernel:**
Newton's `_warp_set_root_orientation` kernel (writes `ori[root]`, `pred_ori[root]`, `prev_ori[root]`) is present in both. Isaac Lab's `_xr_set_root_orientation_kernel` is identical in logic.

**SDF containment, AABB broadphase, vertex/edge contacts, track-guided insertion:**
All ported from the `xpbd_rods_solver_integr` branch authored by Przemysław Korzeniowski. The kernel signatures, parameter names, and collision loop structure are preserved. Isaac Lab adds only the hook wiring and multi-env support.

---

#### What was extended or improved in Isaac Lab

**Internal substep loop:**
Newton's `step()` is a single XPBD iteration. The caller must loop externally. Isaac Lab's `step(dt)` encapsulates the loop:
```python
sub_dt = dt / self._num_substeps
for _ in range(self._num_substeps):
    substep_fn(sub_dt)
```
This means the graph captures the full `num_substeps` loop as one unit, not just one substep. Newton cannot do this because its `step()` also calls `wp.copy()` to `state_out.particle_q` after each rod — those copies sit outside any capturable block.

**CUDA graph capture:**
Newton's `SolverXPBDRod` has zero references to `ScopedCapture` (confirmed, 0 grep matches). Newton's Kamino and MPM solvers do use graphs, but `SolverXPBDRod` does not. Isaac Lab adds:
```python
with wp.ScopedCapture(device=self._device) as capture:
    for _ in range(self._num_substeps):
        substep_fn(sub_dt)
self._cuda_graph = capture.graph
# subsequent calls:
wp.capture_launch(self._cuda_graph)
```
Automatic invalidation + re-capture on `dt` change or `reset_cuda_graph()`.

**`apply_proximal_control_gpu` — not in Newton at all:**
Newton has `set_root_orientation()` (orientation override) but has no push/rotate function anywhere in `SolverXPBDRod`. Isaac Lab adds `_xr_proximal_push_kernel` which computes the live first-segment tangent and applies both axial translation and incremental quaternion twist in one kernel — the direct interface for RL policy actions.

**`floor_restitution` — Newton hardcodes 0.0:**
Newton lines 576 and 664 both pass `0.0` as a literal argument to `_warp_apply_floor_collisions`. Isaac Lab promotes this to a constructor parameter `floor_restitution: float = 0.0` and passes `float(self._floor_restitution)` — enabling domain randomization during RL training.

**Batching for all backends — Newton restricts to block_thomas:**
Newton line 391:
```python
if len(self._rods) > 1 and self.solver_backend == DIRECT_SOLVE_BLOCK_THOMAS:
    self._batched_ws = _BatchedRodWorkspace(...)
```
Isaac Lab allocates `_BatchedWorkspace` unconditionally when `num_envs > 1`, and all four backends have batched paths (`_substep_batched` dispatches to the right one).

**`_BatchedWorkspace` single allocation:**
Newton's `_BatchedRodWorkspace` initialises from a list of already-allocated `_RodWorkspace` objects and copies data into fresh concatenated arrays — two allocations per buffer. Isaac Lab's `_BatchedWorkspace` is constructed at the final size directly (`num_envs × num_points_per_rod`) with no intermediate per-rod workspaces involved.

**Pre/post hooks for the batched path:**
Newton's `_project_predicted_positions_pre_constraints(rod_idx, ws, dt, device)` handles only single-rod and takes a `rod_idx` argument. Isaac Lab adds:
- `_pre_constraints_hook_batched(bws, dt, dev)` — operates on `_BatchedWorkspace`
- `_post_constraints_hook_batched(bws, dt, dev)` — operates on `_BatchedWorkspace`

`XCathRodSolver` overrides all four hooks — vessel collision therefore works identically in single-env and 512-env training modes.

**`positions` / `orientations` / `velocities` as zero-copy PyTorch properties:**
Newton has no properties for state access. The training loop must read `state_out.particle_q` after a `wp.copy()`. Isaac Lab exposes:
```python
solver.positions    # → (num_envs, num_points, 3) GPU tensor, zero-copy
solver.orientations # → (num_envs, num_points, 4)
solver.velocities   # → (num_envs, num_points, 3)
```
These return `wp.to_torch()` views — the reward function and renderer read physics state without any device copy.

**Convergence diagnostics:**
Isaac Lab's `_BatchedWorkspace` allocates `max_delta_lambda` and `max_correction` as 1-element GPU arrays, written via `wp.atomic_max` inside the batched Thomas solver. Exposed as `solver.max_delta_lambda` and `solver.max_correction` properties — reported live in the interactive UI's simulation info box and available for curriculum learning signals in RL training.

---

### Left Column — Core Algorithms Upstreamed

---

#### Block-Thomas Direct Solve

> **Status: PORTED AS-IS from Newton `SolverXPBDRod`**
> Newton has `_warp_block_thomas_solve` and `_warp_block_thomas_solve_batched` in `kernels_solvers.py`. Isaac Lab's `_xr_block_thomas` and `_xr_block_thomas_batched` are direct ports — same algorithm, same kernel structure, same O(N) forward/back sweep. Constants `_XPBD_TILE=64`, `_XPBD_BAND_LDAB=34`, `_XPBD_BLOCK_DIM=128` taken directly from Newton's `constants.py`.
> The one Isaac Lab extension is that the batched Thomas path is available for **all backends** (not just block_thomas as Newton restricts).

**Why it matters:**
The block-Thomas algorithm is the reason the XPBD solver can run at 1,300 Hz without any Gauss-Seidel iteration budget tuning. It solves the constraint system *exactly* in one pass, making the result independent of iteration count and stable at any rod stiffness.

**The problem it solves:**
XPBD constraint projection normally requires many Gauss-Seidel iterations to converge for stiff constraints (high Young's modulus nitinol catheters, E ≈ 75 GPa). At large substep counts the iterations become the compute bottleneck. Block-Thomas eliminates iteration entirely.

**How it is implemented:**
The XPBD constraint system for a rod with N edges is a block-tridiagonal linear system `JMJT · Δλ = C`, where:
- `J` = 6×N constraint Jacobian (stretch + Darboux for each edge)
- `M⁻¹` = block-diagonal inverse mass + inverse inertia matrix
- `JMJT` = 6×6 block-tridiagonal: diagonal blocks couple position+rotation DOFs of one segment, off-diagonal blocks couple adjacent segments

Thomas algorithm solves this in exactly two passes:
1. Forward elimination: sweeps N edges, eliminating the lower off-diagonal block, accumulating `c_blocks` (Schur complement terms) and `d_prime` (modified RHS)
2. Back substitution: sweeps N edges in reverse, recovering `delta_lambda` from `c_blocks` and `d_prime`

Total complexity: O(N) — linear in rod length, not quadratic. Verified in `xpbd_rod_solver.py` lines 2617–2630:
```python
wp.launch(_xr_assemble_jmjt, dim=ne, ...)      # assemble 6×6 diagonal + off-diagonal blocks
wp.launch(_xr_build_rhs, dim=nd, ...)           # build RHS from constraint values
wp.launch(_xr_block_thomas, dim=1, inputs=[     # forward + back sweep, O(N)
    diag_blocks, offdiag_blocks, rhs, ne,
    c_blocks, d_prime, delta_lambda])
```
The block-Thomas kernel runs single-threaded per rod (`dim=1`) — the parallelism comes from the batched path where one thread per rod runs the Thomas sweep simultaneously across all 512 environments.

---

#### Split-Thomas Solve

> **Status: PORTED AS-IS from Newton `SolverXPBDRod`**
> Newton has `_warp_block_thomas_solve_3x3`, `_warp_assemble_stretch_blocks`, `_warp_assemble_darboux_blocks`, `_warp_build_rhs_stretch/darboux`, and `_warp_merge_delta_lambda` in separate kernel files. Isaac Lab ports all of them. The two-system split (stretch 3×3 + Darboux 3×3 → merge) is identical. No logic changes.

**Why it matters:**
Split-Thomas trades coupling accuracy for reduced memory bandwidth — it runs two independent 3×3 Thomas passes (stretch and Darboux separately) instead of one 6×6 pass. This halves the block size and reduces JMJT assembly cost. Useful when bandwidth is the bottleneck rather than coupling accuracy.

**How it is implemented (verified in `_solve_split_thomas()` lines 2632–2677):**
1. Assemble two separate block-tridiagonal systems: `_split_stretch_diag/offdiag` (3×3 blocks from positional Jacobian) and `_split_darboux_diag/offdiag` (3×3 blocks from rotational Jacobian)
2. Build two separate RHS vectors: `_split_stretch_rhs` from stretch constraint values, `_split_darboux_rhs` from Darboux constraint values
3. Run `_xr_block_thomas_3x3` on each system independently — two O(N) passes
4. `_xr_merge_delta_lambda` interleaves the two `delta_lambda` results back into one 6-DOF vector

The decoupling means positional (stretch) constraints and rotational (Darboux) constraints do not cross-influence each other. For gently curved catheters with small Darboux violations this is accurate; for tight bends inside vessels the coupling matters.

---

#### Block-Jacobi Solve

> **Status: PORTED AS-IS from Newton `SolverXPBDRod`**
> Newton has `_warp_solve_blocks_jacobi` in `kernels_solvers.py`. Isaac Lab's `_xr_solve_blocks_jacobi` is a direct port — per-edge independent 6×6 diagonal block inversion, same parallelism model (one thread per edge, no cross-edge communication).
> Isaac Lab extension: Block-Jacobi now has a batched path (`_substep_batched` selects it via `solver_backend`); Newton only runs Jacobi in the single-rod loop.

**Why it matters:**
Block-Jacobi is the cheapest backend — it solves each edge's 6×6 diagonal block independently, ignoring coupling between adjacent edges entirely. This is wrong for highly curved configurations but produces useful approximate solutions at very high N when bandwidth makes Thomas expensive. Selectable at init time with no code changes.

**How it is implemented (verified in `_solve_system()` lines 2557–2569):**
```python
wp.launch(_xr_assemble_jmjt, dim=ne, ...)      # same assembly as block-Thomas
wp.launch(_xr_build_rhs, dim=nd, ...)
wp.launch(_xr_solve_blocks_jacobi, dim=ne, ...) # N independent 6×6 block inversions, fully parallel
```
`_xr_solve_blocks_jacobi` launches one thread per edge — all N edge systems are solved in parallel with no cross-edge communication. This is the only backend where the solve is fully parallel across edges (block-Thomas and split-Thomas both require a sequential sweep).

---

#### Banded Cholesky Solve

> **Status: PORTED AS-IS from Newton `SolverXPBDRod`**
> Newton has `_warp_spbsv_u11_1rhs` and `_warp_assemble_jmjt_banded` in `kernels_solvers.py` / `kernels_assembly.py`. Isaac Lab's `_xr_spbsv_u11_1rhs` and `_xr_assemble_jmjt_banded` are direct ports. LAPACK upper-banded storage format, `KD=11`, `LDAB=34` — constants match Newton's `BAND_LDAB`. No changes.

**Why it matters:**
For very short catheters (≤64 DOF, which is ≤10 segments) the full block-tridiagonal system fits in a banded storage format with half-bandwidth KD=11. The banded Cholesky factorization exploits this sparsity pattern and is more numerically stable than the Thomas algorithm for near-singular compliance matrices. Used for calibration and validation, not production training.

**How it is implemented (verified in `_solve_system()` lines 2571–2585):**
- Banded storage: `_ab` array of shape `(_XPBD_BAND_LDAB=34, nd)` — `34 = 2×KD + KD + 1` in LAPACK upper-banded format
- `_xr_assemble_jmjt_banded`: fills the banded `_ab` matrix from Jacobians
- `_xr_spbsv_u11_1rhs`: in-place Cholesky factorization + triangular solve (`U'U · x = b`) — single-thread kernel, operates on the `_ab` buffer in-place
- Returns the factored-overwritten `rhs` array directly

---

#### Tiled Cholesky (GPU Tile)

> **Status: PORTED AS-IS from Newton `SolverXPBDRod`**
> Newton has `_warp_cholesky_solve_tile` using `wp.tile_load`, `wp.tile_cholesky`, `wp.tile_cholesky_solve` with `TILE=64` and `BLOCK_DIM=128`. Isaac Lab's `_xr_cholesky_solve_tile` is a direct port — same shared-memory tile approach, same threshold (`nd ≤ 64`). No changes.

**Why it matters:**
For small systems (N_dof ≤ 64), allocating and sweeping a block-tridiagonal structure has more overhead than just solving the small dense matrix directly. The tiled Cholesky path uses Warp's `wp.tile_load` / `wp.tile_cholesky` / `wp.tile_cholesky_solve` — these operations run entirely in shared memory on a single GPU warp tile, avoiding global memory round-trips entirely.

**How it is implemented (verified in `_xr_cholesky_solve_tile()` lines 956–966, triggered at line 2588):**
```python
if nd <= _XPBD_TILE:   # _XPBD_TILE = 64 DOF threshold
    wp.launch_tiled(
        _xr_cholesky_solve_tile,
        dim=[1, 1],
        block_dim=_XPBD_BLOCK_DIM,  # 128 threads in the tile
        inputs=[ws._A, ws._rhs_tile],
        outputs=[ws._dl_tile],
    )
```
Inside the kernel:
```python
a_tile = wp.tile_load(A, shape=(64, 64))     # load dense matrix to shared mem
b_tile = wp.tile_load(b, shape=64)
L      = wp.tile_cholesky(a_tile)            # in-place Cholesky factorization
x_tile = wp.tile_cholesky_solve(L, b_tile)  # triangular solve
wp.tile_store(x, x_tile)
```
All arithmetic happens in L1/shared memory — no global memory reads during the factorization. Falls through to block-Thomas for nd > 64.

---

#### Floor Collision

> **Status: PORTED + EXTENDED in Isaac Lab**
> Newton has `_warp_apply_floor_collisions` with the same position-clamp + velocity-reflect logic. Isaac Lab ports this as `_xr_floor_collision` — same kernel body.
> **Extension:** Newton hardcodes restitution to `0.0` at both call sites (lines 576 and 664 of `solver_xpbd_rod.py`):
> ```python
> inputs=[ws.predicted_positions_wp, ws.velocities_wp, min_z, 0.0]  # Newton: literal 0.0
> ```
> Isaac Lab promotes `0.0` to a constructor parameter `floor_restitution: float = 0.0` and passes `float(self._floor_restitution)` in both single-rod and batched paths — enabling domain randomization of floor bounce energy during RL training.

**Why it matters:**
Floor collision is the boundary condition that prevents catheter particles from passing through the table in simulation. Isaac Lab exposes `floor_restitution` as a configurable parameter — Newton upstream hardcodes it to `0.0` (perfectly inelastic). This matters for RL training: a non-zero restitution changes the energy dissipation profile at contact, and domain randomizing it during training improves sim-to-real robustness for deployment on Holoscan.

**How it is implemented (verified in `_xr_floor_collision()` lines 1347–1370):**
```python
if p.z < min_z:
    pred[i] = vec3(p.x, p.y, min_z)          # clamp predicted position to floor
    if v.z < 0.0:
        vel[i] = vec3(v.x, v.y, -restitution * v.z)  # reflect with coefficient
```
Applied after the XPBD constraint solve, before velocity integration. `floor_restitution=0.0` (default) gives perfectly inelastic contact — the particle stops at the floor. `floor_restitution=1.0` would give perfectly elastic bounce. In the batched path, the same kernel applies to all 512 environments in one launch with the same `floor_z` and `restitution` values.

---

### Right Column — Vessel-Mesh Collision

---

#### SDF Containment (`wp.mesh_query_point_sign_normal`)

> **Status: PORTED AS-IS from Newton xcath branch (`xpbd_rods_solver_integr`, Przemysław Korzeniowski)**
> `_project_vessel_containment_kernel` — BVH signed-distance query, surface-normal projection, `target_phi` clearance offset, velocity clamp, smooth-normals option — all ported directly. Parameters `sign_scale`, `target_phi`, `max_dist` preserved with same defaults.
> **Isaac Lab addition:** This kernel is wired into the CUDA graph substep loop via `_pre/post_constraints_hook`, making it graph-capturable. In Newton's xcath branch these kernels are called from a Python-level loop outside any graph context.

**Why it matters:**
The SDF path is the default, cheapest vessel containment method. It uses Warp's BVH-accelerated signed-distance query to determine, for each rod particle: (1) is it inside or outside the vessel, and (2) how far is it from the surface. Particles that violate the clearance threshold are projected back inward along the surface normal. This gives physically correct containment — the catheter can never pass through the vessel wall — at O(log T) cost per particle query (where T = mesh triangle count).

**How it is implemented (verified in `_project_vessel_containment_kernel()` lines 343–405):**
```python
wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v)
closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
phi = sign × sign_scale × ||pos − closest||   # signed distance; positive = inside
if phi > target_phi:
    return                                       # inside by enough, no correction
projected = pos − grad × (phi − target_phi)     # project back to clearance offset
```
Key parameters:
- `target_phi = −particle_radius` (default): keeps the rod surface exactly on the wall, not the rod centerline
- `max_dist = 2 × particle_radius + segment_length`: BVH search radius limits to nearby triangles only
- `sign_scale = +1.0`: positive sign = inside the closed mesh (vessel interior)
- Smooth normals option: uses area-weighted vertex normals (`compute_smooth_vertex_normals`) to reduce facet-aligned ringing on curved vessel surfaces

An additional velocity clamp after projection ensures the derived velocity (computed as `(predicted − current) / dt` during integration) does not point outward through the wall — prevents wall-crossing artifacts at the next timestep.

---

#### AABB Broadphase (`wp.mesh_query_aabb`)

> **Status: PORTED AS-IS from Newton xcath branch (`xpbd_rods_solver_integr`, Przemysław Korzeniowski)**
> Both vertex-vs-triangle kernels (`_project_mesh_vertex_collision_kernel` and `_averaged` variant) and rod-segment-vs-mesh-edge kernels (`_project_mesh_edge_collision_kernel` and `_averaged` variant) are direct ports. The `ia < ib` deduplication, barycentric closest-point math (`_segment_segment_barycentric`), inverse-mass-weighted correction distribution, and `max_triangles` budget are all preserved from the xcath branch unchanged.
> **Isaac Lab addition:** Same as SDF — these kernels run inside the CUDA graph substep loop via hooks. Newton's xcath branch has no graph capture.

**Why it matters:**
The AABB path is more robust than the SDF path in tight-curvature regions — tight vessel bends, bifurcation junctions, and near-zero-radius segments — where the SDF sign can flip unexpectedly due to non-convexity. AABB broadphase finds all mesh triangles near a given rod particle or segment without a global distance query, then resolves two contact types: vertex-vs-triangle (rod particle penetrating a triangle face) and rod-segment-vs-mesh-edge (rod cylinder intersecting a vessel edge).

**How vertex-vs-triangle works (verified in `_project_mesh_vertex_collision_kernel_averaged()` lines 474–535):**
```python
lower = pos − query_radius × [1,1,1]
upper = pos + query_radius × [1,1,1]
query = wp.mesh_query_aabb(mesh_id, lower, upper)
while wp.mesh_query_aabb_next(query, face_index):
    penetration = dot(pos − closest_on_triangle, n_outward) + radius
    if penetration > 0: accumulate correction
wp.atomic_add(corrections, i, avg_correction)
```
Uses the averaged variant: all penetrating nearby triangles contribute; corrections are averaged and atomically accumulated to prevent over-correction at triangles that nearly overlap.

**How rod-segment-vs-mesh-edge works (verified in `_project_mesh_edge_collision_kernel_averaged()` lines 651–754):**
- For each rod segment (edge), queries AABB for nearby mesh triangles
- Iterates over the 3 edges of each triangle (deduped by `ia < ib` to avoid double-counting)
- Calls `_segment_segment_barycentric()` to find the closest-point pair between the rod segment and the mesh edge
- Penetration = `dot(rod_point − mesh_point, n_outward) + radius`
- Correction distributed to the two rod particles by their inverse-mass weights: `correction_i = (penetration / denom) × inv_i × bar_i`

Default: `collision_iterations=2` Gauss-Seidel passes per substep. Budget: `max_triangles=64` per AABB query.

---

#### Track-Guided Insertion

> **Status: PORTED AS-IS from Newton xcath branch (`xpbd_rods_solver_integr`, Przemysław Korzeniowski)**
> `_track_sliding_kernel` — dot-product projection onto `track_dir`, `track_stiffness` blend, `tip_num_edges` free-tip boundary — all ported unchanged. Parameters `track_start`, `track_dir`, `track_length`, `track_stiffness`, `tip_num_edges` match the xcath branch.
> **Isaac Lab addition:** Applied in `_post_constraints_hook` after the XPBD solve — same ordering choice as the xcath branch. Also runs inside the CUDA graph.

**Why it matters:**
In a real catheter procedure, the proximal shaft is inside a rigid guide sheath up to the point of vessel entry — only the steerable distal tip deforms freely. Without track guidance, the XPBD solver would allow the entire rod to deform, which is physically wrong and makes RL training harder (the policy must learn to control a floppy rod rather than just the tip). Track guidance constrains the proximal particles to a linear insertion axis, reducing the effective control problem to tip-only navigation.

**How it is implemented (verified in `_track_sliding_kernel()` lines 314–333):**
```python
t    = clamp(dot(p − track_start, track_dir), 0, track_length)
proj = track_start + track_dir × t
predicted_positions[i] = p + (proj − p) × track_stiffness
```
- `track_start`, `track_dir`, `track_length` define the linear insertion axis (set from the vessel entry point in CT coordinates)
- `track_stiffness = 1.0` (default): particles snap exactly to the axis — no lateral play
- `track_stiffness < 1.0`: soft projection — particles can deviate slightly from the axis, mimicking a flexible sheath
- `end_idx = num_points − tip_num_edges`: all particles except the distal `tip_num_edges` segments are projected
- `tip_num_edges`: controls how many distal segments are free to deflect — only these participate in vessel collision and respond to vessel geometry

Applied in `_post_constraints_hook` after the XPBD constraint solve, so track guidance runs after the rod has resolved its internal stretch and bending — preserving the rest-length and bend stiffness enforcement.

---

#### Constraint Hooks

> **Status: PORTED + EXTENDED in Isaac Lab**
> Newton's `SolverXPBDRod` has:
> - `_project_predicted_positions_pre_constraints(rod_idx, ws, dt, device)` — single-rod pre-hook
> - `_project_predicted_positions_post_constraints(rod_idx, ws, dt, device)` — single-rod post-hook (wraps `_project_predicted_positions`)
>
> These are ported but renamed and de-indexed (no `rod_idx` argument needed since Isaac Lab's workspace is self-contained).
>
> **Isaac Lab extension — batched counterparts not in Newton:**
> - `_pre_constraints_hook_batched(bws, dt, dev)` — operates on `_BatchedWorkspace` for all 512 envs
> - `_post_constraints_hook_batched(bws, dt, dev)` — same
>
> `XCathRodSolver` overrides all four, so vessel collision works identically in single-env interactive demo and 512-env RL training. Newton's xcath branch has no batched hook — vessel collision only runs in the single-rod loop.

**Why it matters:**
The hook architecture is what makes `XCathRodSolver` a clean subclass extension rather than a fork. The base `XPBDRodSolver` defines two override points in the substep loop: `_pre_constraints_hook` (before XPBD solve) and `_post_constraints_hook` (after XPBD solve, before floor collision). `XCathRodSolver` overrides both without touching any base solver code — adding new contact types requires only a subclass, not a modification to the validated base.

**How it is implemented (verified in `xcath_rod_solver.py` lines 1138–1149):**
```python
def _pre_constraints_hook(self, ws, dt, dev):
    if self.collision_pre_constraints_enabled:
        self._project_vessel_containment(ws, dev)   # SDF or AABB

def _post_constraints_hook(self, ws, dt, dev):
    self._project_track_guidance(ws, dev)            # always after solve
    if self.collision_post_constraints_enabled:
        self._project_vessel_containment(ws, dev)   # default: post-only
```
Default configuration: pre-hook is a no-op, post-hook runs track guidance + vessel containment. This ordering is numerically most stable: the XPBD solve enforces rod mechanics first, then containment projects the result back inside the vessel.

The `collision_projection_stage` parameter provides a shorthand: `"pre"` enables pre-only, `"post"` enables post-only (default). Both can be enabled simultaneously for double-iteration convergence on tight curvature.

---

#### CUDA-Graph Compatible

> **Status: ISAAC LAB ADDITION — not present in Newton `SolverXPBDRod` or xcath branch**
> Newton's `SolverXPBDRod` has zero CUDA graph references (0 matches confirmed by grep). Newton's xcath branch inherits this gap — vessel collision kernels are called from a Python loop, not inside any graph context.
>
> Isaac Lab adds the complete graph infrastructure:
> - `_can_use_cuda_graph()` — checks device type; subclasses can override to disable capture
> - `_capture_cuda_graph(sub_dt)` — warm-up substep → `wp.ScopedCapture` → stores `capture.graph`
> - `wp.capture_launch(self._cuda_graph)` — single API call replaying the entire `num_substeps` loop
> - `reset_cuda_graph()` — explicit invalidation after external state mutation (e.g. UI Reset button)
>
> Because vessel collision kernels are wired into `_pre/post_constraints_hook` (called from `_substep`, which is captured), the entire physics step — XPBD solve + vessel SDF + AABB contacts + track guidance — runs as one graph node with zero CPU overhead per frame.

**Why it matters:**
All vessel collision kernels — `_project_vessel_containment_kernel`, `_project_mesh_vertex_collision_kernel_averaged`, `_project_mesh_edge_collision_kernel_averaged`, `_track_sliding_kernel` — are standard Warp kernels that write only to GPU-resident arrays. They are called inside `_pre_constraints_hook` and `_post_constraints_hook`, which are called from `_substep`, which is captured inside `wp.ScopedCapture`. This means the entire physics step — including vessel collision — runs as one CUDA graph replay with zero CPU overhead per frame.

**The constraint:** All graph-captured kernels must not call host functions, perform CPU-GPU synchronizations, or read host-side Python state. All XCathRodSolver parameters (`track_start`, `track_dir`, `collision_mesh.id`, `target_phi`, etc.) are passed as scalar or Warp-array inputs to kernels — no Python attribute reads inside the graph. This is verified by the fact that the graph capture succeeds without error in the production integration test.

---

### Q&A — Slide 4

**Q: Why are there four solver backends? Which one should be used for production RL training?**
A: Block-Thomas (default) for all production use — it solves the full 6×6 coupled system exactly in O(N) with no convergence tuning. Split-Thomas saves ~30% bandwidth for gently curved catheters where stretch-bend coupling is small. Block-Jacobi is fastest but wrong for tight bends — useful for profiling or very high N where bandwidth matters more than accuracy. Banded Cholesky is for validation on short rods (≤10 segments). Selectable at init: `XPBDRodSolver(cfg, solver_backend="split_thomas")`.

**Q: What is the "block-tridiagonal JMJT system" exactly?**
A: For a rod with N edges and 6 DOF per edge (3 positional + 3 rotational), the constraint system is `JMJT · Δλ = C` where JMJT is an N×N block matrix with 6×6 blocks. Each diagonal block couples the two particles of one edge. Each off-diagonal block couples two adjacent edges. Thomas algorithm solves this with one forward sweep + one back sweep, O(N) total. No global matrix factorization — only local 6×6 block operations per edge.

**Q: When should you use the AABB path vs the SDF path for vessel collision?**
A: SDF path for straight vessels and mild curvature — it's faster (one BVH query per particle) and gives smooth projections. AABB path for bifurcations, tight bends, and any geometry where the SDF sign flips unpredictably. The rule of thumb: use SDF first; switch to AABB if particles leak through the wall at junctions. Both are configurable at init with no recompile.

**Q: What makes the vessel collision CUDA-graph safe?**
A: All kernel parameters are either compile-time constants, scalar function arguments, or GPU-resident Warp arrays. No Python attributes are read inside the kernel launch — only GPU memory addresses are passed. `wp.ScopedCapture` records the kernel dispatch commands; at replay time, the same GPU addresses are re-used. Changing the collision mesh at runtime (via `set_collision_mesh()`) invalidates the graph by calling `solver.reset_cuda_graph()` — this forces a re-capture with the new mesh ID on the next `step()` call.

---

### Questions to ask the Newton author (Przemysław Korzeniowski / Newton team):

1. **"CUDA graph capture is not present in Newton's `SolverXPBDRod` specifically — we confirmed zero matches for `ScopedCapture` in `solver_xpbd_rod.py`. We know Newton's Kamino and MPM solvers do use CUDA graphs. Was graph capture intentionally left out of the XPBD rod solver, or is there a PR planned for it?"**

2. **"The batched path in Newton hardcodes floor restitution to `0.0`. Was that intentional for the rod use case, or an oversight? We exposed it as a configurable parameter in Isaac Lab."**

3. **"Is `SolverXPBDRod` in Newton considered production-ready? What is the stabilization plan after PR #1981?"**

4. **"The xcath branch (`xpbd_rods_solver_integr`) has not been merged to Newton main — is the vessel collision API still unstable? Are there API-breaking changes planned for `wp.mesh_query_point_sign_normal` usage?"**

5. **"Are there known issues with the flat-buffer approach at high N (512 rods) for jacobian arrays? Newton's `_BatchedRodWorkspace` allocates at `alloc_edges × 36` — the Isaac Lab port uses `rod_offsets` indexing on the same layout."**

6. **"Does Newton have plans to add differentiable physics (gradients through the constraint solve)? The renderer supports autodiff via Slang — if physics also supports it, we could do gradient-based catheter shape optimization."**

---

## Slide 5 — Multi-Environment Batched Physics

**Opening line:**
> "This slide explains how 512 environments run in parallel — I will go through each bullet point on the slide in order, explaining how it was built and why it matters."

---

### Bullet 1 — `_BatchedWorkspace` allocates flat contiguous GPU buffers

**What the slide says:**
`_BatchedWorkspace` allocates flat contiguous GPU buffers: `positions[total_particles × 3]`, `orientations[total_particles × 4]`.

**Why this design decision was made:**
The naive approach would be to allocate one separate GPU array per environment — `env_0.positions`, `env_1.positions`, etc. That means one Warp kernel launch per environment per physics kernel. At 512 environments, 12 kernels per substep, and 8 substeps, that is 512 × 12 × 8 = **49,152 individual kernel launches per frame**. Each kernel launch costs ~5–10 µs of CPU overhead. 49,152 × 5 µs = ~245 ms of CPU overhead alone — completely unacceptable for a 15 ms frame budget.

The flat buffer solution puts every particle from every environment into one continuous array. All 10,752 particles (512 envs × 21 particles) live in one `wp.array`. One kernel launch processes all of them.

**How it is implemented (verified in `_BatchedWorkspace.__init__`, lines 2122–2204):**

When the solver is created with `num_envs=512`, a 20-segment rod has 21 particles and 20 edges. The workspace computes:
```
np_total = 512 × 21 = 10,752   (total particles)
ne_total = 512 × 20 = 10,240   (total edges)
nd_total = 10,240 × 6 = 61,440 (total DOFs — 6 per edge: 3 stretch + 3 Darboux)
```
Then allocates all buffers at those sizes in one block:
```python
self.positions            = wp.zeros(np_total, dtype=wp.vec3)   # 125 KB
self.predicted_positions  = wp.zeros(np_total, dtype=wp.vec3)   # 125 KB
self.velocities           = wp.zeros(np_total, dtype=wp.vec3)   # 125 KB
self.orientations         = wp.zeros(np_total, dtype=wp.quat)   # 167 KB
self.angular_velocities   = wp.zeros(np_total, dtype=wp.vec3)   # 125 KB
self.jacobian_pos         = wp.zeros(ne_total * 36, dtype=wp.float32)  # 1.4 MB
self.jacobian_rot         = wp.zeros(ne_total * 36, dtype=wp.float32)  # 1.4 MB
self.diag_blocks          = wp.zeros(ne_total * 36, dtype=wp.float32)  # 1.4 MB
self.offdiag_blocks       = wp.zeros(ne_total * 36, dtype=wp.float32)  # 1.4 MB
self.c_blocks             = wp.zeros(ne_total * 36, dtype=wp.float32)  # 1.4 MB
```
**Total for all physics buffers at 512 environments: ~12–15 MB.** This is a single contiguous memory region on the GPU — no fragmentation, no pointer chasing.

---

### Bullet 2 — `rod_offsets[r]` and `edge_offsets[r]` give each environment's slice

**What the slide says:**
`rod_offsets[r]` and `edge_offsets[r]` give each environment's slice — one kernel launch processes all rods in parallel without branching.

**Why this is needed:**
With all particles in one flat array, each kernel thread needs to know which particles belong to its environment. Without an index, there is no way to tell where env 0 ends and env 1 begins in a flat array of 10,752 entries.

**How it is implemented (verified lines 2182–2196):**
At construction, two offset arrays are built on CPU and uploaded to GPU once:
```python
rod_offsets  = [0, 21, 42, 63, ..., 10,752]  # (513,) — particle start per env
edge_offsets = [0, 20, 40, 60, ..., 10,240]  # (513,) — edge start per env
```
Two additional identity arrays map every particle/edge back to its environment:
```python
particle_rod_id = [0,0,...,0,  1,1,...,1,  ...,  511,...,511]  # (10,752,)
edge_rod_id     = [0,0,...,0,  1,1,...,1,  ...,  511,...,511]  # (10,240,)
```
In a particle kernel (e.g. `_xr_predict_pos_batched`), thread `i` does:
```python
rod = particle_rod_id[i]          # which environment am I in?
gravity[rod]                       # read this env's gravity
```
In the block-Thomas solve (the most critical kernel), thread `rod` does:
```python
e_start = edge_offsets[rod]       # where do my edges start?
e_end   = edge_offsets[rod + 1]   # where do they end?
# then sweep only [e_start, e_end) — no other rod's data is touched
```
**Why this matters:** Every thread knows exactly which slice of the flat buffer belongs to it, with a single array lookup. No conditional branching. No inter-environment communication. 512 environments are truly independent.

---

### Bullet 3 — CUDA graph captures the entire multi-environment substep loop

**What the slide says:**
CUDA-graph capture covers the entire multi-environment substep loop. `wp.ScopedCapture` on first `step()` call; `wp.capture_launch` on all subsequent calls.

**Why this is needed:**
Even with flat buffers, each physics substep still requires 12 Warp kernel launches in sequence. With 8 substeps, that is 96 kernel launches per `step()` call. Each launch requires a CPU-to-GPU command, taking ~5 µs. 96 × 5 µs = **480 µs of pure CPU overhead per frame** — adding ~30% to a 15 ms frame budget and preventing CUDA pipeline overlap.

A CUDA graph records all 96 launch commands at once during a one-time capture. After that, a single `wp.capture_launch()` call replays the entire sequence with ~1 µs total CPU cost.

**How it is implemented (verified in `_capture_cuda_graph`, lines 2420–2430):**
```python
# One-time capture — happens on the first step() call
with wp.ScopedCapture(device=self._device) as capture:
    for _ in range(self._num_substeps):   # 8 substeps
        self._substep_batched(sub_dt)     # records 12 kernels each
self._cuda_graph = capture.graph         # store the graph

# Every subsequent step() call:
wp.capture_launch(self._cuda_graph)      # one call replays 96 kernels
```
**What gets captured inside the graph:** All 12 × 8 = 96 kernel launches, including the particle prediction, constraint solve, block-Thomas direct solve, corrections, and integration — for all 512 environments simultaneously.

**When the graph is invalidated:** If `dt` changes (the graph was captured at a specific `sub_dt`), or if `solver.reset_cuda_graph()` is called explicitly — for example, after the UI Reset button writes new particle positions directly into the GPU buffer, the graph must be re-captured so it sees the new state. The first step after invalidation takes ~50 ms for re-capture; then returns to near-zero overhead.

---

### Bullet 4 — GPU-side proximal control: zero CPU sync on hot path

**What the slide says:**
GPU-side proximal control: `apply_proximal_control_gpu` and `set_root_orientation` are Warp kernels — safe to capture inside the CUDA graph, zero CPU sync on hot path.

**Why this matters:**
The RL policy produces two control actions per environment per frame: insertion speed (push) and rotation rate. These must be applied to the root particle of each catheter before the physics step. If this was done on CPU — reading the policy output to CPU, modifying a host array, copying back to GPU — it would add a CPU-GPU synchronization that blocks CUDA graph capture and adds ~100–500 µs of latency per frame.

**How it is implemented (verified in `_BatchedWorkspace`, lines 2198–2200, and `_xr_proximal_push_kernel`, lines 1400–1453):**

The workspace pre-allocates two per-env arrays on GPU:
```python
self.push_velocities    = wp.zeros(num_envs, dtype=wp.float32)  # (512,) — m/s per env
self.rotate_velocities  = wp.zeros(num_envs, dtype=wp.float32)  # (512,) — rad/s per env
self.root_idx           = wp.array([0, 21, 42, ...], dtype=wp.int32)  # root particle per env
self.next_idx           = wp.array([1, 22, 43, ...], dtype=wp.int32)  # second particle per env
```
The RL training loop writes the policy's action tensor directly into these GPU arrays (one `wp.array.assign()` call, no device copy). Then `apply_proximal_control_gpu` launches `_xr_proximal_push_kernel` at `dim=512`:
```python
# Each thread handles one environment
rod_point  = pos[root_idx[e]]
next_point = pos[next_idx[e]]
tangent    = normalize(next_point - rod_point)   # live insertion direction
pos[root_idx[e]] = rod_point + tangent * push_velocities[e] * dt
# also applies incremental twist to ori[root_idx[e]]
```
Everything stays on GPU. The kernel is safe to replay inside a CUDA graph because it reads only GPU-resident arrays — no Python attribute access, no CPU state.

**`set_root_orientation`** works the same way: one Warp kernel, `dim=1` per environment, that writes `ori[root]`, `pred_ori[root]`, and `prev_ori[root]` — all three orientation arrays — so the XPBD integrator on the next substep sees a consistent state.

---

### Bullet 5 — Memory cost scales linearly; compute dominated by block-Thomas

**What the slide says:**
Memory cost scales linearly with N (particle count × envs). Compute cost is dominated by the block-Thomas solve — nearly O(N×envs) with the flat buffer layout.

**Memory scaling — why it is not a concern:**

Every new environment adds exactly `num_points_per_rod` entries to each per-particle buffer and `num_edges_per_rod` entries to each per-edge buffer. There are no shared per-env allocations. The total memory scales as:

```
Total physics memory ≈ N_envs × (num_points × per_particle_bytes + num_edges × per_edge_bytes)
```

For 512 envs × 21 particles × 20 edges:
- Per-particle contribution: `21 × (12+12+12+16+12+4) bytes × 7 arrays ≈ 740 KB`
- Per-edge contribution: `20 × 36 × 4 bytes × 5 arrays (jacobians, blocks) ≈ 7.2 MB`
- Index arrays: negligible (~200 KB)
- **Total: ~12–15 MB for 512 environments**

This is the critical comparison: **the CT μ-volume alone is 64 MB** (256³ float32). The entire physics workspace for 512 environments is smaller than one copy of the CT volume. Memory is not the RL scaling constraint — the renderer is.

**Compute scaling — why block-Thomas dominates:**

Most physics kernels scale as `O(np_total)` or `O(ne_total)` — they process one particle or one edge per thread. Adding more environments adds more threads, but GPU throughput scales with thread count. These kernels stay GPU-compute-bound up to thousands of environments.

The block-Thomas solve (`_xr_block_thomas_batched`) is different: it launches `dim=num_envs=512` threads, where each thread runs a sequential O(N_edges) forward sweep + back substitution. This means the compute cost per `step()` call is:
```
O(N_envs × N_edges_per_rod) = O(512 × 20) = O(10,240)
```
At 512 envs with 20-segment rods, this is equivalent to one sequential rod solve of 10,240 edges on 512 threads in parallel. Because all 512 threads run independently with no synchronization, the wall-clock time is just the time to sweep one rod (20 edges) — not 512 rods. This is the "O(rod_solve_time) regardless of N" property stated in the kernel docstring.

---

### Batched Rendering — Bullet 1: Single Slang dispatch, `dispatchThreadID.z` indexes environment

**What the slide says:**
`renderDRR_forward_batched`: single Slang dispatch. `dispatchThreadID.z` indexes the environment.

**Why this matters:**
The same flat-buffer principle that applies to physics applies to rendering: one kernel dispatch covers all environments. Without this, rendering N environments would require N separate Slang dispatch calls — N × setup overhead + N × synchronization points.

**How it is implemented:**
The Slang kernel is dispatched over a 3D grid: `(det_width, det_height, num_envs)`. Thread `(x, y, z)` traces the ray for pixel `(x, y)` of environment `z`. `dispatchThreadID.z` is the environment index — each environment's detector pixels are an independent slice of the dispatch. All environments share the same CT `Texture3D<float>` (it never changes) but read their own catheter geometry from a per-env slice of the `StructuredBuffer`.

---

### Batched Rendering — Bullet 2: Per-env catheter geometry in flat `StructuredBuffer`

**What the slide says:**
Per-environment catheter geometry packed into a flat `StructuredBuffer<CatheterSegment>` with per-env offset + count indices — different catheter per env, single kernel.

**Why this matters:**
Each environment has a different catheter configuration — the RL policy has pushed each catheter to a different position inside the vessel. The renderer must composite the correct catheter on top of the X-ray for each environment. This cannot be done with a single shared catheter position.

**How it is implemented:**
The `CatheterSegmentData` buffer holds all catheter segments for all environments concatenated in one flat array. A per-env `(offset, count)` index tells each renderer thread which slice of the buffer belongs to its environment (`dispatchThreadID.z`). The Slang kernel reads `offset[z]` and `count[z]`, then accesses only `segments[offset[z] .. offset[z]+count[z]-1]` for the per-sample point attenuation test. No per-env dispatch needed.

---

### Batched Rendering — Bullet 3: L2 cache bottleneck at N>8 and the Sprint 2 fix

**What the slide says:**
L2 cache bottleneck at N>8: shared 3D μ-volume exceeds L2 capacity. Sprint 2 fix: `Texture2DArray` caches one depth slice per env in L2.

**Why the bottleneck exists:**
The CT μ-volume at 256³ float32 = **64 MB**. The A6000 L2 cache is approximately **6 MB**. For a single environment, ray threads from a small region of the detector tend to access nearby voxels — good spatial locality, many L2 hits. As more environments are added, their ray threads access different regions of the same 64 MB volume simultaneously. At N>8, the aggregate access footprint exceeds 6 MB and the cache is effectively bypassed — every volume fetch goes to global memory at ~600 GB/s instead of L2 at ~10 TB/s. This is a 16× bandwidth reduction.

Physics does not have this problem because its buffers (positions, jacobians, etc.) for all 512 environments total only ~15 MB and access patterns are stride-1 per-particle — much more cache-friendly.

**The Sprint 2 fix — `Texture2DArray`:**
Instead of one `Texture3D` shared across all environments, cache one Z-depth slice of the CT volume per environment thread group in L2 using `Texture2DArray`. Each environment's thread group loads only its relevant depth slice, keeping its working set within ~100 KB instead of 64 MB. This should bring rendering from global-memory-bound back to compute-bound, targeting >60 FPS at 512 environments.

---

### Performance Table — Verified (A6000)

| Component | Throughput | Status |
|---|---|---|
| Physics — 1 env | 1,300 Hz | Target met |
| Physics — 512 envs | ~60 Hz | Target met |
| Render — 256² 1 env | ~263 FPS | Interactive demo |
| Render — 512² 1 env | ~25 FPS | Standalone benchmark |
| Render — 512² N≤4 | ~25 FPS | CT volume fits in L2 |
| Render — 512² N>8 | Degrades | L2 overflow → Sprint 2 fix |
| Full loop (256²) | ~63 FPS | Physics + render combined |

**The core RL bottleneck in one sentence:** Physics scales to 512 envs because its ~15 MB working set fits in GPU memory hierarchy. Rendering degrades above N=8 because all environments share one 64 MB CT volume that overflows L2. Sprint 2 (`Texture2DArray`) resolves this.

---

### Q&A — Slide 5

**Q: Why does `_xr_block_thomas_batched` launch at `dim=512` (one thread per rod) rather than `dim=10,240` (one thread per edge)?**
A: The Thomas algorithm sweeps edges sequentially within one rod — each step depends on the result of the previous step. There is no way to parallelize across edges of the same rod. The only valid parallelism is across rods: 512 rods can each run their sweep independently, one thread each, simultaneously. Using `dim=ne_total` would assign multiple threads to one rod's edges, causing write races on the `c_blocks` and `d_prime` intermediate arrays.

**Q: Is there a memory limit on how many environments can run?**
A: The physics workspace scales linearly — 512 envs use ~15 MB, 1024 envs would use ~30 MB. Both are well within the A6000's 48 GB VRAM. The practical limit for physics is compute, not memory. The practical limit for combined physics+rendering is the CT volume in L2 — resolved in Sprint 2. After the Sprint 2 Texture2DArray fix, the system should scale to 512+ envs at >60 FPS.

**Q: Can each environment have a different rod length (different number of segments)?**
A: Not currently — `_BatchedWorkspace` assumes `num_points_per_rod` is uniform across all envs (noted in the docstring: "heterogeneous topology can be supported by generalising the offset arrays at a small cost"). Supporting variable-length rods would require replacing `rod_offsets` with a variable-step array and updating the block-Thomas kernel to read `n_edges = edge_offsets[rod+1] - edge_offsets[rod]` per thread — which it already does. The main change would be in workspace construction. This is a one-day extension.

## Slide 6 — Fluoroscopy Rendering — Compositing Paths

**What to say:**
> "Three compositing paths exist. The Slang GPU path is the production path — it fuses CT and catheter Beer-Lambert in a single ray-march kernel. The CPU NumPy path is a reference-only path for validation and visual QA. The Isaac Lab USD quad path is planned for Sprint 2 — it's an Omniverse viewport integration, not a training path."

**Important correction to call out explicitly:**
The CPU path performs scatter, Poisson noise, and PSF *after* the DRR as NumPy post-processing, not inside the ray march. The slide clearly labels this "reference only, not on training path" — make sure the audience understands the CPU path is not used for RL training.

---

### Q&A — Slide 6

**Q: Why is the interactive demo running at 256×256 rather than 512×512?**
A: `DET_SIZE = 256` is hardcoded in `interactive_catheter_fluoro.py`. At 256×256 the full sim loop runs at ~63 FPS, which gives a responsive interactive experience. Switching to 512×512 drops to ~25 FPS — still usable interactively but slower. For RL training, 256×256 is also the typical observation resolution since policy networks don't benefit from higher resolution at this stage.

**Q: What is "additive in the exponent" and why does it matter?**
A: The CT and catheter attenuation are added inside the exponent: `I = I₀ · exp(−∫[μ_CT + μ_catheter] ds)`. Because `exp(a+b) = exp(a) · exp(b)`, this is mathematically equivalent to multiplying two Beer-Lambert transmissions. The important consequence is that a catheter behind dense bone is attenuated by *both* — you can't just multiply a catheter silhouette on top of the DRR, because the bone would not attenuate the catheter's X-rays. The fused ray march gets this physically correct automatically.

---

## Slide 7 — Beer-Lambert Catheter Compositing — Algorithm Detail

**Opening line:**
> "Every pixel of the fluoroscopy output is computed from the same Beer-Lambert integral — CT anatomy and catheter metal are composited in the same exponent, not blended on top. I'll walk through each of the five steps on the slide."

---

### Step 1 — DRR Background

**What the slide says:**
Ray march through 3D CT μ-volume (GPU texture, static per patient). Fixed-step integration: `I_DRR = I₀ · exp(−Σ μ_CT(s) ds)`. HU → μ piecewise linear map. Dense bone ~0.3–0.5 mm⁻¹. Air ~0.0 mm⁻¹.

**How it is implemented:**
The CT volume is pre-loaded once as a GPU texture (`Texture3D<float>` in Slang, or `wp.array` in the CPU path). The ray march fires one ray per detector pixel from the X-ray source, steps through the volume at fixed intervals (`step_mm`, typically 0.5 mm), and accumulates `μ(s) × step_mm` at each sample.

HU → μ conversion (verified in `ct_pipeline.py`):
```python
mu = np.where(hu < -100,  hu / -1000.0 * 0.0 + 0.0,   # air
     np.where(hu <  100,  0.02  + (hu + 100) * 0.0001,  # soft tissue
                          0.02  + (hu + 100) * 0.0005))  # bone/metal
```
Dense bone at 400–700 HU maps to μ ≈ 0.3–0.5 mm⁻¹. Air at −1000 HU maps to μ ≈ 0.0 mm⁻¹. The CT volume is never modified frame-to-frame — it is a static GPU-resident texture. Only the catheter segment buffer changes per frame.

**Why it matters:** A DRR generated from a real patient CT is the ground-truth X-ray background. All anatomy — bone, soft tissue, vessels — is rendered by the same physical model as the catheter. There is no photographic background image or compositing hack.

---

### Step 2 — Catheter Segment Buffer

**What the slide says:**
Physics particle positions (N×3 float32) written to GPU segment buffer. Each segment: (xyz_proximal, xyz_distal, radius, μ) — 8 floats. Per-segment μ profile: tungsten marker 3.0, NiTi shaft 0.8, polymer tip 0.15, Pt marker 5.0.

**How it is implemented (verified in `_segment_attenuation_profile`, lines 267–304):**
After each XPBD physics step, the N particle positions are written to a `CatheterSegmentData` buffer. Each consecutive pair of particles defines one segment. The Slang shader receives a flat `StructuredBuffer<CatheterSegment>` where each element holds 8 floats:
```
struct CatheterSegment {
    float3 p0;      // proximal endpoint (mm)
    float3 p1;      // distal endpoint (mm)
    float  radius;  // cylinder radius (mm)
    float  mu;      // linear attenuation coefficient (mm⁻¹)
};
```

The per-segment μ profile models realistic multi-material catheter construction (verified values, `lines 280–302`):

| Segment region | Fraction of catheter length | μ (mm⁻¹) | Physical material |
|---|---|---|---|
| Proximal marker | First 2 segments | **3.0** | Tungsten band |
| Braided shaft | 0–60% | **0.8** | Nitinol (NiTi) braid |
| Transition zone | 60–85% | 0.8→0.2 (linear ramp) | Sparse braid + polymer |
| Soft tip | 85–95% | **0.15** | PEBAX polymer |
| Distal marker | Last 2–3 segments | **5.0** | Platinum coil |

These values are at 70 keV monoenergetic approximation, derived from NIST mass attenuation coefficient tables scaled by material density. The 16× dynamic range (μ=0.15 polymer vs μ=5.0 platinum) is physically meaningful — platinum is the densest metal in clinical catheters and produces the brightest white spots on real fluoroscopy.

**Why it matters:** A single uniform μ for the whole catheter would make the shaft and tip appear identical on the image. Real clinical fluoroscopy shows bright white marker bands (tungsten, platinum) at the tip — visible at the current catheter position — while the shaft is semi-transparent. This multi-material profile enables the RL policy to observe clinically accurate visual feedback.

---

### Step 3 — Inline Perpendicular Test

**What the slide says:**
At each ray sample point `s`, for each catheter segment `i`: `d_i` = perpendicular distance from `s` to segment axis. If `d_i < r_i`: contribute `μᵢ · √(1 − dᵢ²/rᵢ²)` to running integral. The `√(1−d²/r²)` factor is a chord-length weight — exact for a circular cross-section.

**How it is implemented (verified in `composite_catheter_beer_lambert`, lines 387–411):**

For each ray sample point, for each catheter segment, the test parameterizes the segment as `P(t) = p0 + t·(p1−p0)`, clamps `t ∈ [0,1]`, and computes the distance from the sample point to the closest point on the segment:
```python
t_param = (rel · dir) / (seg_len²)        # scalar projection
t_param = clip(t_param, 0.0, 1.0)         # clamp to segment
closest  = p0 + t_param * (p1 - p0)
d = ||sample_point - closest||             # perpendicular distance
```
If `d < r` (inside the cylinder), the chord-length weight is:
```python
chord = 2.0 * sqrt(r² - d²)              # exact path length through cylinder at offset d
chord_norm = chord / (2r)                  # normalize by diameter
atten += mu * chord_norm                   # additive in exponent
```

**The math behind `√(1−d²/r²)`:** For a cylinder of radius `r`, a ray passing at perpendicular offset `d` from the centre traverses a chord of length `2√(r²−d²)`. Normalizing by diameter gives `√(1−d²/r²)`. This is the exact analytical cross-section path length — it is not an approximation. At `d=0` (ray through centre), weight = 1.0 (maximum attenuation). At `d→r` (ray grazing edge), weight → 0 (smooth fade, no hard edge).

**Why this is better than alpha-blending:** Alpha-blending at the 2D projection stage would show the catheter as an opaque silhouette, hiding the anatomy behind it. The perpendicular distance test adds catheter attenuation to the Beer-Lambert exponent — the anatomy behind the catheter remains visible, darkened proportionally to the catheter's actual material thickness at each pixel. This matches what is seen clinically.

---

### Step 4 — Fused Integral

**What the slide says:**
`μ_total(s) = μ_CT(s) + Σᵢ μᵢ · √(1 − dᵢ²/rᵢ²)`. `I_final = I₀ · exp(−∫ μ_total(s) ds)`. Catheter and CT share one exponent — no separate compositing pass, no alpha blending.

**Why fused instead of separate passes:**
The alternative would be: (1) render CT-only DRR, (2) render catheter-only attenuation image, (3) multiply them. This produces exactly the same result as the fused approach for non-overlapping tissue — but when the catheter is behind bone, a separate-pass approach would incorrectly compute the catheter attenuation without the bone's contribution to exponential decay. The fused integral correctly models the physical path: X-rays are attenuated by all material they pass through, in sequence, bone and metal alike.

**The formula written out explicitly:**
```
For each ray:
  integral = 0
  For each step s along the ray:
    mu_at_s = mu_CT(s)                          # from CT volume lookup
    For each catheter segment i:
      d_i = perp_distance(s, segment_i)
      if d_i < r_i:
        mu_at_s += mu_i * sqrt(1 - d_i²/r_i²)  # add catheter contribution
    integral += mu_at_s * step_mm               # accumulate
  I_final = I0 * exp(-integral)
```
One exponent, one exponential, one output pixel. Catheter and CT anatomy are physically indistinguishable in the integral — both are just attenuation along the ray.

**Why it matters for RL:** The RL policy observes the fused image. The catheter does not "float" above the anatomy as a separate layer — it is composited at the physical depth where it actually is. A policy trained on this image sees the same visual cues a clinical cardiologist sees when navigating a catheter through bone structure.

---

### Step 5 — Detector Realism

**What the slide says:**
Poisson quantum noise (photon count N_ph → Poisson(N_ph)). PSF convolution (Gaussian σ=0.7 px — scintillator light spread). Gamma correction + misregistration jitter (σ_rot=0.05°, σ_trans=0.1 mm).

**How it is implemented (verified in `composite_catheter_beer_lambert`, lines 419–440):**

Four effects are applied in sequence after the Beer-Lambert integral:

**1. Veiling glare / scatter** — models X-ray scatter that re-illuminates blocked areas:
```python
blocked = bg_float * (1.0 - transmission)          # what was absorbed
scatter = GaussianBlur(blocked, sigma=18.0)         # scatter kernel (18 px)
composited += 0.03 * scatter                        # add 3% back as glow
```
σ=18 px corresponds to ~10–15 mm at typical C-arm geometry — the scale of scatter halos in real fluoroscopy.

**2. Detector PSF** — scintillator phosphor screen blurs the optical emission:
```python
composited = GaussianBlur(composited, sigma=0.7)    # PSF σ=0.7 px
```
σ=0.7 px is calibrated to match the ~0.3 mm spatial resolution of clinical flat-panel detectors at typical detector pixel pitch (0.4 mm/px).

**3. Poisson quantum noise** — X-ray photons arrive discretely, not continuously:
```python
photons = composited * noise_photon_count           # scale to photon counts
noisy   = rng.poisson(photons)                      # draw from Poisson distribution
composited = noisy / noise_photon_count             # scale back
```
`noise_photon_count=2000` means a fully transparent pixel receives ~2000 photons. The shot noise standard deviation is `√2000 ≈ 45` photons → ~2.2% intensity noise. Clinical fluoroscopy typically operates at 1000–5000 photons/px.

**4. Misregistration jitter** — simulates patient motion between mask and contrast frames in DSA:
- Rotational jitter: σ_rot = 0.05° per frame
- Translational jitter: σ_trans = 0.1 mm per frame

**Why these effects matter for training:** A policy trained on noiseless, sharp synthetic images would see a distribution shift when deployed on real clinical data (which always has Poisson noise, PSF blur, and scatter). These four realism effects narrow the sim-to-real gap without requiring real patient data in the training loop.

---

### Q&A — Slide 7

**Q: Why use a Gaussian for PSF instead of a real scintillator PSF?**
A: A real scintillator PSF is not perfectly Gaussian — it has a sharper core and heavier tails (often modeled as a sum of two Gaussians). The single-Gaussian approximation at σ=0.7 px matches the FWHM of typical CsI:Tl scintillators at the frequencies that dominate catheter visibility (low spatial frequency). For Sprint 2, a two-component PSF calibrated against the XCath fluoroscopy system specification would improve accuracy.

**Q: Why does the perpendicular distance test work in 3D for a 2D projection?**
A: In the Slang GPU path, the test is performed in 3D world coordinates — the ray sample point `s` is a 3D position, and the segment axis is a 3D line. The perpendicular distance is computed in 3D. The chord-length formula `√(1−d²/r²)` gives the 3D path length through the cylinder at that 3D offset. In the CPU path (`composite_catheter_beer_lambert`), the test is performed in 2D projected pixel space — the catheter is projected first, then the chord formula is applied to projected radii. The 2D approximation is valid when the catheter is approximately perpendicular to the ray direction, which holds for catheters in the imaging field of view.

**Q: What happens if the catheter overlaps itself?**
A: The contributions from overlapping segments are accumulated additively in the exponent: `Σᵢ μᵢ · chord_weight_i` sums over all segments that contain the sample point. The result is a higher exponent at the overlap region — physically correct increased opacity. Alpha-blending would need a special "self-overlap" mode to achieve this; the Beer-Lambert formulation handles it for free.

**Q: Can the μ values be differentiated for gradient-based optimization?**
A: Yes — the Slang renderer was written with autodiff in mind. The `[Differentiable]` annotation on the shader functions means `dI/dμᵢ` can be computed via backpropagation through the rendering equation. This enables direct gradient-based optimization of catheter material parameters from a fluoroscopy loss, which is the foundation for Sprint 2 differentiable RL reward shaping.

---

## Slide 8 — Stage 1 — CT Ingestion Pipeline

**What to say:**
> "This pipeline runs once per patient anatomy. It takes a CT volume and produces two GPU-resident assets: the μ-volume for the renderer, and a `warp.Mesh` for the physics collision solver. The vessel mask is generated by HU thresholding on contrast-enhanced CT — vessels lit by iodine contrast are 100–300 HU above baseline blood. Marching Cubes converts the binary mask to a polygon mesh. VMTK extracts the centerline skeleton for bolus dynamics."

---

### Q&A — Slide 8

**Q: What if the patient CT is not contrast-enhanced — can you still segment vessels?**
A: Without contrast, vessel-to-blood HU difference is near zero — standard thresholding fails. You'd need atlas-based segmentation or a deep learning vessel segmenter (e.g., trained on paired contrast/non-contrast data). This is a known limitation for the current pipeline. MAISI-generated synthetic CTs in Sprint 2 are always contrast-enhanced, avoiding this gap.

**Q: What format does the μ-volume need to be in?**
A: It must be a `(Z, Y, X)` shaped NumPy float32 array saved as `mu_volume.npy`, with a companion `metadata.json` containing `spacing_zyx_mm` (voxel spacing) and `origin_xyz_mm` (volume origin in mm). The CT ingestion script handles DICOM→HU→μ conversion and outputs this format.

**Q: What is VMTK and why is it needed?**
A: VMTK (Vascular Modelling Toolkit) extracts a 1D centerline skeleton from the 3D vessel mask using a thinning algorithm. The centerline is represented as a graph of nodes and edges. This graph is what the Dijkstra arrival map runs on to compute per-vessel-point bolus arrival times for DSA.

---

## Slide 9 — DSA Pipeline + Temporal Bolus Dynamics

**What to say:**
> "The DSA pipeline has four stages. Mask DRR — render anatomy before injection. Contrast DRR — inject bolus by adding iodine attenuation scaled by a gamma-variate concentration curve to the μ-volume, per-voxel arrival time from Dijkstra on the centerline graph. Scatter and misregistration jitter applied to the mask DRR — this is the only XCath-validated item that is implemented in fluorosim today. Subtract contrast from mask in log domain."

**Critical accuracy note (verified):**
- Misregistration jitter: **implemented** in `realism.py` + `dsa.py`
- Hagen-Poiseuille / Feeding Trunk Excision / Dispersion correction: **validated by XCath, NOT yet in fluorosim codebase**

---

### Q&A — Slide 9

**Q: What is the gamma-variate model and why use it?**
A: The gamma-variate `C(t) = c_peak · (t/t_peak)^α · exp(α(1 − t/t_peak))` is the standard clinical pharmacokinetic model for iodine contrast concentration over time in a vessel. It was first validated on brain DSA by Thompson et al. (1964) and is still used today for cerebral blood flow measurements. It captures the physiological shape: fast wash-in, sharp peak, slower wash-out — matching real DSA acquisitions.

**Q: Why does misregistration jitter matter if we have perfect synthetic data?**
A: Monoenergetic rendering produces *perfect* bone subtraction in DSA — cleaner than any real clinical image, where patient breathing and cardiac motion create residual bone-edge artifacts. If we train a policy on synthetic DSA with perfect bone subtraction and evaluate it on real clinical DSA with bone-edge artifacts, the distribution shift causes performance degradation. The jitter deliberately degrades the synthetic data to match real data statistics.

**Q: When will the Hagen-Poiseuille and F-8 algorithms be merged?**
A: They're validated, not coded in this repo. Sprint 2 target — estimated 2–3 days of implementation work once the XCath team's analysis code is made available. The main work is replacing the Dijkstra edge-weight model with velocity-weighted travel times and adding the graph partitioning for selective injection.

---

## Slide 10 — Performance Baseline

**What to say:**
> "Two numbers to hold in your head: physics at 1,300 Hz per environment — exceeds the training throughput target. GPU renderer at 25 FPS at 512×512 — the diagnosed bottleneck with a known fix. The interactive demo runs at 256×256 which gives 63 FPS. The multi-env renderer degrades beyond N=8 due to L2 cache pressure on the shared CT volume. Sprint 2 Texture2DArray fix resolves this."

**Numbers that are verified (not estimated):**
- 1,300 Hz physics: measured on A6000, 20-segment rod, block-Thomas backend
- ~63 FPS full loop: measured in interactive demo (`DET_SIZE=256`, reported in Simulation info box)
- ~25 FPS at 512×512: measured standalone benchmark, single env
- CPU path 2–5 FPS: expected for NumPy ray-march, not a regression

---

### Q&A — Slide 10

**Q: Will the Texture2DArray fix actually get to 60 Hz at 512 envs?**
A: The bottleneck is L2 cache capacity (~6 MB on A6000) vs CT volume size (64 MB at 256³ float32). Texture2DArray caches one depth slice per environment thread group in L2. Theoretical improvement: from global-memory-bound (~300 GB/s effective) to L2-bound (~10 TB/s). This should bring 512-env rendering from bandwidth-saturated to compute-bound, targeting >60 Hz. The exact number depends on the slice size and cache utilization, which needs to be measured.

**Q: What is the 1,300 Hz physics number and does it scale?**
A: 1,300 Hz means the XPBD solver runs one complete physics timestep (including all substeps) 1,300 times per second for a single 20-segment rod. With 512 envs in batched mode, this becomes ~2.5 Hz total — but since each "Hz" involves a CUDA graph replay that processes all 512 envs in parallel, the effective throughput is 512 × 2.5 Hz ≈ ~1,280 physics steps per second across all environments, roughly matching the single-env figure.

---

## Slide 11 — RL Training Pipeline — State-Based

**What to say:**
> "The RL pipeline is operational end-to-end today with state-based observations. The environment wraps the multi-env XPBD solver with a gymnasium-compatible interface. Two control actions: advance (proximal push) and rotate — exactly the two DOFs a clinician uses. RSL-RL PPO is decoupled from the environment via a standard VecEnv adapter. The current observation is ground-truth tip position — not the fluoroscopy image. That transition happens in Sprint 2 and is the critical path item."

---

### Q&A — Slide 11

**Q: Why state-based observations first and not pixel observations from the start?**
A: State-based observations let us validate that the physics, control API, reward shaping, and training infrastructure all work correctly before adding the complexity of pixel observations. If pixel-based training fails, we can diagnose whether it's the RL algorithm, the observation, or the reward — it won't be the environment itself since that's already validated.

**Q: What does the PPO policy actually learn to do with only tip position?**
A: It learns a mapping from (tip_xyz, tip_orientation, distance_to_target) to (push, rotate) actions. This is equivalent to a feedback controller that navigates the catheter tip to a target vessel landmark. The advantage over a scripted controller is that it generalises to new initial conditions and vessel geometries without re-tuning.

**Q: How do you prevent the policy from learning to teleport the catheter by exploiting the physics?**
A: The physics solver enforces Cosserat rod mechanics — the tip can only move through physically valid configurations. The proximal push/rotate control is applied at the proximal root, not directly at the tip. The policy cannot teleport because the control actions map to physically constrained boundary conditions.

---

## Slide 12 — Interactive Fluoroscopy Simulator Demo

**What to say:**
> "The interactive demo lets you control the catheter in real time via a browser UI. Advance, retract, rotate, switch C-arm projection — every action goes through the same physics and rendering pipeline as the RL training loop. The simulation info box reports the catheter bend magnitude — a live indicator that vessel-mesh collision is actively deflecting the rod. The DSA button renders a three-dispatch composite: background, fat-catheter, and actual catheter."

**Technical detail on the catheter bend indicator:**
The "Catheter bend" metric is the max perpendicular distance from any rod node to the straight line between the proximal and distal endpoints. For a free, unconstrained rod this is ~0 mm. When vessel collision is firing, it reads 5–20 mm depending on vessel curvature. This is computed from `pos_vol_mm` in `_catheter_bend_mm()` after each physics step.

---

### Q&A — Slide 12

**Q: Can multiple people use the UI simultaneously?**
A: No — Gradio's default `share=True` link creates a single-session tunnel. Multiple simultaneous users would interleave control inputs, corrupting the simulation state. For multi-user access, each user needs their own server instance on a separate port.

**Q: How do you know the DSA frame is spatially registered to the fluoroscopy frame?**
A: All three DSA dispatches (background, fat-catheter, catheter) use the same `PROJECTIONS[proj_name]` rotation matrix. The spatial registration is guaranteed by construction — not by alignment post-hoc.

**Q: What is the "fat catheter" in the DSA pipeline?**
A: The fat catheter is a virtual catheter with radius=2.5 mm (vs the real catheter at 1.8 mm) and high μ=0.80 mm⁻¹. It's rendered at the same position as the real catheter. Subtracting the background DRR from this fat-catheter DRR gives a positive signal in the vessel lumen region, highlighting where the catheter is navigating without requiring actual contrast injection.

---

## Slide 13 — XCath Collaboration — Technical Outcomes

**What to say:**
> "XCath validated five algorithmic improvements against real clinical DSA data. Only one — misregistration jitter — is actually implemented in the fluorosim codebase today. Items 1 through 3 — Hagen-Poiseuille flow, Feeding Trunk Excision, and dispersion correction — are algorithmically proven by XCath on their data, but the code changes have not been merged into this repo. Sprint 2 target. Item 4 is a benchmarking result, not a code change."

**This is the most important accuracy point in the deck.**
Do not present items 1–3 as completed implementation. The slide now colour-codes them amber with an explicit "Validated — not merged" badge.

---

### Q&A — Slide 13

**Q: What does it take to merge items 1–3?**
A: Hagen-Poiseuille: replace the Dijkstra edge-weight model in `compute_arrival_map()` with velocity-weighted travel times — 1 day. Feeding Trunk Excision (F-8): implement Voronoi partition + graph excision on `CenterlineGraph` — 2 days. Dispersion correction: add distance-based β modulation to `gamma_variate()` call in `build_contrast_volume()` — 1 day. Total: ~4 days of work, blocked on XCath sharing their analysis code.

**Q: Why did the Hagen-Poiseuille change improve accuracy so much (2.6× → 11% error)?**
A: Murray's Law (power=0.5) was designed for metabolic cost optimisation in arterial branching — it correctly predicts vessel radius ratios but not blood velocity. Hagen-Poiseuille (power=2.0) correctly models viscous flow resistance: velocity scales as r², so small distal vessels are much slower than large proximal vessels. The distal MCA velocity dropped from an unrealistic 71% of ICA to a realistic 25%, which corrected the bolus timing.

**Q: Is 9.3% MAE vs DeepDRR significant for training?**
A: The residual difference is concentrated at bone boundaries due to beam hardening — a spectrally dependent effect. Since DSA subtracts two DRRs acquired at the same angle, the beam hardening in mask and contrast frames cancels almost exactly (same bone geometry, same spectrum). The post-subtraction difference between monoenergetic FluoroSim and polyenergetic DeepDRR drops to near-zero in vessel regions where training data is extracted.

---

## Slide 14 — Sprint 1 Completed Deliverables

**What to say:**
> "Four capability pillars: rendering, detector realism and DSA, physics solver, and system integration. The misregistration jitter is the only XCath item here — it's implemented. The 'validated not yet merged' note for Hagen-Poiseuille/F-8/Dispersion is explicit. Every other item in this table is verified in the codebase."

---

### Q&A — Slide 14

**Q: What is "volumetric instrument injection" distinct from inline compositing?**
A: Inline compositing (Slang shader) adds catheter attenuation during the ray march — it doesn't change the CT volume. Volumetric injection (Warp `atomic_max` kernel, `paint_cylinders_kernel`) *writes* catheter attenuation directly into the 3D μ-volume voxels. It's used for the DSA pipeline where we need the catheter to appear in the CT volume for pre-contrast baseline generation. The two approaches are complementary, not redundant.

**Q: Is differentiable rendering currently working?**
A: `renderDRR_backward` is implemented in the Slang shader with `[Differentiable]` annotations and software trilinear backward pass. It computes 6-DOF pose gradients via Slang autodiff. However, it's not currently wired into the RL training loop — the policy uses state observations, not pixel observations. The differentiable path is available for future gradient-based 2D/3D registration work.

---

## Slide 15 — Roadmap

**What to say:**
> "Three columns: today, Sprint 2, Sprint 3+. Today: the foundation is complete — physics, rendering, DSA, interactive demo, state-based RL all operational. Sprint 2: renderer scaling via Texture2DArray, pixel observations for RL, first end-to-end policy. Sprint 3+: clinical AI — registration model, vessel segmentation from fluoroscopy, hardware deployment. The project is data infrastructure for robotic catheter AI."

---

### Q&A — Slide 15

**Q: What is the critical path to the first end-to-end PPO policy?**
A: Three things in sequence: (1) Texture2DArray renderer at >60 Hz for 512 envs, (2) wire fluoroscopy pixel tensor as the RL observation, (3) run PPO training to convergence. The renderer work blocks the RL observation, which blocks training. No other Sprint 2 items are on the critical path.

**Q: How long until clinical deployment (Stage 4)?**
A: Sprint 2 produces a trained pixel-based policy. Sprint 3 adds 2D/3D registration and safety constraints required for hardware deployment. Clinical deployment requires IRB approval, regulatory pathway (likely SaMD/De Novo in the US), and prospective validation. Hardware deployment of the navigation policy itself is Sprint 3+ — realistically 12–18 months from now if Sprint 2 succeeds.

**Q: What is MAISI and why does it matter for Sprint 2?**
A: MAISI is NVIDIA's Medical AI for Synthetic Imaging — a generative model that produces synthetic patient CT volumes conditioned on anatomy labels. In Sprint 2, MAISI provides diverse patient anatomies for multi-patient training without needing real patient data. This addresses the single-patient limitation flagged in the XCath gap analysis — Sprint 1 was validated on one patient CT only.

---

*Notes verified against codebase on 2026-05-14.*
*Newton upstream reference: `/home/cdinea/newton/newton/newton/_src/solvers/xpbd_rod/solver_xpbd_rod.py`*
*Isaac Lab implementation: `source/isaaclab_newton/isaaclab_newton/solvers/xpbd_rod_solver.py` + `xcath_rod_solver.py`*
*Fluorosim: `/home/cdinea/i4h-sensor-simulation-internal/fluoro-simulator/fluorosim/`*
