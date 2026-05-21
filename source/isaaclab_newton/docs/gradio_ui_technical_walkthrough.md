# Gradio UI — Technical Walkthrough

**File:** `source/isaaclab_newton/examples/interactive_catheter_fluoro.py`
**Entry point:** `xcath-fluoro` (registered `console_scripts` in `setup.py`)

---

## 1. What It Is

The Gradio UI is a **single-environment, human-in-the-loop interactive simulator** for X-ray–guided catheter intervention. It connects two GPU-resident subsystems — the XPBD Cosserat rod physics solver and the Slang DRR fluoroscopy renderer — and exposes control of both through a browser-accessible web interface.

It is not an RL training loop. There is no policy, no reward, no episode reset triggered by a terminal condition. A human operator drives the catheter by clicking buttons; the physics and renderer respond in real time. This is the primary tool for:

- **Visual validation** — confirming that the physics, collision, and rendering produce clinically plausible fluoroscopy output before deploying a trained policy.
- **Human demonstration collection** — recording insertion trajectories for imitation learning.
- **Debugging** — observing the catheter bend indicator live to confirm vessel-mesh collision constraints are firing.
- **Stakeholder demos** — showing the full system end-to-end from a browser with no Isaac Sim window required.

---

## 2. Startup and Package Resolution

The script is designed to work whether packages are `pip install -e` installed or run from a bare checkout. The `_ensure_importable` helper implements a two-stage resolution:

```python
def _ensure_importable(package: str, *candidate_dirs: str) -> None:
    if importlib.util.find_spec(package) is not None:
        return                             # already on sys.path — done
    for d in candidate_dirs:
        if Path(d).is_dir():
            sys.path.insert(0, d)
            if importlib.util.find_spec(package) is not None:
                return
    raise ImportError(...)
```

It checks `importlib.util.find_spec` first — if the package is already importable (installed via `pip install -e`), no `sys.path` modification happens. Only if the import fails does it try the candidate directory list. This means the same script works in a clean pip-installed environment and in a developer checkout with no install step.

The two resolved packages:
- `isaaclab_newton` — physics solver, collision, rod data
- `fluorosim` — Slang DRR renderer, vessel mesh extraction

---

## 3. Simulation Initialisation — `init_simulation()`

This function runs once at startup and populates the global `_sim` dictionary, which holds all long-lived GPU objects for the session lifetime. The order of operations matters:

### 3.1 CT Volume Load

```python
mu_zyx = np.load(os.path.join(CT_DIR, 'mu_volume.npy'))     # (Z, Y, X) float32
meta   = json.load(open(os.path.join(CT_DIR, 'metadata.json')))
```

The μ-volume is a pre-processed float32 array in Z-Y-X order. Voxel spacing and origin are read from `metadata.json` (`spacing_zyx_mm`, `origin_xyz_mm`). The volume is passed directly to the renderer constructor, which uploads it to GPU as a `Texture3D<float>` in the Slang shader.

### 3.2 Vessel Mesh Construction

```python
mask, sp_ds, origin_xyz, (cx_mm, cy_mm, cz_mm) = \
    _build_vessel_mask_downsampled(mu_zyx, meta)
mesh = extract_vessel_mesh(mask, spacing_zyx_mm=sp_ds,
                           origin_xyz_mm=origin_xyz, device='cuda')
```

`_build_vessel_mask_downsampled` generates a synthetic elliptical tube mask at 4× downsampled resolution (one-time cost, ~10 ms). The tube follows a sinusoidal path in Z to simulate a curved vessel. `extract_vessel_mesh` converts the binary mask to a `wp.Mesh` (Warp triangle mesh) via Marching Cubes — this mesh is used for physics collision, not for rendering. The mesh lives entirely on GPU.

### 3.3 Physics Solver Construction

```python
solver = XCathRodSolver(
    rod_cfg,
    collision_mesh=mesh,
    track_start=track_start_m,
    track_dir=np.array([1., 0., 0.], dtype=np.float32),
    track_length=rod_len_m,
    tip_num_edges=6,
    particle_radius=0.001,
    segment_length=seg_len,
    collision_iterations=2,
    sign_scale=1.0, target_phi=-0.001, max_dist=0.025,
    initial_height=float(track_start_m[2]),
)
```

`XCathRodSolver` is a subclass of `XPBDRodSolver` that adds two collision hooks:
- `_pre_constraints_hook` — AABB broadphase query against `wp.Mesh` to find candidate vessel-wall edges
- `_post_constraints_hook` — SDF containment constraint push particles back inside the vessel lumen

`track_start` and `track_dir` define the proximal insertion axis — the catheter enters along the X axis at the vessel centerline. `initial_height` pins the starting Z coordinate so the rod is initialised inside the vessel.

**Key config values (verified from `init_simulation`):**
| Parameter | Value | Meaning |
|---|---|---|
| `num_segments` | 20 | 21 particles, 20 edges |
| `num_substeps` | 8 | Substeps per `step()` call |
| `particle_radius` | 0.001 m | 1 mm collision sphere radius |
| `collision_iterations` | 2 | SDF projection passes per substep |
| `rod_length` | derived from CT X-extent | ~100–150 mm |

### 3.4 Renderer Construction

```python
renderer = SlangDiffDRRRenderer(
    mu_zyx,
    spacing_zyx_mm=(sz_mm, sy_mm, sx_mm),
    cfg=SlangDiffDRRConfig(det_width_px=DET_SIZE, det_height_px=DET_SIZE,
                           step_mm=1.0, i0=1.0),
    num_envs=1,
)
```

`DET_SIZE=256` — the interactive UI renders at 256×256 for ~3.8 ms/frame (~263 FPS render budget). `step_mm=1.0` sets the ray march step size. The renderer constructor uploads the μ-volume to GPU as a `Texture3D` and initialises the Slang shader. `num_envs=1` — the interactive UI uses single-environment rendering; the batched path (`num_envs=512`) is for RL training only.

### 3.5 Initial State Snapshot

```python
_sim['initial_pos'] = solver.positions.cpu().numpy().copy()
_sim['initial_ori'] = solver.orientations.cpu().numpy().copy()
```

The solver's initial particle positions and orientations are snapshotted to CPU NumPy arrays at construction. These are used by `do_reset()` to restore the solver to its initial state without re-constructing it. The copy is explicit because `solver.positions` returns `wp.to_torch(...).clone()` — a fresh tensor each call, not a view into the Warp buffer.

---

## 4. Per-Frame Execution — `_step_and_render()`

Every button click calls `_step_and_render(velocity, torque, proj_name, steps)`. This is the hot path:

```python
for _ in range(steps):
    solver.apply_proximal_control(velocity, torque, DT)
    solver.step(DT)
if _CUDA_AVAILABLE:
    torch.cuda.synchronize()

img = _render(proj_name)
```

**Step count per action:**
| Action | `steps` | Reason |
|---|---|---|
| Advance / Retract | 3 | 3 physics substep groups = 3 × 8 = 24 XPBD substeps |
| Rotate CW / CCW | 2 | Torque-only; fewer steps needed to see rotation |
| Idle | 1 | Gravity deformation; minimal step needed |

`apply_proximal_control(velocity, torque, dt)` writes insertion speed and rotation rate to the solver's root particle before each `step()`. Inside `step()`, the CUDA graph replays 8 substeps × 12 kernels in a single `wp.capture_launch` call. The `torch.cuda.synchronize()` ensures the GPU has finished before the render call reads particle positions.

---

## 5. Coordinate System — `_pos_to_vol_mm()`

The physics solver operates in **metric world space** (metres, origin at `track_start_m`). The renderer expects positions in **CT volume coordinates** (millimetres, origin at CT volume origin). The conversion:

```python
def _pos_to_vol_mm(pos_m: np.ndarray):
    pos_ct_mm  = (pos_m - _sim['local_z0_m'] + _sim['ct_offset_m']) * 1000.0
    pos_vol_mm = pos_ct_mm - _sim['ct_origin_mm']
    return pos_vol_mm, pos_ct_mm
```

**Stage 1 — physics world → CT world (mm):**
Subtract the physics origin (`local_z0_m`), add the CT anchor (`ct_offset_m`), scale from metres to millimetres.

**Stage 2 — CT world (mm) → CT volume coordinates (mm):**
Subtract the CT volume origin (`ct_origin_mm`) from `metadata.json`.

This two-stage conversion ensures that when the physics solver places the catheter at `track_start_m`, the renderer sees the catheter centred at the correct location in the CT volume — not at the world origin.

---

## 6. Rendering — `_render()` and `_render_dsa()`

### 6.1 Standard Fluoroscopy — `_render()`

```python
cat = CatheterSegmentData(positions=pos_vol_mm, radii=CATHETER_R, mu_values=CATHETER_MU)
img = _sim['renderer'].render_batch_with_catheter(
    PROJECTIONS[proj_name], _TRANS_ZERO, [cat])[0]
```

One Slang dispatch over a 256×256 grid. The ray march integrates:
```
μ_total(s) = μ_CT(s) + Σᵢ CATHETER_MU · √(1 − dᵢ²/CATHETER_R²)
I_final = exp(−∫ μ_total(s) ds)
```

`CATHETER_R = 1.8 mm` (wire radius), `CATHETER_MU = 0.50 mm⁻¹` (NiTi shaft). The catheter appears as a darkening on the DRR — not an opaque overlay.

The four C-arm projections are pre-built as `(1, 3)` Euler-angle arrays:

| Label | Rotation (Y-axis) | Clinical meaning |
|---|---|---|
| AP (0°) | 0 rad | Anteroposterior — standard frontal view |
| LAO-45 | +45° | Left anterior oblique — standard neuro view |
| Lateral (90°) | +90° | True lateral |
| RAO-30 | −30° | Right anterior oblique |

The output is converted to a `PIL.Image` via a direct NumPy stack (`np.stack([arr, arr, arr], axis=-1)`) — no intermediate grayscale conversion, no extra PIL copy.

### 6.2 DSA Frame — `_render_dsa()`

The DSA mode fires **three separate Slang dispatches** per frame:

```python
bg_drr  = renderer.render_batch(rot, _TRANS_ZERO)[0]                  # Dispatch 1
fat_drr = renderer.render_batch_with_catheter(..., DSA_FAT_R, ...)    # Dispatch 2
fluoro  = renderer.render_batch_with_catheter(..., CATHETER_R, ...)   # Dispatch 3
```

| Dispatch | Catheter radius | μ | Purpose |
|---|---|---|---|
| 1 — Background | None | — | Anatomy only (mask frame) |
| 2 — Fat catheter | 2.5 mm | 0.80 mm⁻¹ | Over-sized footprint marks vessel lumen |
| 3 — Actual catheter | 1.8 mm | 0.50 mm⁻¹ | True wire for live fluoro panel |

**Vessel lumen signal extraction:**
```python
signal  = clip(fat_drr - bg_drr, 0, None)       # positive where fat catheter added attenuation
dsa_raw = sqrt(signal / signal.max())            # sqrt for contrast stretch
```

The fat-catheter dispatch (radius 2.5 mm) projects a footprint larger than the actual wire. Subtracting the background DRR isolates the fat-catheter's additional attenuation signal — which approximates the vessel lumen width. The sqrt stretch enhances low-signal regions for visibility.

**Output compositing:**
```python
g_ch = clip(bg_i16 + boost * 160, 0, 255)       # green: boost where vessel lumen signal is high
r_ch = b_ch = clip(bg_i16 - boost * 80, 0, 255) # red+blue: dampen for green tint
```

The right panel (blue header bar) is the live fluoroscopy with actual catheter wire. The left panel (green header bar) is the DSA roadmap with vessel lumen highlighted in green. Both share the same C-arm rotation matrix — they are spatially registered.

---

## 7. Catheter Bend Indicator — `_catheter_bend_mm()`

```python
def _catheter_bend_mm(pos_vol_mm: np.ndarray) -> float:
    start, end = pos_vol_mm[0], pos_vol_mm[-1]
    axis = end - start
    axis_unit = axis / ||axis||
    vecs      = pos_vol_mm - start
    proj      = outer(dot(vecs, axis_unit), axis_unit)   # projections onto axis
    perp_dist = norm(vecs - proj, axis=1)                # perpendicular distances
    return max(perp_dist)
```

This computes the maximum perpendicular deviation of any particle from the straight line connecting the catheter's two endpoints — geometrically, the maximum sagitta of the rod. 

- A straight, unconstrained catheter returns ~0 mm (numerical noise only)
- Active vessel-wall deflection produces 5–30 mm
- The info box shows `← vessel wall deflecting rod` when `bend_mm > 2.0`

This is the primary live diagnostic for confirming that `XCathRodSolver`'s SDF containment constraints are firing correctly — without it, there is no visual indication from the image alone whether collision is active or not.

---

## 8. Reset — `do_reset()`

The Reset action must bypass the solver's public API and write directly into the Warp GPU buffers:

```python
ws = solver._ws

wp.to_torch(ws.positions).copy_(init_pos)
wp.to_torch(ws.predicted_positions).copy_(init_pos)
wp.to_torch(ws.velocities).zero_()
wp.to_torch(ws.orientations).copy_(init_ori)
wp.to_torch(ws.predicted_orientations).copy_(init_ori)
# ...

solver.reset_cuda_graph()
torch.cuda.synchronize()
```

`solver.positions` is a property that returns `wp.to_torch(ws.positions).clone()` — a detached copy. Writing to that clone is a no-op against the actual GPU allocation. Reset must call `wp.to_torch(ws.positions).copy_(...)` to write into the underlying Warp buffer in-place.

After writing the initial state, `solver.reset_cuda_graph()` invalidates the captured CUDA graph. The CUDA graph was recorded at the original particle configuration; after Reset changes the positions, the next `step()` re-captures the graph from the new state. Without this call, the graph would replay with stale captured values for any kernel that reads particle positions as compile-time constants.

---

## 9. Info Box Telemetry

Every action updates the info box with six fields:

```
Projection   : LAO-45
Tip (CT mm)  : X=87.4  Y=51.2  Z=63.1
Catheter bend: 12.3 mm  ← vessel wall deflecting rod
Physics step : 4.2 ms  (3 substep(s))
Render (GPU) : 3.8 ms
Sim loop     : 8.0 ms  (~125 fps)
Frame #      : 47
```

All timing is wall-clock via `time.perf_counter()`. Physics timing includes the `torch.cuda.synchronize()` barrier. Render timing is measured inside `_render()` around the single `render_batch_with_catheter` call. `Sim loop` is their sum — this is the true per-click latency for the human operator.

---

## 10. Why the Gradio UI Is Different from the PPO State Policy

These two components use the same physics solver and the same renderer, but they differ in **who generates actions, how many environments run simultaneously, and what the observation is**.

### 10.1 Control authority

| | Gradio UI | PPO State Policy (`CatheterStateEnv`) |
|---|---|---|
| Action source | Human button click | Neural network output `(512, 2)` tensor |
| Action per frame | One `(velocity, torque)` pair | 512 simultaneous `(push_vel, rot_vel)` pairs |
| Action space | Discrete buttons (fixed velocity/torque values) | Continuous `Box(2)` normalised to `[-1, 1]` |
| Control frequency | Event-driven (human click rate ~1–5 Hz) | Fixed 60 Hz per GPU step |

### 10.2 Environments

| | Gradio UI | PPO State Policy |
|---|---|---|
| `num_envs` | 1 | 512 |
| Physics workspace | Single-env `_RodWorkspace` | Batched `_BatchedWorkspace` (flat buffers) |
| CUDA graph | Captures single-env substep loop | Captures 512-env substep loop (96 kernels) |
| Renderer | `num_envs=1`, 256×256 | `num_envs=512`, 512×512 (Sprint 2) |

### 10.3 Observations

The PPO environment (`CatheterStateEnv`) constructs a flat numerical observation vector and never calls the renderer during training:

```python
# observation = concat of:
# - all segment positions  (N_seg * 3)  — from physics workspace
# - tip position           (3)
# - tip velocity           (3)
# - target position        (3)
# - insertion depth        (1)
# Total obs_dim = 20*3 + 3 + 3 + 3 + 1 = 70 floats per env
```

The policy reads 70 floats, outputs 2 floats. No image is generated during training. The renderer is only invoked at evaluation time to produce visualisations.

The Gradio UI uses no observation vector at all — the human is the "policy." The only feedback is the fluoroscopy image and the info box text.

### 10.4 Episode management

The PPO environment has explicit episode termination:
```python
terminated = tip_distance_to_target < threshold      # success
truncated  = episode_length_buf >= max_episode_steps # timeout
```
When any environment terminates, it is reset automatically mid-batch and the new observation is returned for the next policy inference step.

The Gradio UI has no episode concept. The simulation runs indefinitely until the user clicks Reset. There is no reward, no terminal state, no environment reset triggered by the code.

### 10.5 Summary table

| Dimension | Gradio UI | PPO Policy (`CatheterStateEnv`) |
|---|---|---|
| Purpose | Human validation, demos, demonstration collection | Autonomous policy training |
| Action source | Human button click | Neural network |
| Environments | 1 | 512 |
| Observation | None (human sees image) | 70-float state vector |
| Renderer called | Every frame | Never during training |
| Episode management | None (manual Reset) | Automatic terminal + reset |
| Collision | `XCathRodSolver` (SDF + AABB + track) | `RodSolver` (no vessel mesh — Sprint 2) |
| Physics backend | `XPBDRodSolver`, single-env | `RodSolver`, batched multi-env |
| Output | `PIL.Image` → Gradio browser | `(obs, reward, done)` → RSL-RL PPO |

The Gradio UI and the PPO environment are not alternative implementations of the same thing — they are complementary tools. The interactive UI validates the physics and rendering pipeline that the RL environment will eventually train inside. A policy trained in `CatheterStateEnv` can be exported and its actions replayed through the Gradio UI for human-interpretable evaluation.

---

## 11. Live Demo Walkthrough — Presenting the UI to an Audience

This section is a step-by-step presenter's guide. Each paragraph describes what to say, what to click, and what is happening in the unified sim loop underneath.

---

### 11.1 Opening — Orient the Audience

**What to say:**
> "What you are looking at is a real-time X-ray simulation of a catheter being navigated inside a patient's skull. The background is a Digitally Reconstructed Radiograph — a synthetic X-ray computed from a real CT scan using the Beer-Lambert attenuation law. The catheter is being simulated by a GPU physics solver, and every frame you see is rendered in real time by a Slang GPU shader. There is no pre-recorded animation here — the physics and the rendering are both live."

**What the audience sees:**
The initial fluoroscopy image in LAO-45 projection — grayscale skull anatomy with the catheter wire visible as a dark thin line inside the cranium.

**What is running underneath:**
At startup, `init_simulation()` has already loaded the CT μ-volume into GPU memory as a `Texture3D<float>`, built the vessel collision mesh from the CT mask, initialised the XPBD rod solver with 20 segments and 8 substeps, and fired one warm-up render. The sim is idle — no physics steps are running. GPU objects are fully constructed and waiting.

---

### 11.2 C-arm Projection Dropdown

**What to say:**
> "The C-arm projection dropdown changes the viewing angle — just like rotating a real fluoroscopy C-arm arm around the patient. AP is the standard frontal view. LAO-45 is left anterior oblique at 45 degrees, which is the standard view for neuro-interventional procedures. Lateral gives you a 90-degree side view, and RAO-30 is right anterior oblique. Notice that the catheter and the skull anatomy all change perspective consistently — there is no pre-rendered image for each view. The same ray-march kernel re-executes at the new C-arm angle."

**What to click:**
Change the dropdown from `LAO-45` to `AP (0°)`, then to `Lateral (90°)`.

**What is running underneath:**
`do_change_view(proj, speed)` calls `_render(proj_name)` directly — no physics step is executed. The renderer's `render_batch_with_catheter` fires a new Slang dispatch with the updated rotation matrix `PROJECTIONS[proj_name]`, which is a `(1, 3)` Euler-angle array. The Slang shader recomputes the X-ray source position and detector orientation from this matrix, re-marches all 256×256 rays through the same static CT volume, and re-tests each ray sample point against the same catheter segment buffer. The output changes purely because the ray directions changed — no data was reloaded, no CPU computation happened.

---

### 11.3 Advance Button — Pushing the Catheter

**What to say:**
> "When I click Advance, the catheter moves forward along its insertion axis. Let me show you what happens in the unified sim loop each time this button is pressed."

**What to click:**
Click `Advance` three or four times in succession. Watch the tip position in the info box update.

**What the audience sees:**
The catheter wire moves forward along the vessel path. The tip position values in the Simulation Info box update with each click. Physics step and render times are visible.

**What is running underneath — step by step:**
Each click calls `do_advance(proj, speed)` → `_step_and_render(velocity=speed/1000, torque=0.0, proj, steps=3)`.

Inside that function, the **unified sim loop** executes 3 times:

**Step 1 — Root Control (GPU kernel):**
```python
solver.apply_proximal_control(velocity, 0.0, DT)
```
`apply_proximal_control` writes the insertion velocity (e.g. 5 mm/s → 0.005 m/s) into the solver's `push_velocity` scalar, then launches `_xr_proximal_push_kernel` at `dim=1`. That kernel reads the current root particle position, computes the live tangent direction by differencing the root and second particle, and displaces the root particle along that tangent by `velocity × DT`. No CPU waits — the kernel runs and returns immediately.

**Step 2 — Physics Step (CUDA graph replay):**
```python
solver.step(DT)
```
On the first call after startup, the CUDA graph is captured — the solver records 8 substeps × 12 Warp kernel launches = 96 kernel dispatches. Every subsequent `step()` call replays this graph via a single `wp.capture_launch()` — ~1 µs of CPU overhead. The 96 kernels execute on GPU: predict positions, evaluate Cosserat stretch + Darboux constraints, assemble the JMJT matrix, run the block-Thomas direct solve, apply corrections, integrate. **At each substep, the `_pre_constraints_hook` runs an AABB broadphase query against the vessel mesh**, and the `_post_constraints_hook` applies SDF containment — pushing any particle that has left the vessel lumen back inside.

**Step 3 — Segment Buffer Update:**
```python
pos_vol_mm, pos_ct_mm = _pos_to_vol_mm(solver.positions.cpu().numpy())
cat = CatheterSegmentData(positions=pos_vol_mm, radii=CATHETER_R, mu_values=CATHETER_MU)
```
After the physics step, the 21 particle positions are read from GPU (`wp.to_torch`) to CPU NumPy, converted from physics-world metres to CT volume millimetres via the two-stage coordinate transform, and packed into a `CatheterSegmentData` struct. This is the only CPU-GPU data transfer in the hot path — 21 × 3 × 4 bytes = 252 bytes per frame.

**Step 4 — Render Call (Slang DRR):**
```python
img = renderer.render_batch_with_catheter(PROJECTIONS[proj_name], _TRANS_ZERO, [cat])[0]
```
One Slang dispatch over a 256×256 grid. Each of the 65,536 threads traces one ray. At every sample point along the ray, the thread tests its 3D position against each catheter segment: computes perpendicular distance `d`, checks `d < CATHETER_R`, adds `CATHETER_MU × √(1 − d²/r²)` to the running attenuation integral if inside. The CT μ-value at that 3D sample point is fetched from the `Texture3D` and also added to the integral. After all steps, `I = exp(−integral)` is written to the output buffer.

**Step 5 — Output:**
The output float32 array is clipped, scaled to uint8, stacked into RGB, and wrapped in a `PIL.Image`. Gradio receives the image and updates the browser frame. The info box is updated with tip position (in CT mm), bend metric, physics time, render time, and total sim loop time.

> "The entire round trip — root control, 96 kernel physics solve, coordinate conversion, ray march, image update — takes about 8 milliseconds. That is ~125 frames per second of simulation capacity. The human clicking rate is the actual bottleneck, not the simulator."

---

### 11.4 Retract Button

**What to say:**
> "Retract does the same thing as Advance but with a negative insertion velocity. The physics solver pushes the root particle backward along the tangent direction. The vessel collision constraints continue to apply — the catheter does not pass through vessel walls during retraction either."

**What to click:**
Click `Retract` twice to pull the catheter back, then `Advance` again to re-advance.

**What is running underneath:**
Identical to Advance except `velocity = −speed/1000`. The same 3 physics steps + 1 render execute. The CUDA graph is replayed unchanged — the only difference is the value written into `push_velocity` before each `step()`.

---

### 11.5 Rotate CW / CCW Buttons

**What to say:**
> "The Rotate buttons apply a torque at the proximal end — the catheter root. This is how a clinical cardiologist steers a catheter: by torquing the proximal end outside the patient's body, the distal tip rotates inside the vessel. The torque propagates along the Cosserat rod model as a torsional wave."

**What to click:**
Click `Rotate CW` several times. Watch the catheter tip change orientation in the image. Then click `Rotate CCW` to reverse.

**What is running underneath:**
`do_rotate_cw` calls `_step_and_render(velocity=0.0, torque=+0.015, proj, steps=2)`. `apply_proximal_control(0.0, 0.015, DT)` launches `_xr_proximal_push_kernel` with zero insertion velocity but non-zero torque — the kernel applies an incremental quaternion rotation `q_delta = quat_from_axis_angle(tangent, torque × DT)` to the root particle's orientation, updating `ws.orientations[root]`, `ws.predicted_orientations[root]`, and `ws.prev_orientations[root]`. The XPBD Darboux (torsion) constraints in the subsequent `step()` propagate this orientation change from root to tip over the 8 substeps.

---

### 11.6 Advance Speed Slider

**What to say:**
> "The speed slider controls how fast the catheter advances per click — from 1 mm/s to 20 mm/s. At 20 mm/s the catheter moves aggressively and you will see it deflect more strongly off vessel walls. At 1 mm/s it moves slowly enough to observe the elastic deformation in detail."

**What is running underneath:**
The slider value is passed as `speed` to `do_advance(proj, speed)`, which divides it by 1000 to convert mm/s to m/s: `velocity = float(speed) / 1000`. The physics time step is fixed at `DT = 1/30 s`, so at 20 mm/s the root particle moves `20/1000 × 1/30 ≈ 0.67 mm` per substep group. At 1 mm/s it moves 0.033 mm. The XPBD solver handles both — stiffness and collision constraints are scale-independent.

---

### 11.7 Catheter Bend Indicator

**What to say:**
> "The Catheter bend value in the info box is the most important diagnostic in this demo. It measures the maximum perpendicular deviation of any point on the catheter from the straight line connecting its two endpoints — in millimetres. A value near zero means the catheter is straight and unconstrained. A value above 2 mm confirms that the vessel-mesh collision constraints are actively deflecting the rod. This is how we know, without looking at the image, that the physics collision is working."

**What to demonstrate:**
Advance the catheter until the bend indicator rises above 2.0 mm. The info box will read `← vessel wall deflecting rod`. Point to this in the info box.

**What is running underneath:**
`_catheter_bend_mm(pos_vol_mm)` runs on CPU after each `_render()` call. It projects all 21 particle positions onto the axis between tip and root, computes the perpendicular residual distance for each, and returns the maximum. It adds ~0.1 ms to the frame time (21 NumPy dot products). The threshold of 2.0 mm is conservative — even small vessel-wall contacts produce deflections of 5–15 mm at typical catheter stiffness.

---

### 11.8 Idle Step Button

**What to say:**
> "The Idle Step button applies zero control — no insertion, no rotation — and runs one physics step. This lets gravity act on the catheter. If the catheter is unsupported, it will sag downward. If it is inside a vessel, the collision constraints will hold it in place against gravity. This is a useful test: click Idle Step repeatedly and watch whether the catheter stays in position or drifts."

**What is running underneath:**
`do_idle` calls `_step_and_render(0.0, 0.0, proj, steps=1)`. One `apply_proximal_control(0.0, 0.0, DT)` with both values zero, followed by one `step(DT)`. The CUDA graph replays with zero control input — gravity (`[0, 0, −9.81]` m/s²) is baked into the `_xr_predict_pos_batched` kernel which applies `v += gravity × DT` each substep.

---

### 11.9 Reset Button

**What to say:**
> "Reset restores the catheter to its initial straight position. This is not just setting a flag — it writes new particle position and orientation data directly into the GPU memory buffers. Then it invalidates the CUDA graph, because the graph was captured at the original particle state. The next physics step re-captures the graph from the restored state. After Reset, the simulation is back to frame zero."

**What to click:**
Click `Reset`. The catheter snaps back to its initial position. The frame counter resets to 0.

**What is running underneath:**
```
do_reset() →
  wp.to_torch(ws.positions).copy_(initial_pos)          # overwrite GPU buffer
  wp.to_torch(ws.predicted_positions).copy_(initial_pos)
  wp.to_torch(ws.velocities).zero_()
  wp.to_torch(ws.orientations).copy_(initial_ori)
  solver.reset_cuda_graph()                              # force re-capture on next step()
  torch.cuda.synchronize()
  _render(proj)                                          # one render of the clean state
```
The `solver.positions` property returns a `.clone()` of the Warp buffer — writing to it would have no effect on the actual GPU allocation. Reset bypasses the property and writes into `ws.positions` directly via `wp.to_torch(...).copy_()`. This is intentional — the property's clone behaviour is correct for read paths but must not be used for write paths.

---

### 11.10 Show DSA Frame Button

**What to say:**
> "The DSA button shows a Digital Subtraction Angiography frame. In a real clinical procedure, DSA is how you see the blood vessels — you inject iodine contrast, subtract a pre-injection mask frame, and the vessel tree lights up. We simulate this with three separate GPU ray-march dispatches. On the left panel you see the vessel lumen highlighted in green — that is the DSA roadmap. On the right panel you see the live fluoroscopy with the catheter wire. Both panels share the same C-arm angle so they are spatially registered."

**What to click:**
Click `Show DSA Frame`. Point to the green highlighted region on the left panel and the catheter wire on the right.

**What is running underneath:**
`do_dsa()` calls `_render_dsa(proj)`, which fires three Slang dispatches in sequence:

1. **Background dispatch** — no catheter, renders skull anatomy only. This is the DSA mask.
2. **Fat-catheter dispatch** — catheter with radius `DSA_FAT_R = 2.5 mm` and `DSA_FAT_MU = 0.80 mm⁻¹`. The oversized cylinder footprint covers the vessel lumen around the catheter wire.
3. **Actual catheter dispatch** — catheter with radius `CATHETER_R = 1.8 mm` and `CATHETER_MU = 0.50 mm⁻¹`. This is the live fluoro panel.

The vessel signal is computed as `clip(fat_drr − background_drr, 0, None)` — the difference between the fat-catheter render and the anatomy-only render isolates the region the fat catheter covered. A square-root stretch enhances low-signal areas. The result is composited as a green tint (`g_ch = bg + boost × 160`, `r_ch = b_ch = bg − boost × 80`) and stacked side-by-side with the live fluoro.

> "Total render time for DSA is roughly 3× the single-view render time — three dispatches at ~3.8 ms each = ~11 ms. You can see this in the info box."

---

### 11.11 Closing — Connect to RL Training

**What to say:**
> "Everything you just saw — the physics, the collision, the rendering — is exactly what runs inside the RL training environment. The difference is that instead of one catheter controlled by a human clicking buttons, we run 512 catheters simultaneously on GPU, each controlled by a neural network policy. The policy reads a 70-float state vector — tip position, segment positions, target position, insertion depth — and outputs push and rotate velocities. The same Beer-Lambert ray march that produced these images is what the policy will eventually observe as pixel-based observations. Today the policy trains on the state vector; pixel observations are the Sprint 2 target."

**What this demonstrates technically:**
The unified sim loop — root control, CUDA graph physics, segment buffer update, Slang render call — is the same code path whether `num_envs=1` with human control or `num_envs=512` with a policy. Scaling from the demo to training is a parameter change, not an architectural change.
