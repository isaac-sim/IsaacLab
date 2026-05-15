# Interactive X-Ray Fluoroscopy Catheter Simulator

A browser-accessible simulation environment that runs the full unified physics + rendering loop on demand, one step at a time, driven by human button presses.

---

## Setup (one time per environment)

The simulator is part of the `isaaclab_newton` package. Install it together with the `fluorosim` renderer:

```bash
# 1. Isaac Lab physics + simulator extension
pip install -e /path/to/IsaacLab/source/isaaclab_newton

# 2. Fluoro-simulator renderer (declared as a dependency of isaaclab_newton)
pip install -e /path/to/i4h-sensor-simulation-internal/fluoro-simulator[all]
```

`fluorosim` is listed in `isaaclab_newton`'s `install_requires`, so both installs are required once. After that, no `PYTHONPATH`, no `cd`, no hardcoded paths.

If the packages are **not** pip-installed the script auto-detects them relative to its own file location:

- `isaaclab_newton` → one directory up from `examples/`
- `fluorosim` → sibling repo at `../../i4h-sensor-simulation-internal/fluoro-simulator`

If neither succeeds, the script raises an `ImportError` with the exact `pip install -e` command to run.

---

## CT Data

The renderer requires a pre-processed CT volume. The data directory must contain:

```
<ct-dir>/
├── mu_volume.npy      # (Z, Y, X) float32 linear attenuation coefficients
└── metadata.json      # {"spacing_zyx_mm": [...], "origin_xyz_mm": [...]}
```

Default location is `/tmp/patient_001`. Override with `--ct-dir`.

---

## How to Run

After `pip install -e`, a `xcath-fluoro` command is registered in your environment:

```bash
conda activate isaaclab

# default CT dir (/tmp/patient_001), default port (7860)
xcath-fluoro

# custom CT directory
xcath-fluoro --ct-dir /data/ct/patient_042

# custom port (e.g. if 7860 is in use)
xcath-fluoro --port 8080
```

You can also invoke the script directly without the installed command:

```bash
python3 source/isaaclab_newton/examples/interactive_catheter_fluoro.py [--ct-dir PATH] [--port PORT]
```

Then open **http://localhost:7860** (or the port you chose) in any browser.

If the server is remote, forward the port over SSH first:

```bash
ssh -L 7860:localhost:7860 <your-server>
```

---

## Unified Simulation Loop — Architecture

The unified loop is a strict 5-step cycle executed once per UI action (button click). It is architecturally identical to the RL training loop — the only difference is that human input replaces the policy network.

```
┌──────────────────────────────────────────────────────────────────┐
│                    UNIFIED SIMULATION LOOP                       │
│                                                                  │
│  1. CONTROL INPUT   Human button → push_velocity / torque        │
│         │                                                        │
│         ▼                                                        │
│  2. ROOT KINEMATIC  apply_proximal_control(v, τ, dt)             │
│         │            moves root particle (inv_mass = 0)          │
│         ▼                                                        │
│  3. PHYSICS STEP    solver.step(dt)                              │
│         │            XPBD substep loop (8 substeps, CUDA graph)  │
│         │            → positions[], orientations[] updated       │
│         ▼                                                        │
│  4. COORDINATE XFM  positions (m) → CT volume space (mm)        │
│         │                                                        │
│         ▼                                                        │
│  5. GPU RENDER      render_batch_with_catheter(rot, trans, cat)  │
│                      Slang Beer-Lambert ray march                │
│                      → 256×256 float32 DRR image                 │
└──────────────────────────────────────────────────────────────────┘
```

### Step 1 — Control Input

Each Gradio button maps to a `(push_velocity, torque)` pair:

| Button | push_velocity | torque | substeps |
|---|---|---|---|
| Advance | +speed/1000 m/s | 0 | 3 |
| Retract | −speed/1000 m/s | 0 | 3 |
| Rotate CW | 0 | +0.015 rad/s | 2 |
| Rotate CCW | 0 | −0.015 rad/s | 2 |
| Idle | 0 | 0 | 1 |

### Step 2 — Root Kinematic Control

`apply_proximal_control(push_velocity, torque, dt)` displaces the root particle (particle index 0) along the current rod tangent direction by `push_velocity × dt` metres. Because the root has `inv_mass = 0`, the XPBD solver never moves it — kinematic updates are the only way to advance it.

The tangent direction is computed live from positions[1] − positions[0] at each call, so the insertion direction automatically follows the rod's current proximal orientation.

### Step 3 — Physics Step (XPBD with CUDA Graph)

`solver.step(dt)` runs the eXtended Position-Based Dynamics loop:

```
for each substep:
    predict positions (explicit Euler)
    assemble JMJT  (Cosserat stretch + bend + twist + collision constraints)
    direct solve   (block-Thomas tridiagonal, O(N))
    apply position corrections
    update velocities
```

On the first call the substep loop is captured into a **CUDA graph** (`wp.ScopedCapture`). All subsequent calls replay the graph with near-zero CPU overhead. The graph is invalidated (and re-captured on the next step) only when `reset_cuda_graph()` is called explicitly — which the Reset button does after writing new initial positions.

### Step 4 — Coordinate Transform

Rod positions come out of the solver in physics-world metres. The renderer expects positions in **CT volume space** (millimetres, origin at CT volume corner). The transform is:

```python
pos_ct_mm  = (pos_m - local_z0_m + ct_offset_m) * 1000.0
pos_vol_mm = pos_ct_mm - ct_origin_mm
```

Where:
- `ct_offset_m` — world-frame position of the vessel entry point (from vessel mask centroid)
- `local_z0_m` — the solver's initial root position in world coordinates
- `ct_origin_mm` — the CT volume's physical origin (from `metadata.json`)

This registration ensures the catheter geometry and the CT anatomy are in the same coordinate frame for the ray march.

### Step 5 — GPU Ray March (Slang Beer-Lambert)

`render_batch_with_catheter(rot, trans, [catheter])` fires a single Slang GPU dispatch with thread count `(W, H, N)` = `(256, 256, 1)`. Each thread traces one ray through the CT μ-volume and accumulates attenuation from both the volume and the catheter geometry:

```
μ_total(s) = μ_CT(s) + Σᵢ μᵢ · sqrt(1 − dᵢ²/rᵢ²)
I = I₀ · exp(−∫ μ_total(s) ds)
```

where `dᵢ` is the perpendicular distance from sample point `s` to segment `i`'s axis, and `rᵢ` is the segment radius. The `sqrt(1 − d²/r²)` factor is the chord-length weight — the fraction of the cylinder diameter the ray traverses at offset `dᵢ`. This is a physically exact Beer-Lambert composite: the catheter attenuates the ray in proportion to its material thickness, not as a 2D overlay.

---

## DSA (Digital Subtraction Angiography) Path

The DSA button fires three GPU dispatches in sequence, all sharing the same C-arm rotation matrix to guarantee spatial registration:

| Dispatch | Catheter | Output |
|---|---|---|
| 1 — background | none | `bg_drr` — anatomy only |
| 2 — fat catheter | radius = 2.5 mm, μ = 0.80 | `fat_drr` — anatomy + lumen highlight |
| 3 — actual wire | radius = 1.8 mm, μ = 0.50 | `fluoro` — anatomy + catheter wire |

Vessel lumen signal:

```python
signal  = clip(fat_drr − bg_drr, 0, ∞)   # positive at fat-catheter footprint
dsa_raw = sqrt(signal / signal.max())      # sqrt gamma to boost faint edges
```

The fat catheter acts as a virtual contrast agent: it occupies the vessel lumen cross-section and adds X-ray attenuation exactly where iodine contrast would appear. The difference `fat_drr − bg_drr` isolates the vessel projection without requiring a real contrast injection or a texture swap.

The final output is a side-by-side composite:

```
┌────────────────────────┬────────────────────────┐
│  DSA PANEL (green bar) │ FLUORO PANEL (blue bar) │
│  Skull + vessel lumen  │  Skull + catheter wire  │
│  highlighted in green  │  as dark Beer-Lambert   │
│                        │  absorption shadow      │
└────────────────────────┴────────────────────────┘
```

---

## Performance (A6000, 256×256 detector, 20-segment rod)

| Stage | Time |
|---|---|
| Physics step (3 substeps, CUDA graph replay) | ~25 ms |
| GPU ray march render | ~3–4 ms |
| DSA (3 dispatches) | ~14–15 ms |
| End-to-end sim loop per click | ~28–29 ms (~35 fps) |
| Reset | ~1 ms |

Physics dominates (8 XPBD substeps per step call) because the solver re-solves the full tridiagonal system every substep. The GPU render is ~3–4 ms and is consistent with texture bandwidth at 256² resolution.

---

## Key Source Files

| File | Role |
|---|---|
| `setup.py` | Package install; declares `fluorosim` dependency and registers `xcath-fluoro` entry point |
| `examples/__init__.py` | Makes `examples/` a subpackage so the entry point can resolve it |
| `examples/interactive_catheter_fluoro.py` | `main()` entry point, unified sim loop, DSA pipeline |
| `isaaclab_newton/solvers/xcath_rod_solver.py` | Physics solver with vessel mesh collision |
| `isaaclab_newton/solvers/xpbd_rod_solver.py` | XPBD engine, CUDA graph capture, proximal control |
| `fluorosim/rendering/diffdrr_slang_renderer.py` | Batched Slang GPU renderer (fluoro-simulator repo) |
| `fluorosim/rendering/diffdrr_slang.slang` | Slang shader — Beer-Lambert ray march + catheter compositing |
| `fluorosim/vasculature.py` | CT ingestion, vessel mask extraction, mesh generation |

---

## Relationship to RL Training Loop

The interactive UI is architecturally identical to the RL rollout loop:

```
RL training:    obs → policy → action → env.step() → render → obs′
Interactive UI: button → control → solver.step() → render → display
```

The physics solver, renderer, and coordinate transforms are the same production components. The only difference is that button presses replace the policy network. This means any behaviour visible in the UI is exactly what an RL agent would experience during training.
