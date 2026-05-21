# Technical Plan: RL-Trained Catheter Navigation with Fluoroscopy Observations

> Architecture for training catheter navigation policies in Isaac Lab with
> Beer-Lambert fluoroscopy compositing as the image observation.

---

## 0 — Goal

Train a visuomotor policy that navigates a simulated catheter to a target
location inside patient vasculature, receiving fluoroscopy image observations
and outputting proximal push/rotate motor commands — the same interface a
robotic catheter driver uses in the physical system.

---

## 1 — Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      CatheterFluoroEnv(DirectRLEnv)                 │
│                                                                     │
│  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │  Action      │   │  Physics        │   │  Observation         │  │
│  │  Decoder     │   │  XPBDRodSolver  │   │  Beer-Lambert        │  │
│  │             │   │  (multi-env)    │   │  Compositing (Warp)  │  │
│  │  push_vel   │──▶│  step(dt)       │──▶│  GPU image tensor    │  │
│  │  rotate_vel │   │  pos (E,N,3)    │   │  (E, H, W)          │  │
│  └─────────────┘   │  + collision    │   └──────────────────────┘  │
│                     └─────────────────┘                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Reward: -d(tip, target) - λ·F_contact - τ·dt + bonus_reached ││
│  └─────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
         │                                        ▲
         │ obs_dict["policy"]                     │ actions (E, 2)
         ▼                                        │
┌──────────────────────────────────────────────────────────────────────┐
│                  RL Algorithm (rsl_rl / skrl / rl_games)             │
│                  CNN encoder → MLP policy                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2 — Phase 1: State-Based RL (no images, fastest path to training)

Build a minimal `DirectRLEnv` using the production `RodSolver` (already
supports `num_envs > 1`) with vector observations.  This validates the full
training loop before adding image complexity.

### 2.1 Proximal Control API

Add to `RodSolver` / `XPBDRodSolver`:

```python
def apply_proximal_control(
    self,
    push_velocity: torch.Tensor,    # (num_envs,) m/s along insertion axis
    rotate_velocity: torch.Tensor,  # (num_envs,) rad/s axial rotation
    dt: float,
) -> None:
```

Implementation: before the predict phase of each substep, update the root
particle's position and orientation:

```
x_root += push_velocity * dt * insertion_direction
q_root  = q_root ⊗ quat_from_axis_angle(insertion_axis, rotate_velocity * dt)
```

The root particle has `inv_mass = 0` (fixed), so the XPBD solver will not
move it — our kinematic update is the sole controller.

For multi-env `RodSolver`, this operates on `self.data.positions[:, 0, :]`
and `self.data.orientations[:, 0, :]` with batch indexing.

### 2.2 Environment: `CatheterStateEnv(DirectRLEnv)`

```python
@configclass
class CatheterStateEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=2)

    # Rod config
    num_segments: int = 20
    rod_length: float = 0.2         # 200 mm catheter
    rod_radius: float = 0.00045     # 0.45 mm (3F)
    young_modulus: float = 1e8      # Nitinol
    density: float = 6450.0

    # Action space: (push_velocity, rotate_velocity)
    action_space = 2
    action_scale_push = 0.01        # m/s max push speed
    action_scale_rotate = 1.0       # rad/s max rotation speed

    # Observation: tip pos (3) + tip vel (3) + segment positions (N*3)
    #            + target pos (3) + insertion depth (1)
    observation_space = 20 * 3 + 3 + 3 + 3 + 1    # = 70

    # Reward
    rew_distance_scale = -10.0
    rew_contact_penalty = -0.1
    rew_time_penalty = -0.01
    rew_reached_bonus = 100.0
    target_reached_threshold = 0.005  # 5 mm

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=2.0, replicate_physics=True,
    )
```

#### `_setup_scene()`

1. Instantiate `RodSolver(config, num_envs=self.num_envs)`
2. Load vessel SDF mesh as collision geometry
3. Place target markers at randomised vascular target locations
4. Optionally create `VisualizationMarkers` for viewport debugging

#### `_pre_physics_step(actions)`

```python
push_vel  = actions[:, 0] * self.cfg.action_scale_push
rot_vel   = actions[:, 1] * self.cfg.action_scale_rotate
self.solver.apply_proximal_control(push_vel, rot_vel, self.cfg.sim.dt)
```

#### `_apply_action()` / `_physics_step()`

```python
self.solver.step(dt=self.cfg.sim.dt)
```

#### `_get_observations()`

```python
positions = self.solver.data.positions          # (E, N, 3)
velocities = self.solver.data.velocities        # (E, N, 3)
tip_pos = positions[:, -1, :]                   # (E, 3)
tip_vel = velocities[:, -1, :]                  # (E, 3)
insertion_depth = positions[:, 0, 0:1]          # (E, 1) root x

obs = torch.cat([
    positions.reshape(self.num_envs, -1),       # (E, N*3)
    tip_pos,                                     # (E, 3)
    tip_vel,                                     # (E, 3)
    self.target_positions,                       # (E, 3)
    insertion_depth,                             # (E, 1)
], dim=-1)

return {"policy": obs}
```

#### `_get_rewards()`

```python
tip_pos = self.solver.data.positions[:, -1, :]
dist = torch.norm(tip_pos - self.target_positions, dim=-1)

reward = (
    self.cfg.rew_distance_scale * dist
    + self.cfg.rew_time_penalty
    + self.cfg.rew_reached_bonus * (dist < self.cfg.target_reached_threshold).float()
)
# Add contact penalty if collision is available
if hasattr(self.solver.data, 'contact_forces'):
    contact_mag = self.solver.data.contact_forces.norm(dim=-1).sum(dim=-1)
    reward += self.cfg.rew_contact_penalty * contact_mag

return reward
```

#### `_get_dones()`

```python
dist = torch.norm(tip_pos - self.target_positions, dim=-1)
reached = dist < self.cfg.target_reached_threshold
timed_out = self.episode_length_buf >= self.max_episode_length
return reached | timed_out, timed_out
```

#### `_reset_idx(env_ids)`

```python
self.solver.data.reset(env_ids)
# Randomise target within reachable workspace
self.target_positions[env_ids] = sample_targets(len(env_ids))
```

### 2.3 Training Launch

```bash
# Register the env
# In isaaclab_newton/envs/__init__.py:
#   gymnasium.register(id="CatheterState-v0", entry_point="...", kwargs={...})

./isaaclab.sh -p -m rsl_rl.train \
    --task CatheterState-v0 \
    --num_envs 512 \
    --headless
```

### 2.4 Deliverable

A trained MLP policy that navigates a catheter tip to a target using
push/rotate commands, validated in the Isaac Lab viewport.

---

## 3 — Phase 2: Multi-Env XPBDRodSolver

Extend `XPBDRodSolver` to support `num_envs > 1`.  This is needed if you want
the self-contained solver (no external Newton dependency) for training.

### 3.1 Workspace Array Extension

Every array in `_Workspace` gains a leading environment dimension:

```
positions:        (E * N_pts,) wp.vec3    → flat with env_id * N_pts + particle_id
orientations:     (E * N_pts,) wp.quat
rest_lengths:     (E * N_edges,) float
diag_blocks:      (E * N_edges * 36,) float
...
```

Flat layout avoids warp divergence.  A helper maps `(env_id, local_id)` to
flat index: `flat = env_id * stride + local_id`.

### 3.2 Kernel Modifications

All per-particle and per-edge kernels change from `dim=N` to `dim=E*N` with:

```python
@wp.kernel
def _xr_predict_pos_batched(
    ...,
    n_pts: int,          # per-env particle count
):
    tid = wp.tid()
    env_id = tid // n_pts
    i = tid % n_pts
    # ... same logic, but index arrays with tid instead of i
```

The block-Thomas kernel changes from `dim=1` to `dim=E`:

```python
@wp.kernel
def _xr_block_thomas_batched(
    ...,
    n_edges: int,
    n_edges_stride: int,  # = n_edges * 36 (array stride per env)
):
    env_id = wp.tid()
    # Forward/backward sweep on env_id's block-tridiagonal system
    # All array accesses offset by env_id * stride
```

This gives embarrassingly parallel direct solves across environments.

### 3.3 Performance Target

| Metric | Target | Basis |
|--------|--------|-------|
| 512 envs, 20 seg | > 60 Hz | Each env ~50 KB; 512 × 50 KB = 25 MB |
| 4096 envs, 20 seg | > 60 Hz with CUDA graphs | 200 MB; A6000 has 48 GB |

---

## 4 — Phase 3: GPU Beer-Lambert Compositing (Warp Kernel)

Port the NumPy/OpenCV `composite_catheter_beer_lambert()` to a Warp kernel
that produces batched image tensors for all environments simultaneously.

### 4.1 Data Flow

```
Per env:
  positions (N, 3) [m]
      │  ×1000
      ▼
  pos_mm (N, 3) [mm]
      │
      ├──▶ project_points_3d_to_2d(K, Rt) → uv (N, 2) [px]
      │
      ├──▶ compute_projected_radii(K, Rt, r_mm) → radii (N-1,) [px]
      │
      └──▶ mu_profile (N-1,)  [precomputed, static per env]
              │
              ▼
      @wp.kernel beer_lambert_composite(
          drr_texture,      # (H, W) float, shared across envs (or per-env if randomised)
          uv,               # (E, N, 2)
          radii_px,          # (E, N-1)
          mu_profile,       # (E, N-1)  or (N-1,) if shared
          output_images,    # (E, H, W) float
      )
```

### 4.2 Warp Kernel Design

Each thread handles one pixel of one environment:

```python
@wp.kernel
def _beer_lambert_composite(
    drr: wp.array2d(dtype=wp.float32),          # (H, W) background
    uv: wp.array(dtype=wp.float32),             # (E * N * 2) flat
    radii: wp.array(dtype=wp.float32),           # (E * (N-1)) flat
    mu: wp.array(dtype=wp.float32),              # (E * (N-1)) flat
    n_segments: int,
    img_w: int,
    img_h: int,
    output: wp.array(dtype=wp.float32),          # (E * H * W) flat
):
    tid = wp.tid()
    env_id = tid // (img_h * img_w)
    pixel_id = tid % (img_h * img_w)
    py = pixel_id // img_w
    px = pixel_id % img_w

    atten = float(0.0)
    seg_base = env_id * n_segments

    for seg in range(n_segments):
        # Load segment endpoints in pixel space
        idx0 = (env_id * (n_segments + 1) + seg) * 2
        idx1 = (env_id * (n_segments + 1) + seg + 1) * 2
        u0 = uv[idx0]; v0 = uv[idx0 + 1]
        u1 = uv[idx1]; v1 = uv[idx1 + 1]
        r_px = radii[seg_base + seg]
        mu_val = mu[seg_base + seg]

        # Perpendicular distance from pixel to segment line
        # ... (same math as NumPy version) ...
        # Cylinder chord: t(d) = 2*sqrt(r^2 - d^2) for d < r
        # atten += mu_val * chord_norm

    # Beer-Lambert
    bg = drr[py, px]
    output[tid] = bg * wp.exp(-atten)
```

**Thread count:** `E * H * W`.  At E=512, H=W=128: 8.4M threads — well
within GPU occupancy.  Use 128×128 images for training (not 512×512) to keep
throughput high.

### 4.3 Projection as a Warp Kernel

The 3D→2D projection and per-segment radius computation are also ported to
Warp to avoid CPU round-trips:

```python
@wp.kernel
def _project_catheter(
    positions: wp.array(dtype=wp.vec3),    # (E * N_pts)
    K_flat: wp.array(dtype=wp.float32),    # (9,) intrinsic matrix
    Rt_flat: wp.array(dtype=wp.float32),   # (E * 12) per-env extrinsic
    scale: float,                           # 1000.0 (m→mm)
    n_pts: int,
    uv_out: wp.array(dtype=wp.float32),    # (E * N_pts * 2) flat
    radii_out: wp.array(dtype=wp.float32), # (E * (N_pts-1)) flat
    physical_radius_mm: float,
):
    tid = wp.tid()
    env_id = tid // n_pts
    i = tid % n_pts
    # ... transform, project, compute per-segment radius magnification
```

### 4.4 Imaging Noise as Domain Randomization

Add Poisson noise via a Warp kernel using `wp.rand_init` / `wp.poisson`
(or approximate with Gaussian noise for speed).  Randomize per-env:

- Photon count (500–5000): X-ray dose variation
- Scatter fraction (1–5%): patient thickness variation
- C-arm angle: per-env `Rt` matrix from sampled LAO/RAO angle
- DRR background: index into a library of pre-rendered patient DRRs

### 4.5 Performance Target

| Component | Budget | Notes |
|-----------|--------|-------|
| Physics step | < 5 ms | 512 envs, 20 segments, 2 substeps |
| Projection | < 0.5 ms | 512 envs, 21 points each |
| Compositing | < 3 ms | 512 envs, 128×128 images |
| Noise | < 0.5 ms | Per-pixel Poisson |
| **Total frame** | **< 10 ms** | **> 100 Hz training throughput** |

---

## 5 — Phase 4: Image-Based RL Environment

### 5.1 `CatheterFluoroEnv(DirectRLEnv)`

```python
@configclass
class CatheterFluoroEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=4)

    # Rod
    num_segments: int = 20
    young_modulus: float = 1e8

    # Fluoroscopy image observation
    image_width: int = 128
    image_height: int = 128
    num_stacked_frames: int = 4       # temporal context

    # C-arm
    sid: float = 1000.0               # mm
    sod: float = 600.0                # mm
    pixel_spacing: float = 0.81       # mm/px
    carm_angle_range: tuple = (-30.0, 30.0)   # domain randomization

    # Action
    action_space = 2                  # push, rotate
    action_scale_push = 0.01          # m/s
    action_scale_rotate = 1.0         # rad/s

    # Observation: stacked fluoroscopy frames
    observation_space = [image_height, image_width, num_stacked_frames]

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=2.0, replicate_physics=True,
    )
```

### 5.2 Observation Pipeline

```python
def _get_observations(self):
    # 1. Project catheter positions to 2D (Warp kernel)
    wp.launch(_project_catheter, dim=self.num_envs * self.n_pts, inputs=[
        self.solver.positions_flat,
        self.K_flat, self.Rt_flat,
        1000.0, self.n_pts,
        self.uv_buf, self.radii_buf, self.radius_mm,
    ], device=self.device)

    # 2. Composite via Beer-Lambert (Warp kernel)
    wp.launch(_beer_lambert_composite, dim=self.num_envs * self.H * self.W, inputs=[
        self.drr_textures,
        self.uv_buf, self.radii_buf, self.mu_profile,
        self.n_segments, self.W, self.H,
        self.current_frame_buf,
    ], device=self.device)

    # 3. Add noise (Warp kernel)
    wp.launch(_poisson_noise, dim=self.num_envs * self.H * self.W, inputs=[
        self.current_frame_buf, self.dose_levels, self.rng_states,
    ], device=self.device)

    # 4. Stack frames (shift buffer, insert new frame)
    self.frame_stack[:, :, :, 1:] = self.frame_stack[:, :, :, :-1].clone()
    self.frame_stack[:, :, :, 0] = wp.to_torch(self.current_frame_buf).reshape(
        self.num_envs, self.H, self.W
    )

    return {"policy": self.frame_stack}
```

### 5.3 Policy Architecture

```
Input: (E, 128, 128, 4)  — 4 stacked grayscale fluoroscopy frames

CNN Encoder:
  Conv2d(4, 32, 8, stride=4) → ReLU           # → (E, 32, 31, 31)
  Conv2d(32, 64, 4, stride=2) → ReLU          # → (E, 64, 14, 14)
  Conv2d(64, 64, 3, stride=1) → ReLU          # → (E, 64, 12, 12)
  Flatten → Linear(64*12*12, 512) → ReLU      # → (E, 512)

MLP Head:
  Linear(512, 256) → ReLU
  Linear(256, 2)                               # → (E, 2) = (push, rotate)
```

Standard Nature-DQN encoder adapted for continuous control via PPO.

### 5.4 Domain Randomization Schedule

| Parameter | Range | When |
|-----------|-------|------|
| C-arm LAO/RAO angle | [-30, +30] deg | Per episode reset |
| Photon dose | [500, 5000] photons/px | Per episode reset |
| Scatter fraction | [0.01, 0.05] | Per episode reset |
| Catheter stiffness | [0.8E, 1.2E] | Per episode reset |
| Target position | Within reachable workspace | Per episode reset |
| DRR background | Random from patient library | Per episode reset |

---

## 6 — Phase 5: Vessel Collision + Realistic Navigation

### 6.1 SDF Collision Kernel

Port `RodSolver`'s mesh BVH collision into `XPBDRodSolver`:

```python
@wp.kernel
def _xr_sdf_collision(
    pred_pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    sdf_volume: wp.Volume,
    n_pts: int,
):
    tid = wp.tid()
    p = pred_pos[tid]
    d = wp.volume_sample_f(sdf_volume, p, wp.Volume.LINEAR)
    if d < 0.0:
        grad = wp.volume_sample_grad_f(sdf_volume, p, wp.Volume.LINEAR)
        n = wp.normalize(grad)
        pred_pos[tid] = p - d * n          # project to surface
        vn = wp.dot(vel[tid], n)
        if vn < 0.0:
            vel[tid] = vel[tid] - vn * n   # zero normal velocity
```

Insert between constraint projection and integration in `_substep()`.

### 6.2 Contact Force Observation

Accumulate per-particle contact normal forces for reward computation:

```python
contact_force[tid] = -d * contact_stiffness * n
```

Sum over particles to get `tip_contact_force` for the reward penalty term.

---

## 7 — Implementation Order

| Phase | What | Depends On | Effort | Outcome |
|-------|------|-----------|--------|---------|
| **1** | `apply_proximal_control()` + `CatheterStateEnv` | Nothing | 1 week | State-based RL training works |
| **2** | Multi-env `XPBDRodSolver` | Phase 1 | 2 weeks | 512+ parallel envs |
| **3** | GPU Beer-Lambert Warp kernels | Phase 2 | 2 weeks | Batched fluoroscopy images on GPU |
| **4** | `CatheterFluoroEnv` + CNN policy | Phase 3 | 1 week | Image-based RL training works |
| **5** | SDF collision + contact rewards | Phase 2 | 2 weeks | Realistic vessel navigation |
| **6** | Domain randomization + sim2real | Phase 4+5 | 2 weeks | Transfer-ready policies |
| **7** | CUDA graph capture | Phase 2 | 1 week | 2-3x training speedup |

**Total:** ~11 weeks to full image-based catheter RL with vessel collision and
domain randomization.

**Quick win (Phase 1 alone):** ~1 week to a trainable state-based catheter
environment with the existing `RodSolver`.

---

## 8 — File Structure (Proposed)

```
source/isaaclab_newton/
├── isaaclab_newton/
│   ├── solvers/
│   │   ├── xpbd_rod_solver.py           # ← add multi-env + proximal control
│   │   ├── rod_solver.py                # ← add proximal control
│   │   └── ...
│   ├── envs/                            # NEW
│   │   ├── __init__.py                  # gymnasium.register()
│   │   ├── catheter_state_env.py        # Phase 1: state-based DirectRLEnv
│   │   └── catheter_fluoro_env.py       # Phase 4: image-based DirectRLEnv
│   ├── rendering/                       # NEW
│   │   ├── __init__.py
│   │   ├── beer_lambert_warp.py         # Phase 3: Warp compositing kernels
│   │   └── carm_projection_warp.py      # Phase 3: Warp projection kernels
│   └── ...
├── examples/
│   ├── train_catheter_state.py          # Phase 1: rsl_rl training script
│   └── train_catheter_fluoro.py         # Phase 4: image-based training
└── agents/
    ├── rsl_rl_catheter_state_ppo.yaml   # Phase 1: PPO hyperparams
    └── skrl_catheter_fluoro_ppo.yaml    # Phase 4: CNN-PPO hyperparams
```
