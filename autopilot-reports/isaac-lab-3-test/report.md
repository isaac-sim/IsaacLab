# Isaac Lab 3.0 Beta — Validation Report

**Date:** 2026-03-24
**Node:** GPU validation node (NVIDIA L40, 49,140 MiB VRAM; 128 CPU cores; 1 TB RAM)
**Branch:** autopilot/isaac-lab-3-test

---

## Version Info

| Component | Version |
|-----------|---------|
| Isaac Lab | 4.5.22 (v3.0.0-beta, commit a4a7602f29e) |
| Isaac Sim | kit-less (Newton/MuJoCo-Warp backend only) |
| PyTorch | 2.10.0+cu128 |
| Warp | 1.12.0 |
| MuJoCo | 3.6.0 |
| MuJoCo-Warp | 3.6.0 |
| Newton | 1.0.0 |
| CUDA | 12.8 |
| Driver | 570.158.01 |
| Python | 3.12.13 |
| GPU | NVIDIA L40 (49,140 MiB) |

---

## Installation

**Method:** `./isaaclab.sh --install` (kit-less mode — Newton/Warp backend)
**Status:** SUCCESS

All 13 Isaac Lab submodules installed:

| Package | Version |
|---------|---------|
| isaaclab | 4.5.22 |
| isaaclab_assets | 0.3.0 |
| isaaclab_contrib | 0.3.0 |
| isaaclab_experimental | 0.0.2 |
| isaaclab_mimic | 1.2.3 |
| isaaclab_newton | 0.5.9 |
| isaaclab_ov | 0.1.1 |
| isaaclab_physx | 0.5.11 |
| isaaclab_rl | 0.5.0 |
| isaaclab_tasks | 1.5.11 |
| isaaclab_tasks_experimental | 0.0.1 |
| isaaclab_teleop | 0.3.3 |
| isaaclab_visualizers | 0.1.0 |

**RL Frameworks installed:** rl_games 1.6.1, rsl-rl-lib 5.0.1, stable-baselines3 2.7.1, skrl 1.4.3, robomimic 0.4.0

**Note:** `./isaaclab.sh --install` installs the Newton/Warp backend only. To install Isaac Sim 6.0 (PhysX/Kit backend): `./isaaclab.sh --install isaacsim`.

---

## Test Results

### Unit Tests (pytest, non-slow)

```bash
python3 -m pytest source/isaaclab_physx/test/test_mock_interfaces/ \
    source/isaaclab/test/cli/ \
    source/isaaclab/test/managers/test_manager_base.py \
    -v -k "not slow" --timeout=120
```

**Result: 210 PASSED, 35 warnings, 0 failures** ✅

Tests covering:
- Mock articulation view (Torch/Warp backends)
- Mock rigid body view (Torch/Warp backends)
- Mock rigid contact view
- Backend factory
- CLI install utilities
- Manager base

Tests requiring Isaac Sim `AppLauncher`/`EXP_PATH` (PhysX-kit) are expected to be skipped in kit-less mode.

---

## Cartpole Rollout (100 Steps, 64 Envs)

**Task:** `Isaac-Cartpole-Direct-Warp-v0` (from `isaaclab_tasks_experimental`)
**Backend:** Newton / MuJoCo-Warp (CUDA graph enabled)
**Device:** `cuda:0` (NVIDIA L40)
**Num Envs:** 64

### Configuration
- Solver: MJWarpSolverCfg (implicitfast integrator, 1 substep)
- CUDA graph: enabled
- episode_length_s: 5.0, decimation: 2
- Physics dt: 8.33ms, env step dt: 16.67ms

### Output

```
[INFO]: Environment device: cuda:0
[INFO]: Time taken for scene creation: 1.603s
[INFO]: InteractiveSceneWarp: 64 envs, spacing 4.0m
        Initialize solver: 2.857s, CUDA graph: 0.317s

Reset obs[0]: [-0.2203, 0.0000, 0.0000, 0.0000]
Step 1 obs[0]: [0.5057, -0.4300, -0.0051, -0.4108]
Step 1 reward: mean=0.756, min=0.378, max=0.994
100 steps × 64 envs: 0.341s (18,766 env-steps/sec)
Mean reward (100 steps): 0.2213
```

---

## Throughput Benchmark

| Config | Steps | Envs | Time (s) | **Env-Steps/s** |
|--------|-------|------|----------|-----------------|
| Warp env, CUDA graph ON | 100 | 64 | 0.341 | **18,766** |

**Throughput: 18,766 env-steps/sec** (Isaac-Cartpole-Direct-Warp-v0, 64 envs, NVIDIA L40)

---

## GPU Utilization

| Metric | Value |
|--------|-------|
| GPU | NVIDIA L40 |
| Driver | 570.158.01 |
| CUDA | 12.8 |
| Memory Used | 1,147 MiB / 49,140 MiB (2.3%) |
| GPU Utilization | 97% during rollout |

---

## Success Criteria

- [x] Isaac Lab 3 installs without errors (kit-less, Newton/Warp backend)
- [x] Built-in task (`Isaac-Cartpole-Direct-Warp-v0`) registers and steps correctly
- [x] 100-step rollout completes with obs/reward logged
- [x] **Throughput: 18,766 env-steps/sec** (64 envs, NVIDIA L40, CUDA graph)
- [x] Report pushed to `autopilot-reports/isaac-lab-3-test/`

---

## Notes

- **Kit-less mode**: This installation uses Newton/MuJoCo-Warp physics backend.
  No Omniverse/carb modules required.
- `Isaac-Cartpole-v0` (manager-based) and `Isaac-Cartpole-Direct-v0` require IsaacSim.
  `Isaac-Cartpole-Direct-Warp-v0` from `isaaclab_tasks_experimental` works kit-less.
- CUDA graph (`use_cuda_graph=True`) is the primary performance enabler.
- Inertia warnings on Cartpole USD bodies (slider, cart, pole) are benign.
- Tests requiring `AppLauncher`/`EXP_PATH` are expected to be skipped kit-less.
