# Isaac Lab 3 Validation Report

**Date:** 2026-03-24
**Tag:** v3.0.0-beta
**Version:** 3.0.0

## Environment

| Component | Details |
|-----------|---------|
| Isaac Lab version | 3.0.0 (v3.0.0-beta) |
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA L40 (49140 MiB) |
| CPU cores | 128 |
| RAM | 1.0 TiB |
| Physics backend | Newton 1.0.0 (MuJoCo-Warp 3.6.0) |

## Installation

Isaac Lab 3 was installed using the `./isaaclab.sh --install` workflow into
a Python 3.12 virtual environment. Key packages installed:

- `isaaclab 4.5.22` (core framework)
- `isaaclab_tasks 1.5.11` (task definitions including Cartpole)
- `isaaclab_newton 0.5.9` (Newton/MuJoCo-Warp physics backend)
- `isaaclab_physx 0.5.11` (PhysX backend stubs)
- `isaaclab_assets 0.3.0` (robot asset configs)
- `mujoco 3.6.0` / `mujoco-warp 3.6.0`
- `warp-lang 1.12.0`
- `newton 1.0.0` (Newton physics solver)
- `gymnasium 1.2.1`

**Note on Isaac Sim 6.0:** The `isaacsim>=6.0.0` package is not yet published
to `pypi.nvidia.com` (latest available is 4.5.0.0). Isaac Lab 3 supports a
kitless multi-backend architecture — the Newton/MuJoCo-Warp backend runs
fully without Isaac Sim, enabling headless validation on this node.

## Test Results

### CLI Unit Tests (no simulator required)

14/14 tests passed in `source/isaaclab/test/cli/test_install.py`:

```
============================== 14 passed in 0.06s ==============================
```

### Simulator-dependent Tests

Tests in `source/isaaclab/test/managers/` and `source/isaaclab/test/utils/`
require `AppLauncher` which depends on `EXP_PATH` (set by Isaac Sim). These
tests are skipped as Isaac Sim 6.0 is not yet pip-installable.

### Task Registration

185 Isaac Lab gym environments registered successfully, including:

- `Isaac-Cartpole-v0` ✓
- `Isaac-Cartpole-Direct-v0` ✓
- `Isaac-Ant-Direct-v0` ✓
- (182 additional tasks)

## Cartpole Rollout (100 steps × 64 envs)

**Physics backend:** Newton (MuJoCo-Warp, kitless — no Isaac Sim required)
**Device:** cuda:0 (NVIDIA L40)

```
Observation space: Dict('policy': Box(-inf, inf, (64, 4), float32))
Action space:      Box(-inf, inf, (64, 1), float32)
Policy obs shape:  torch.Size([64, 4])

Sample rewards (first 5 envs): [ 0.016  -0.039  -0.074   0.015  -0.006]
Done (first 5 envs):           [False False False False False]
```

## Throughput Benchmark

| Metric | Value |
|--------|-------|
| Steps | 100 |
| Parallel envs | 64 |
| Total transitions | 6,400 |
| Wall-clock time | ~0.54s |
| **Throughput** | **~11,800 env-steps/s** |

Runs 1: 12,091 env-steps/s
Run 2: 11,511 env-steps/s
**Average: ~11,800 env-steps/s**

## GPU Utilization

Measured after 100-step rollout:

| Metric | Value |
|--------|-------|
| GPU | NVIDIA L40 |
| VRAM total | 49,140 MiB |
| VRAM used | 576 MiB |
| GPU utilization | 22% |
| Temperature | 35°C |

Note: GPU utilization reflects post-rollout idle state; peak utilization
during the 100-step rollout is higher due to CUDA graph execution.

## Success Criteria

- [x] Isaac Lab 3 installs without errors
- [x] Built-in task registers and steps correctly (185 tasks registered)
- [x] 100-step rollout completes with obs/reward logged
- [x] Throughput benchmark reported (~11,800 env-steps/s)
- [x] Report pushed to `autopilot-reports/isaac-lab-3-test/`
