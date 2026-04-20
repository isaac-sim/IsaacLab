# OVPhysX PVA Sensor — Design Spec

**Issue:** [#5319 — \[OVPHYSX\] PVA sensor](https://github.com/isaac-sim/IsaacLab/issues/5319)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `Pva` and `PvaData` for the OVPhysX backend, satisfying the `BasePva` and `BasePvaData` contracts. PVA is a superset of IMU — it provides world-frame pose, body-frame velocities, body-frame linear and angular accelerations, and projected gravity. Uses numerical differentiation for accelerations.

**Validation environment:** No existing task uses `PvaCfg`. Write a standalone validation script similar to IMU.

**Depends on:** Shares patterns with #5318 (IMU) — implement IMU first, then PVA extends it.

## Contract to Satisfy

### BasePva (source/isaaclab/isaaclab/sensors/pva/base_pva.py)

Inherits `SensorBase`. Additional abstract:

| Property/Method | Type |
|-----------------|------|
| `data` | `BasePvaData` |
| `_initialize_impl()` | method |
| `_update_buffers_impl(env_mask)` | method |

### BasePvaData (source/isaaclab/isaaclab/sensors/pva/base_pva_data.py)

| Property | Shape | Type | Description |
|----------|-------|------|-------------|
| `pose_w` | `(N,)` | `wp.transformf` | World-frame pose |
| `pos_w` | `(N,)` | `wp.vec3f` | World position [m] |
| `quat_w` | `(N,)` | `wp.quatf` | World orientation |
| `lin_vel_b` | `(N,)` | `wp.vec3f` | Body-frame linear velocity [m/s] |
| `ang_vel_b` | `(N,)` | `wp.vec3f` | Body-frame angular velocity [rad/s] |
| `lin_acc_b` | `(N,)` | `wp.vec3f` | Body-frame linear acceleration [m/s^2] |
| `ang_acc_b` | `(N,)` | `wp.vec3f` | Body-frame angular acceleration [rad/s^2] |
| `projected_gravity_b` | `(N,)` | `wp.vec3f` | Gravity in body frame |

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/sensors/
└── pva/
    ├── __init__.py
    ├── __init__.pyi
    ├── pva.py           (~350 lines)
    ├── pva_data.py      (~120 lines)
    └── kernels.py       (~150 lines)
```

### Implementation Pattern

Identical to IMU but with more outputs:

**pva.py:**

1. `_initialize_impl()`: Same as IMU — find ancestor, resolve offset, get gravity, allocate buffers
   - Additional: allocate `prev_ang_vel_w` for angular acceleration differentiation

2. `_update_buffers_impl(env_mask)`:
   - Read body transforms and velocities from ovphysx bindings
   - Launch `pva_update_kernel` per environment:
     - Apply sensor offset to get world-frame pose (pos_w, quat_w)
     - COM correction on linear velocity
     - Numerically differentiate both linear and angular velocities
     - Rotate all to body frame
     - Project gravity to body frame: `projected_gravity_b = quat_apply_inverse(quat, gravity_vec)`
     - Store 7 outputs + update velocity history

**pva_data.py:**
- Container with 8 warp array properties
- Simple property accessors

### OVPhysX API Requirements

**No new ovphysx APIs needed** — same data sources as IMU (body transforms, velocities, COM).

### Warp Kernels

| Kernel | Purpose |
|--------|---------|
| `pva_update_kernel` | Per-env: all IMU computation + angular acceleration + pose output + gravity projection |
| `pva_reset_kernel` | Zero all 8 output arrays + velocity history |

## Tests

**Source:** `source/isaaclab_physx/test/sensors/test_pva.py`
**Target:** `source/isaaclab_ovphysx/test/sensors/test_pva.py`

Validation script: `check_pva.py` — verify all 8 outputs for a body with known motion.

## Dependencies

- **Requires PR #4852 merged**
- **No new ovphysx APIs needed**
- **Implement after #5318 (IMU)** — shares patterns and kernels

## Estimated Scope

- `pva.py`: ~350 lines
- `pva_data.py`: ~120 lines
- `kernels.py`: ~150 lines
- Tests: ~150 lines of adaptations
