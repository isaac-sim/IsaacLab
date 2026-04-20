# OVPhysX IMU Sensor — Design Spec

**Issue:** [#5318 — \[OVPHYSX\] IMU sensor](https://github.com/isaac-sim/IsaacLab/issues/5318)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Implement `Imu` and `ImuData` for the OVPhysX backend, satisfying the `BaseImu` and `BaseImuData` contracts. The IMU sensor reports angular velocity and linear acceleration in the sensor's body frame. It uses numerical differentiation of velocities (same as PhysX approach).

**Validation environment:** No existing task uses `ImuCfg` directly. Write a standalone validation script that attaches an IMU to a falling/rotating rigid body and verifies output matches expected physics (gravity acceleration, angular velocity from known torque).

## Guiding Principles

- Mirror PhysX `Imu` implementation
- Do not modify `BaseImu` or `BaseImuData` contracts
- Numerical differentiation for linear acceleration (same as PhysX — ovphysx does not provide native acceleration)

## Contract to Satisfy

### BaseImu (source/isaaclab/isaaclab/sensors/imu/base_imu.py)

Inherits `SensorBase`. Additional abstract:

| Property/Method | Type | Description |
|-----------------|------|-------------|
| `data` | `BaseImuData` | IMU data container |
| `_initialize_impl()` | method | Setup rigid body view and offsets |
| `_update_buffers_impl(env_mask)` | method | Fetch state and compute IMU outputs |

### BaseImuData (source/isaaclab/isaaclab/sensors/imu/base_imu_data.py)

| Property | Shape | Type | Description |
|----------|-------|------|-------------|
| `ang_vel_b` | `(N,)` | `wp.vec3f` | Angular velocity in body frame [rad/s] |
| `lin_acc_b` | `(N,)` | `wp.vec3f` | Linear acceleration in body frame [m/s^2], includes gravity |

## Architecture

### File Layout

```
source/isaaclab_ovphysx/isaaclab_ovphysx/sensors/
├── __init__.py
├── __init__.pyi
├── kernels.py           (shared sensor kernels)
└── imu/
    ├── __init__.py
    ├── __init__.pyi
    ├── imu.py           (~300 lines)
    ├── imu_data.py      (~80 lines)
    └── kernels.py       (~100 lines)
```

### Implementation Pattern

**imu.py:**

1. `_initialize_impl()`:
   - Find target prim and its rigid-body ancestor
   - Resolve fixed transform from ancestor body to target prim (done once)
   - Compose with configured sensor offset
   - Get body state from ovphysx via TensorBindings:
     - `RIGID_BODY_ROOT_POSE` or `ARTICULATION_BODY_POSE` for transforms
     - `RIGID_BODY_ROOT_VELOCITY` or `ARTICULATION_BODY_VELOCITY` for velocities
     - Body COM data for lever arm correction
   - Query world gravity from OvPhysxManager (needed for gravity bias)
   - Allocate buffers including `prev_lin_vel_w` for numerical differentiation

2. `_update_buffers_impl(env_mask)`:
   - Read body transforms and velocities from ovphysx bindings
   - Read COM positions (may be CPU — copy to GPU)
   - Launch `imu_update_kernel` per environment:
     - Apply sensor offset quaternion to body orientation
     - Correct linear velocity for COM offset (lever arm cross product)
     - Numerically differentiate: `lin_acc = (vel_current - vel_prev) / dt + gravity_bias`
     - Rotate angular velocity and linear acceleration to body frame
     - Store results and update `prev_lin_vel_w`

**imu_data.py:**
- Simple container with `_ang_vel_b` and `_lin_acc_b` warp arrays
- Properties return the arrays directly

### OVPhysX API Requirements

| API | Purpose | Notes |
|-----|---------|-------|
| Body transforms | Orientation for frame rotation | Already available via Articulation bindings |
| Body velocities | Linear + angular velocity | Already available |
| Body COM | Lever arm correction | Already available (CPU) |
| Gravity vector | Acceleration bias | From OvPhysxManager or USD scene |

**No new ovphysx APIs needed** — all required data is already available through existing TensorBindings used by the Articulation. The IMU just reads body state from a different entry point (the sensor's target body rather than the root).

### Warp Kernels

| Kernel | Purpose |
|--------|---------|
| `imu_update_kernel` | Per-env: offset composition, COM correction, numerical diff, frame rotation |
| `imu_reset_kernel` | Zero velocity history and output buffers |

These are direct ports of the PhysX kernels — the math is identical, only the data source changes.

## Tests

### Backend-specific tests (copy from PhysX)

**Source:** `source/isaaclab_physx/test/sensors/test_imu.py`
**Target:** `source/isaaclab_ovphysx/test/sensors/test_imu.py`

Changes: swap physics config to `OvPhysxCfg`.

### Validation script

Write `source/isaaclab_ovphysx/test/sensors/check_imu.py`:
- Create a rigid body with known initial angular velocity
- Attach IMU sensor
- Step physics
- Verify `ang_vel_b` matches expected value
- Verify `lin_acc_b` includes gravity (~9.81 m/s^2 downward in body frame)

## Dependencies

- **Requires PR #4852 merged**
- **No new ovphysx APIs needed**
- **Does not depend on #5316** — can use Articulation body bindings directly

## Estimated Scope

- `imu.py`: ~300 lines
- `imu_data.py`: ~80 lines
- `kernels.py`: ~100 lines
- Tests: ~150 lines of adaptations
- Validation script: ~100 lines
