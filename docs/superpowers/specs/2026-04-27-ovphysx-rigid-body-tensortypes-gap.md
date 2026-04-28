# OVPhysX wheel — `RIGID_BODY_*` TensorType gap spec

**Audience:** @marcodiiga (ovphysx wheel maintainer)
**Consumer:** IsaacLab `antoiner/feat/ovphysx_rigidobject` branch — implements [#5316 — \[OVPHYSX\] RigidObject asset](https://github.com/isaac-sim/IsaacLab/issues/5316)
**Date:** 2026-04-27
**Status:** Updated per Marco's feedback — renames and shape corrections applied.

The IsaacLab `RigidObject` / `RigidObjectData` asset for the OVPhysX backend is written assuming the `ovphysx` wheel ships dedicated `RIGID_BODY_*` `TensorType` enum values. This document is the contract IsaacLab codes against. Once the wheel ships these values, the `pytest.importorskip` guards on the IsaacLab side unblock automatically.

## 0. Naming

The original spec used `RIGID_BODY_ROOT_POSE` and `RIGID_BODY_ROOT_VELOCITY`. Marco's feedback prefers the shorter `RIGID_BODY_POSE` and `RIGID_BODY_VELOCITY` — "Root" is articulation vocabulary; a standalone rigid body IS the body. IsaacLab has been updated to use these shorter names throughout.

## 0.1. Mass shape

The original spec specified `(N, 1)` for `RIGID_BODY_MASS` and `RIGID_BODY_INV_MASS`. The wheel ships `(N,)` instead. IsaacLab consumes `(N,)` from the wheel and reshapes to `(N, 1)` internally in the `body_mass` property to satisfy the `BaseRigidObjectData` contract (`Shape is (num_instances, 1)`).

## 1. Already-shipping `TensorType` enum values

The following six variants are already exposed by the wheel. `N` = number of rigid actor instances matched by the binding pattern.

| Enum value | Shape | Components | Units | Device | R/W |
|---|---|---|---|---|---|
| `RIGID_BODY_POSE` | `(N, 7)` | `(px, py, pz, qx, qy, qz, qw)` | m, dimensionless | GPU | R/W |
| `RIGID_BODY_VELOCITY` | `(N, 6)` | `(vx, vy, vz, wx, wy, wz)` | m/s, rad/s | GPU | R/W |
| `RIGID_BODY_WRENCH` | `(N, 9)` | `(fx, fy, fz, tx, ty, tz, px, py, pz)` | N, N·m, m | GPU | W |
| `RIGID_BODY_MASS` | `(N,)` | scalar | kg | CPU | R/W |
| `RIGID_BODY_COM_POSE` | `(N, 7)` | `(px, py, pz, qx, qy, qz, qw)` in actor-link frame | m, dimensionless | CPU | R/W |
| `RIGID_BODY_INERTIA` | `(N, 9)` | row-major flatten of 3×3 `(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)` in COM frame | kg·m² | CPU | R/W |

IsaacLab uses these as-is. No wheel changes needed for these six.

dtype for all variants: `float32`.

## 2. Still-missing `TensorType` enum values (three remaining gaps)

Add three new `TensorType` variants. These are the only items still needed from the wheel.

| Enum value | Shape | Components | Units | Device | R/W | IsaacLab status |
|---|---|---|---|---|---|---|
| `RIGID_BODY_ACCELERATION` | `(N, 6)` | `(ax, ay, az, αx, αy, αz)` | m/s², rad/s² | GPU | R | **Optional** — FD from velocity |
| `RIGID_BODY_INV_MASS` | `(N,)` | scalar | 1/kg | CPU | R | Forward-compat alias only |
| `RIGID_BODY_INV_INERTIA` | `(N, 9)` | row-major flatten of 3×3 inverse inertia in COM frame | 1/(kg·m²) | CPU | R | Forward-compat alias only |

dtype for all variants: `float32`.

**`RIGID_BODY_ACCELERATION` is now optional.** IsaacLab finite-differences
`body_com_acc_w` from `body_com_vel_w` locally, mirroring the Newton backend
(kernel `derive_body_acceleration_from_body_com_velocities` in
`isaaclab_ovphysx.assets.kernels`). When the wheel ships
`RIGID_BODY_ACCELERATION`, it will serve as a direct hardware-read path and can
replace the FD path as a performance optimization — no IsaacLab code changes are
required on that side. Marco can land it at his convenience.

**`RIGID_BODY_INV_MASS` and `RIGID_BODY_INV_INERTIA` are forward-compat aliases.**
IsaacLab declares the aliases in `tensor_types.py` (guarded with
`try/except AttributeError` so the module imports cleanly today) but does not
consume them in any property. They become usable as soon as Marco ships the
matching enum values in the wheel — the `_CPU_ONLY_TYPES` set picks them up
automatically via the `_RIGID_BODY_OPTIONAL_CPU` tuple.

### `RIGID_BODY_ACCELERATION` docstring

Rigid actor spatial acceleration — read-only, GPU. Shape `(N, 6)`,
components `(ax, ay, az, αx, αy, αz)` [m/s², rad/s²].

### `RIGID_BODY_INV_MASS` docstring

Rigid actor inverse mass — read-only, CPU. Shape `(N,)` [1/kg].
Zero indicates an immovable actor.

### `RIGID_BODY_INV_INERTIA` docstring

Rigid actor inverse inertia tensor in COM frame — read-only, CPU.
Shape `(N, 9)`, row-major flatten of the 3×3 matrix [1/(kg·m²)].
Zero rows indicate locked rotational DOFs.

## 3. Pattern resolution behavior

The wheel's `create_tensor_binding(pattern, RIGID_BODY_*)` currently resolves against `UsdPhysics.RigidBodyAPI` prims using best-effort matching. This may currently include articulation links, since strict standalone-rigid-body filtering is not yet implemented at the binding level. An explicit selection policy that excludes articulation-owned links is on the wheel-side roadmap and will be added in a future iteration. The IsaacLab `RigidObject` surfaces a clear `RuntimeError` at init time if no matching prim is found.

Sample valid patterns:

- `/World/envs/env_*/object`
- `/World/cube_.*`
- `/World/Props/.*/RigidBody`

Failure mode: if `pattern` matches zero rigid-body prims, `create_tensor_binding` should return `None` (or raise the same exception the articulation analog raises today) — IsaacLab's `_get_binding` already handles that via `try/except` + `logger.debug`.

## 4. `RIGID_BODY_WRENCH` write semantics

`binding.write(buf)` with `buf.shape == (N, 9)`:

- Component layout: `(fx, fy, fz, tx, ty, tz, px, py, pz)`.
- `(fx, fy, fz)` is a **world-frame** force [N].
- `(tx, ty, tz)` is a **world-frame** torque [N·m].
- `(px, py, pz)` is the **world-frame** point of application [m]. Point is in world frame, not body-local. (IsaacLab's `_body_wrench_to_world` kernel rotates body-frame inputs to world frame and writes the world-frame application point — wheel implementation should expect inputs already in world frame.)
- Semantics: instantaneous (single-step). Cleared after each sim step. No persistent forces stored on the wheel side — those are layered in IsaacLab's `WrenchComposer`.

This must match `ARTICULATION_LINK_WRENCH` semantics for a degenerate single-link articulation, so the wheel implementation can share kernels.

## 5. CPU/GPU routing

CPU-only: `RIGID_BODY_MASS`, `RIGID_BODY_INV_MASS`, `RIGID_BODY_COM_POSE`, `RIGID_BODY_INERTIA`, `RIGID_BODY_INV_INERTIA`.

GPU: `RIGID_BODY_POSE`, `RIGID_BODY_VELOCITY`, `RIGID_BODY_ACCELERATION`, `RIGID_BODY_WRENCH`.

Matches the routing of the corresponding `ARTICULATION_*` analogs.

## 6. Test parity expectation

For a USD scene containing a single rigid-body actor (no articulation), `RIGID_BODY_*` reads/writes should yield numerically identical behavior to a degenerate single-link articulation accessed via `ARTICULATION_*` (mass propagation, COM offset, inertia tensor handling, gyroscopic torque if `enable_gyroscopic_forces=True`, gravity application).

If the wheel implementer wants a smoke-test scene, `Isaac-Repose-Cube-Allegro-Direct-v0` with the IsaacLab-side preset additions on `antoiner/feat/ovphysx_rigidobject` (a `DexCube` rigid-body + Allegro hand articulation) is ready to run via `./scripts/run_ovphysx.sh source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env.py --num_envs 4 --headless`.

## 7. Versioning & integration

- The wheel version bump and packaging are at the wheel maintainer's discretion.
- IsaacLab pins to whichever wheel ships these enums; the IsaacLab-side bump is `isaaclab_ovphysx 0.1.2 → 0.2.0`.
- Once the wheel is available, IsaacLab's `pytest.importorskip` gates unblock and CI runs the new mock-based interface and extension tests.

## Open questions for the wheel implementer

(None expected — this is the frozen contract. Raise on the IsaacLab issue thread if any item above is ambiguous or impractical to implement, and we will revise both this gap spec and the IsaacLab side together.)
