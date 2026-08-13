# Newton/MJWarp Asset Preparation Reference

This reference follows the [Prepare an Asset for Newton with MJWarp how-to](../../../docs/source/how-to/prepare_asset_for_newton.rst).

## Contents

- Prerequisites
- Import A Multi-Physics Asset
- Separate Common And Solver-Specific Properties
- Audit The Mechanical Model
- Match Collision, Contact, And Friction Behavior
- Validate Actuators And Limits
- Run Paired Smoke Tests
- Diagnose Newton-Only Failures

## Prerequisites

Select the explicit `newton_mjwarp` backend preset. Use the backend-and-preset documentation to distinguish intended task configuration from the shared policy contract.

## Import A Multi-Physics Asset

- Newton can parse supported authored USD Physics and PhysX properties, but verify support instead of assuming every field is used.
- For a new asset, use `scripts/tools/convert_urdf.py` or `scripts/tools/convert_mjcf.py`.
- Keep `run_asset_transformer=True` and `run_multi_physics_conversion=True` so conversions contain neutral physics, PhysX, and MuJoCo payloads.
- Expect the current converter's nested rigid-body structure.

## Separate Common And Solver-Specific Properties

| Property | Configuration |
| --- | --- |
| Common USD Physics | `RigidBodyBaseCfg`, `JointDriveBaseCfg`, and other base cfgs |
| MJWarp-specific | `MujocoRigidBodyPropertiesCfg`, `MujocoJointDrivePropertiesCfg`, `MujocoCollisionCfg` |
| Newton-native | matching `Newton*PropertiesCfg` |
| PhysX-only | matching `Physx*PropertiesCfg` |

A field present in an asset or imported model is not proof that MJWarp consumes it. Check the Newton/MuJoCo and PhysX schema APIs.

## Audit The Mechanical Model

Check intentional positive mass, COM, positive-definite inertia, and the corresponding frames. Verify explicit colliders, approximation, scale, margins or offsets, materials, restitution, filters, articulation root, fixed-base and fixed-joint representation, joint types, axes, limits, and body-level gravity behavior.

## Match Collision, Contact, And Friction Behavior

1. Verify colliders, material bindings, contact locations/counts, and available normal force.
2. Inspect `condim`: `1` is frictionless, `3` adds tangential friction, `4` adds torsional friction, and `6` adds rolling friction.
3. Tune material friction against measured slip; do not map PhysX static/dynamic settings numerically.
4. Use `priority`, `solmix`, `solref`, and `solimp` only for measured per-collider requirements. Use `NewtonCollisionCfg.contact_margin`, `contact_gap`, and `NewtonMeshCollisionCfg.max_hull_vertices` rather than raw importer attributes.

Track fixed-grasp displacement, contact count, effort, penetration, and success. Do not hide missing contacts, bad geometry, or insufficient effort with friction or `condim`. Use the MJWarp solver guide for global solver settings.

## Validate Actuators And Limits

- Audit per-joint effort, gains, friction, armature, action scale, and control period.
- Apply armature only to articulated coordinates, based on physical reflected inertia or controlled response. Retune damping when armature changes.
- `velocity_limit` is rated speed and `velocity_limit_sim` requests a solver clamp, but MJWarp enforces neither while stepping. Enforce required bounds in task or control logic.

## Run Paired Smoke Tests

Run the same fixed task state in PhysX and MJWarp through multiple resets. Check finite state, first-step impulses, saturation, angular velocity, contact loss, and importer/solver warnings. Reject penetration, impossible mimic states, and invalid randomized geometry before stepping.

## Diagnose Newton-Only Failures

Reproduce the first bad step with one environment, a fixed state, no randomization, and identical actions. Diagnose initialization/model, contact/capacity, controlled motion, or dense-scene demand before tuning. Use `NewtonCfg.debug_mode`; raise overflowing capacities before changing convergence settings.
