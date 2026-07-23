# Newton/MJWarp Asset Migration Reference

## Contents

- Source-of-truth rule
- Asset classifications
- Migration record
- Concrete tooling
- Authored asset audit
- Per-solver asset configuration
- Coupled-joint decisions
- Actuator and action audit
- Armature and damping calibration
- MJWarp solver profile selection
- Task integration audit
- Failure triage
- Maintained source patterns

## Source-of-Truth Rule

Read the [official migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst) first. Use this reference as an agent-facing worksheet, not as a replacement for that guide.

## Asset Classifications

Use these labels when reporting status:

- PhysX-compatible: the asset works in the current PhysX task or standalone smoke.
- MJWarp-runnable: Newton can parse the asset and MJWarp can simulate it enough for a limited smoke.
- MJWarp-clean: authored metadata, task spawn path, and control path pass the validation checklist.

Do not collapse MJWarp-runnable and MJWarp-clean. A parser success can still hide placeholder inertials, render meshes used as colliders, stale actuator patterns, invalid reset penetrations, or a changed policy interface.

## Migration Record

Capture a row for the baseline and each candidate asset:

| Field | Required evidence |
| --- | --- |
| Source | URDF/MJCF/USD path and revision |
| Output | Interface USD and physics payload paths |
| Conversion | Fixed base, fixed-joint merge, collision approximation, multi-physics flags |
| Topology | Ordered bodies, joints, active DOFs, mimic/equality constraints |
| Physics | Mass, COM, inertia, collision, material, gravity, self-collision |
| Control | Actuator expressions, action order, scale, gains, armature, limits |
| Task | Sensors, frames, resets, rewards, terminations, `dt`, decimation |
| Validation | PhysX/MJWarp commands, seeds, residual warnings, verdict |

## Concrete Tooling

Run the checked-in task agents against the exact task, not an isolated asset stage:

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task TASK --num_envs 4 --headless physics=physx
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task TASK --num_envs 4 --headless physics=newton_mjwarp
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task TASK --num_envs 4 --headless physics=physx
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task TASK --num_envs 4 --headless physics=newton_mjwarp
```

Use a nominal task/play config that disables randomization before construction. Save stdout and
stderr from each run, including the printed observation and action spaces, importer diagnostics,
solver warnings, reset behavior, and the first divergent step. Separately record the ordered body,
joint, actuator, action, and observation names from the resolved task configuration and verify every
contact-relevant object and heterogeneous clone group. Compare the PhysX and MJWarp records field by
field; do not infer contract parity from a successful spawn.

Use `scripts/tools/convert_urdf.py` or `scripts/tools/convert_mjcf.py` for reproducible conversion. Save the exact command, source revision, and generated interface/payload paths. Run `scripts/environments/zero_agent.py` and `random_agent.py` under both physics presets for continuous reset/step smokes.

## Authored Asset Audit

Inspect:

- Every dynamic link has `RigidBodyAPI` and intentional mass [kg], center of mass [m], and inertia [kg*m^2].
- Inertia is positive-definite and expressed in the intended local frame.
- Every intended collider has `CollisionAPI`; visual-only meshes remain non-colliding.
- Collision approximation, scale, margins/offsets, and material binding are intentional.
- Articulation root, fixed-base representation, joint axes, limits, and merged fixed joints match the mechanism.
- Mimic/equality constraints are authored once and parsed by both backends.
- Self-collision filters and disabled pairs match the baseline.
- References, payloads, meshes, and textures resolve outside the author's workstation.
- Task-level schema overrides target the same prims after conversion.

For URDF/MJCF conversion, verify `run_asset_transformer=True` and `run_multi_physics_conversion=True`. Do not use the deprecated `convert_mimic_joints_to_normal_joints` option as a migration strategy.

## Per-Solver Asset Configuration

Do not infer support from a field being present in a PhysX asset, USD layer, or imported Newton model. Use the [schema configuration class guide](../../../docs/source/overview/core-concepts/schema_cfgs.rst) and the generated Newton/MuJoCo and PhysX schema APIs to select supported properties:

- Put portable USD Physics properties in solver-common classes such as `RigidBodyBaseCfg`, `JointDriveBaseCfg`, `CollisionBaseCfg`, and `ArticulationRootBaseCfg`.
- Use `MujocoRigidBodyPropertiesCfg` and `MujocoJointDrivePropertiesCfg` only for properties implemented by Newton's MuJoCo/MJWarp solver.
- Use `Newton*PropertiesCfg` for supported Newton collision, material, articulation, and other Newton-native properties.
- Keep per-body damping, stabilization and solver-iteration controls, friction-patch settings, compliant-contact settings, and other PhysX-only parameters in `Physx*PropertiesCfg`.

Verify the exact field and runtime semantics in the current [Newton/MuJoCo schema API](../../../docs/source/api/lab_newton/isaaclab_newton.sim.schemas.rst) or [PhysX schema API](../../../docs/source/api/lab_physx/isaaclab_physx.sim.schemas.rst). Authoring or importing a value does not guarantee that the selected solver consumes it.

## Coupled-Joint Decisions

Answer in order:

1. Is the mechanism mechanically one DOF? If no, keep independent actuators and actions.
2. Does the asset already author the coupling? If yes, do not add a second task-side equality.
3. Which joint is the leader? Give it the active drive and action.
4. Is the follower passive? Set follower stiffness/damping to zero unless the physical model requires otherwise.
5. Does an existing checkpoint include both joints as actions? Preserve that width/order consistently or retrain; do not silently delete the follower action only in one backend. Check whether `last_action` is also part of the observation contract.
6. Reset coupled joint positions from one random sample. Do not draw independent values.
7. Randomize leader response, not follower damping independently.

## Actuator and Action Audit

Passing asset import is not enough. Also verify:

- Actuator joint name patterns resolve to the converted USD joint names.
- Every intended implicitly actuated joint has an authored drive. `JointDrivePropertiesCfg.ensure_drives_exist` is asset-wide and seeds every fully zero-gain drive, so leave it off when a passive follower must remain passive and author the intended active drives explicitly.
- Controller body names and frame names resolve.
- Action dimension, ordered names, scale, offset, clipping, and target type are identical across backends.
- Stiffness, damping, armature, effort limits, rated velocity limits, solver clamps, and friction are intentional.
- `velocity_limit` is a rated-speed value for actuator/task logic; MJWarp does not parse it into its solver model or enforce it.
- `velocity_limit_sim` can populate Newton's `Model.joint_velocity_limit`, but the MuJoCo/MJWarp solver drops that field and does not enforce the requested clamp. Check rated-speed boundaries explicitly.
- Armature is physically plausible rather than a small uniform placeholder.
- Damping and action scale yield a smooth step response without bang-bang sign changes.
- The policy cannot exploit hard joint stops or a backend-specific compliant limit.
- Zero-action and small nonzero-action rollouts are finite and move the expected joints or bodies.

## Armature and Damping Calibration

Joint armature adds reflected motor/transmission inertia to the diagonal of the generalized mass matrix:

`M_eff(q) = M(q) + diag(a)` and, approximately, `delta_qdot = inverse(M_eff) * impulse`.

For a geared revolute joint, start from `a = J_rotor * gear_ratio^2` when motor data is available. Otherwise use an identified or controlled impulse/step response. A small effective inertia at a distal joint, finger, or articulated object's free rotational coordinate can turn a moderate contact or drive impulse into a large velocity. MJWarp and PhysX regularize contacts, constraints, drives, and clamps differently, so a poorly conditioned model can appear stable in PhysX and produce large joint or angular velocities in MJWarp.

Zero gravity does not damp rotation. It keeps objects unsupported or in sustained hand contact instead of letting them settle onto a support. Repeated impulses can therefore excite low-inertia coordinates. Increase armature on articulated/free-joint object coordinates only when the representation exposes armature and the value is physically or empirically justified. A plain `RigidObject` has no actuator armature: correct body inertia first, and do not assume PhysX angular-damping or maximum-velocity attributes are consumed by MJWarp.

Armature changes control dynamics. For a joint, `omega_n = sqrt(kp / I_eff)` and `zeta = kd / (2 * sqrt(kp * I_eff))`; increasing armature with fixed damping lowers the damping ratio. Retune armature, stiffness, and damping together. Use the smallest justified armature that keeps impulse and step responses finite and plausible, and monitor action saturation/sign changes for bang-bang control.

Do not use armature to hide bad units, missing body inertia, invalid reset penetration, excessive effort/action scale, incorrect `dt * decimation`, or undersized contact capacity.

For Franka, source joint torque and speed limits from the Franka data sheet, take impedance guidance from the maintained libfranka controller, and document the origin of armature and tuned gains separately. Do not imply that every actuator value came from one source.

## MJWarp Solver Profile Selection

Read the official [MJWarp solver configuration and task profiles](../../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst) before authoring the `newton_mjwarp` preset. That page is the source of truth for current values and the PhysX-to-MJWarp mapping.

1. Classify the task as simple articulation/reach, locomotion, dexterous manipulation, or dense manipulation.
2. Copy the nearest documented profile as an explicit baseline; do not translate PhysX iteration or GPU-buffer values numerically.
3. Use `integrator="implicitfast"` and preserve the policy period. Establish `dt` and `num_substeps` before making contacts harder.
4. Size the per-environment `njmax` and `nconmax` budgets against the worst-case randomized state before changing `iterations`, `ls_iterations`, or `tolerance`.
5. Keep `use_mujoco_contacts=True` unless the task needs a feature of Newton's collision pipeline. Set `collision_cfg` only when `use_mujoco_contacts=False`.
6. Enable `debug_mode` while tuning. Sweep one convergence cap at a time and retain a change only when the target metric improves without a guardrail regression.
7. Treat `cone`, `impratio`, `ccd_iterations`, `update_data_interval`, collision refresh, and collision-pipeline capacities as symptom-specific settings, not boilerplate.

PhysX `bounce_threshold_velocity`, `friction_correlation_distance`, TGS/PGS actor iteration counts, stabilization, CCD, and GPU contact buffers have no direct numeric MJWarp equivalents. Reproduce the physical behavior and capacity requirement rather than the number.

## Task Integration Audit

- Use a task-local converted asset config if changing a shared config would invalidate other tasks or checkpoints.
- Expose `physx` and `newton_mjwarp` through a physics `PresetCfg`.
- Keep `dt * decimation` identical; treat Newton/MJWarp `num_substeps` as integration inside that policy interval.
- Size MJWarp contact and constraint capacity for the task before tuning convergence.
- Preserve observation and action shape/order while validating the asset. Any intentional contract change belongs to a retraining migration and must be handed off to the sim-to-sim workflow.
- Validate contact sensor paths, filters, thresholds, clipping, and frames after every hierarchy change.
- Reject invalid reset penetrations and equality violations before stepping.
- If caching valid resets, cover every heterogeneous asset group and rebuild after geometry changes.
- Compare nominal deterministic behavior before enabling domain randomization.
- Follow Newton's diagnose-first tuning order for MJWarp: reproduce, classify, validate the model, establish contact representation and time stepping, size capacity, run bounded convergence sweeps, tune contacts or drives according to the symptom, and optimize performance last.
- Change one parameter family at a time. Accept it only when the target metric improves without non-finite states or unacceptable penetration, residual, contact-count, constraint-count, or runtime regressions.
- Verify MJWarp's supported options, exact names, and defaults in the installed Newton version. Do not copy MJWarp settings to another Newton solver.

## Failure Triage

| Symptom | Investigate first |
| --- | --- |
| Placeholder inertia or explosive acceleration | Authored mass/COM/inertia, units, armature |
| Asset looks correct but contacts differ | Explicit colliders, approximation, material, scale, contact capacity |
| First step jumps | Reset penetration, independently reset mimic joints, stale default state |
| Joint chatters or alternates at full action | Damping, action scale, control period, armature, hard-stop reliance |
| Velocity behavior or termination differs | MJWarp does not enforce `velocity_limit` or `velocity_limit_sim`; a PhysX clamp may hide an explicit task termination |
| Gripper force disappears under DR | Stacked generic and gripper-specific gain randomization, passive follower damping |
| Checkpoint will not load | Observation/action width or ordered name changes |
| Checkpoint loads but behavior is nonsense | Normalizer state, observation order/frame/units, action scale/offset |
| Object works alone but not in task | Support collision, task overrides, contact sensor paths, reset validity |
| Some clone groups hang during reset prefill | Valid-state bank did not generate candidates for every heterogeneous group |

Use the [Newton Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html) to classify solver-specific symptoms and select the matching solver reference. Treat a backend change as a model port: replay the same commands, compare task state and penetration, and compare contact, constraint, force, or residual traces only when their definitions are equivalent.

## Maintained Source Patterns

Use these files to verify current behavior:

- `source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/franka/dexsuite_franka_env_cfg.py`: task-local converted asset, calibrated actuator groups, passive mimic follower, shared reset, closing-speed randomization.
- `source/isaaclab_assets/isaaclab_assets/robots/kuka_allegro.py`: identified per-joint limits, damping, friction, armature, and gravity behavior.
- `source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/dexsuite_env_cfg.py`: PhysX/MJWarp presets, randomization, observation history, solver capacity.
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`: shared rough-terrain MJWarp contact-pipeline and shape-margin pattern.
- `source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/mdp/events.py`: collision-valid reset bank.
- `source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager_cfg.py`: current Isaac Lab MJWarp options and defaults.
- `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`: rated versus solver velocity-limit semantics.
- [Schema configuration classes](../../../docs/source/overview/core-concepts/schema_cfgs.rst): solver-common and per-backend property classes and routing.
- `source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py`: layered, multi-physics URDF conversion.
- [Newton Simulation Tuning guide](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html): diagnose-first workflow and current solver-specific tuning references.
