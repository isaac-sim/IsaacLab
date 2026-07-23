# Newton/MJWarp Asset Migration Reference

Use this as a worksheet after reading the [official migration guide](../../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst).

## Migration Record

| Area | Record and compare |
| --- | --- |
| Source | URDF/MJCF/USD path, revision, conversion command, generated payloads |
| Mechanism | ordered bodies/joints, active DOFs, mimic/equality coupling, fixed base |
| Physics | mass, COM, inertia and frames, colliders, materials, gravity, self-collision |
| Control | actuator matches, action order/type/scale, gains, friction, armature, limits |
| Task | sensors, frames, resets, rewards, terminations, `dt`, decimation |
| Evidence | PhysX/MJWarp commands, seeds, warnings, first divergence, verdict |

Classify an asset as:

- **PhysX-compatible:** works in the source task.
- **MJWarp-runnable:** parses and passes a limited smoke.
- **MJWarp-clean:** passes the complete model, task, control, and contract gates.

## Reproducible Smoke Commands

Run the exact task in both backends:

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

Use a nominal configuration without startup randomization first. Save observation/action spaces, ordered names, importer and solver logs, reset behavior, and the first divergent step. Then cover every contact-relevant object and heterogeneous clone group.

## Model And Configuration Decisions

| Concern | Required action |
| --- | --- |
| Conversion | Prefer `convert_urdf.py` or `convert_mjcf.py`; keep `run_asset_transformer=True` and `run_multi_physics_conversion=True`. |
| Common properties | Use solver-common base cfgs such as `RigidBodyBaseCfg` and `JointDriveBaseCfg`. |
| Backend properties | Use `Mujoco*PropertiesCfg` for MJWarp-supported fields, `Newton*PropertiesCfg` for Newton-native fields, and `Physx*PropertiesCfg` only for PhysX. |
| Mass properties | Author positive mass and positive-definite inertia in the intended COM frame; never tune around placeholders. |
| Collision | Use explicit intended colliders; verify approximation, scale, margins, materials, filters, and support surfaces. |
| Coupled joints | Author one coupling, drive the leader, set follower stiffness/damping to zero, and reset from one shared sample. |
| Drives | Avoid asset-wide `ensure_drives_exist` when any follower must remain passive; author active drives explicitly. |
| Limits | Treat `velocity_limit` as rated speed and `velocity_limit_sim` as a solver request. MJWarp enforces neither; add explicit task/control checks when required. |
| Armature | Use reflected rotor inertia or identified response for articulated coordinates. A plain `RigidObject` has body inertia, not armature. |
| Damping | Retune after armature changes; compare step response and action sign changes instead of hiding oscillation with a clamp. |
| Resets | Reject penetration, invalid geometry, and inconsistent coupled states before the first step. |

For Franka, source torque and rated speed from the Franka data sheet, use maintained controller data as gain guidance, and document armature or identification assumptions separately.

## MJWarp Starting Profiles

The [MJWarp solver page](../../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst) is authoritative. Its current explicit baseline uses:

- `solver="newton"`, `integrator="implicitfast"`;
- `iterations=100`, `ls_iterations=50`, `tolerance=1e-6`; and
- `debug_mode=True` while tuning.

| Profile | `njmax` | `nconmax` | Cone / `impratio` | Substeps |
| --- | ---: | ---: | --- | ---: |
| Simple articulation/reach | 50 | 20 | pyramidal / 1 | 1 |
| Locomotion | 100 | 40 | pyramidal / 1 | 1 |
| Dexterous manipulation | 200 | 70 | elliptic / 10 | 2 |
| Dense manipulation | 300 | 200 | task-dependent | 2 |

Treat these as initial per-environment budgets, not fidelity guarantees. Tune in this order:

1. validate model, topology, contacts, and resets;
2. establish `dt`, `implicitfast`, and substeps;
3. size `nconmax` and `njmax`;
4. use `debug_mode` before changing convergence;
5. tune cone, `impratio`, and materials from measured slip/contact behavior; and
6. optimize performance last.

Keep `use_mujoco_contacts=True` unless the task needs Newton's collision pipeline. Set `collision_cfg` only with `use_mujoco_contacts=False`. PhysX iteration counts, GPU buffers, stabilization, CCD, bounce threshold, and friction-correlation distance have no numeric MJWarp conversion.

## Failure Triage

| Symptom | Inspect first |
| --- | --- |
| Placeholder inertia or explosive acceleration | units, mass/COM/inertia frames, armature |
| First-step jump | reset penetration, default state, coupled-joint reset |
| Missing or different contacts | colliders, scale, materials, contact capacity/path |
| Joint chatter or bang-bang actions | damping, armature, action scale, policy period, hard stops |
| Speed behavior differs | explicit rated-speed checks; PhysX clamp versus unenforced MJWarp limits |
| Checkpoint shape or semantics differ | ordered actions/observations and converted joint names |
| Standalone asset works but task fails | task overrides, supports, sensors, reset validity |
| Dense/randomized states fail | worst-case `njmax`/`nconmax`, collision-pipeline capacity |

Useful source patterns include the Franka Dexsuite task config, `kuka_allegro.py`, the MJWarp manager cfg, actuator cfgs, schema cfg documentation, and URDF converter cfg.
