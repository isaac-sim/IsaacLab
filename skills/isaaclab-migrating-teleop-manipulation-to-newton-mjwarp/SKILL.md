---
name: isaaclab-migrating-teleop-manipulation-to-newton-mjwarp
description: Port an Isaac Teleop humanoid or manipulator environment from PhysX to the Newton MJWarp backend. Use when adding a newton_mjwarp physics preset to a teleop task, when a Newton port goes NaN, when the hand slips through or sticks to the grasped object, or when Newton replay success is far below PhysX.
---

# Porting an Isaac Teleop Env to Newton MJWarp

## When To Use

Use this when an Isaac Teleop task (GR1T2, G1, a dexterous hand, or an arm plus gripper) already
works on PhysX and must also run under `physics=newton_mjwarp`. It assumes the task uses
`PresetCfg`, an IK action term, and optionally a recorded MCAP replayed through
`teleop_replay_agent.py`.

Read the [asset migration guide](../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst)
and the [MJWarp solver page](../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst)
first. This skill is the delta: failures those pages do not yet cover, each traced to a root
cause on a real port.

## Order Of Work

Work in this order. Each step's failure mode masks the ones below it, so skipping ahead wastes
whole benchmark runs.

1. Add the preset and get the model to compile.
2. Fix asset import: mass, collision geometry, materials.
3. Fix the actuator set: reset survival, mimic joints, undriven joints.
4. Only then tune the solver, and expect little.

## 1. Add The Preset

Attach a `PhysicsCfg(PresetCfg)` to `sim.physics`. Keep `default` as the untouched PhysX config
so that path stays byte-identical, and put every Newton change behind `newton_mjwarp`.

Two traps:

* **`__post_init__` runs BEFORE presets resolve.** If it reads `scene.robot.spawn.usd_path`, the
  `spawn` object itself cannot be a preset. Preset individual *fields* instead
  (`spawn.rigid_props`, `spawn.func`, `spawn.physics_material`).
* **Sibling env classes do not inherit it.** A `-WaistEnabled` variant that subclasses
  `ManagerBasedRLEnvCfg` rather than the base task needs its own
  `self.sim.physics = PhysicsCfg()`, or `physics=newton_mjwarp` fails with `Unknown preset(s)`.

## 2. Asset Import: Probe Before You Tune

Write a probe that prints imported values and run it on **both** backends before changing
anything. Guessing here costs 20-replay runs; the probe costs ninety seconds.

```python
obj = env.unwrapped.scene["object"]
print(obj.data.default_mass, obj.data.default_inertia)

from isaaclab_newton.physics import NewtonManager as NM

model = NM.get_model()
print(model.shape_type.numpy())  # GeoType: 8 = MESH, 10 = CONVEX_MESH
print(model.shape_body.numpy())  # shapes per body; a link with none has no collider
```

### Mass and inertia

An asset that authors no mass gets one derived from each backend's own collision volume, and the
two disagree. Measured on a steering wheel: PhysX 0.5845 kg, Newton 0.2812 kg, a factor of 2.08,
with inertia off by the same factor. Demonstrations recorded against the PhysX value then replay
with forces tuned for an object of twice the mass.

Author the mass explicitly. **`MassPropertiesCfg` on the spawn cfg silently does nothing when the
asset's `RigidBodyAPI` sits on a nested prim rather than the spawn root.** Apply
`UsdPhysics.MassAPI` to the body prim inside a custom spawner instead, then re-probe to confirm
the value actually changed.

### Collision geometry

`GeoType.CONVEX_MESH` means a convex hull. On a curved, tapered finger a hull fills every
concavity, so the collider stands proud of the rendered mesh and the hand contacts objects before
it looks like it should. Import hand links as `GeoType.MESH`:

```python
UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
approximation = mesh_prim.GetAttribute("physics:approximation")
if not approximation:
    approximation = mesh_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
approximation.Set("none")
```

Three separate reasons this silently does nothing, each worth a debug cycle:

* **The attribute may not exist.** Overwriting only when present skips every link that authors
  nothing. Use `CreateAttribute` when missing.
* **The meshes may be behind a USD instance.** A link's `collisions` scope is often
  `IsInstanceable()`, so `Usd.PrimRange` never descends into it and instance proxies cannot be
  edited in place. Call `SetInstanceable(False)` on those scopes first. Detect it when
  `prim.GetChildren()` is empty but
  `prim.GetFilteredChildren(Usd.TraverseInstanceProxies())` is not.
* **Decomposition can be rejected.** Meshes authored at 100x and scaled to 0.01 produce pieces
  below MuJoCo's minimum volume: `mesh volume is too small`. Use `"none"` for those.

Log how many prims you actually modified. Do not infer success from behaviour.

### Colliders authored on Xforms are silently dropped

An asset may author its collider as a `boundingCube` `PhysicsCollisionAPI` on an **Xform** rather
than on mesh prims. PhysX resolves that and builds the collider; **Newton emits no shape at all**,
so anything resting on it falls straight through. `Props/PackingTable/packing_table.usd` does this,
and it is shared by several pick-place tasks.

There is no import-time error -- the scene loads, the surface renders, and objects drop through it.
Reproduce the bounding volume with an invisible static box whose collider is enabled only under
`newton_mjwarp`, so PhysX keeps using the asset's own collider:

```python
packing_table_collider = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/PackingTableCollider",
    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.49705]),
    spawn=sim_utils.CuboidCfg(
        size=(2.4736, 0.762, 0.9941),
        visible=False,
        collision_props=preset(
            default=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            newton_mjwarp=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    ),
)
```

Offset the proxy to match the table prim's own placement -- the same asset sits at `z = 0` in the
fixed-base pick-place scene and `z = -0.3` in the locomanipulation one. Verify by stepping with
zero actions and watching the object's height settle rather than fall.

### Concave objects are the hard limit

**MuJoCo has no concave mesh-mesh collision.** A torus, such as a steering-wheel rim, a handle or
a ring, cannot be collided as a single mesh. As `MESH` the hand passes straight through and grips
air; as `convexHull` the hole fills in and it becomes a solid disc. Neither is graspable, and no
amount of friction, stiffness, damping, margin or substep tuning changes it.

Options, best first:

1. Ship the asset with a `convexDecomposition` rim so the ring is many convex segments.
2. Substitute a Newton-validated primitive under the `newton_mjwarp` preset only.
   `Isaac-Lift-Franka` runs MJWarp by default and grasps `MeshCuboidCfg` / `MeshSphereCfg`
   primitives, so those shapes and their physics are already proven there.
3. SDF collision (`NewtonSDFCollisionPropertiesCfg`), as the NIST factory task does. SDF is
   mutually exclusive with any mesh approximation: `convex_hull` and `coacd` both raise
   *"replaces the mesh with non-mesh geometry"*, so the approximation must be `"none"`.

If you substitute the object, preset the contact sensors alongside it. A
`filter_prim_paths_expr` still pointing at the old asset's nested body fails sensor
initialization outright and the env never builds.

### Materials: the zero-friction trap

**Newton resolves an omitted friction value to zero.** A body with no authored physics material
has no friction at all and slides out of the grasp. This is invisible in the config, and
`NewtonShapeCfg.mu` will not save you: that default only applies to shapes *lacking* an explicit
material, so sweeping it changes nothing while the real problem sits elsewhere.

Author friction on **both** the object and the robot:

```python
preset(
    default=None,
    newton_mjwarp=[
        UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
    ],
)
```

## 3. Actuators

### Actuator properties do not survive a hard reset

`sim.reset(soft=False)` re-finalizes the Newton model from the builder, and the rebuilt model
carries the **USD-authored** drives, so every value from `ArticulationCfg.actuators` is silently
discarded. Measured on GR1T2: stiffness 4400 to 53026, damping 40 to 2148, armature 0.1 to 0.0,
effort limit to `inf`. MJWarp cannot integrate those, so the first *commanded motion* goes
non-finite.

The signature is distinctive and misleading. State is clean after reset, actions are finite, IK
targets are byte-identical to PhysX, and a **zero** command still looks perfectly healthy; only a
real command NaNs. It presents as a physics-tuning problem and is not one.

Scripts calling only `env.reset()` (`teleop_se3_agent.py`) never hit it. Scripts following the
`record_demos.py` sequence (`teleop_replay_agent.py`, which calls `env.sim.reset()`) hit it every
run. **If live teleop works but replay goes NaN, look here first.**

Fix by re-applying the actuator configs on `PhysicsEvent.PHYSICS_READY`, not just at init.

### Mimic joints

Newton lowers USD mimic couplings to `mjEQ_JOINT` equality constraints. If the config drives both
leader and follower, the drive and the constraint fight and the fingers oscillate at tens of
rad/s while holding a static pose. Drive the **leader** only and leave followers passive
(`stiffness=0, damping=0`), mirroring `panda_finger2_passive` in the Franka lift task.

Give passive followers a capped `joint_effort_limit` and an `armature`. The Shadow Hand asset
documents the failure directly: *"an uncapped effort limit on either end diverges to NaN."*

### Undriven joints

PhysX may never excite unactuated legs or head because gravity is disabled per body. Newton does
not read `physxRigidBody:disableGravity`; use
`MujocoRigidBodyPropertiesCfg(disable_gravity=True, gravcomp=1.0)`. Undriven joints then pick up
energy and spin (a head yaw joint reached 47 rad/s), so add a modest posture drive.

## 4. Solver Tuning: Last, And Expect Little

On a real port, after the fixes above, every solver parameter was measured at n=20 and none
closed the gap to PhysX:

| change | result |
|---|---|
| friction `mu` 2.0 / 3.0 / 4.0 | 0-1/20 |
| contact stiffness `ke` 25k / 100k / 160k | 0-4/20 |
| contact damping `kd` 500 / 2000 / 5000 | 1-4/20 |
| `num_substeps` 16 | 0/20 |
| `njmax` / `nconmax` raised 5x | 2/20 |
| contact `margin` 5 mm | 0/20 |
| object mass parity | 1/20 |
| best combination found | 4/20, against PhysX 12/20 |

Two very different `(ke, kd)` points peaked at exactly the same rate, which is a real ceiling
rather than a missed sweet spot. **Treat a plateau as evidence the problem is upstream in the
asset, not as a reason to sweep harder.**

`use_mujoco_contacts=True` segfaulted on a humanoid-plus-mesh scene, and is mutually exclusive
with `collision_cfg` (clear it with `collision_cfg=null`; Hydra's `~key` delete does not work
here).

## Per-Asset Checks That Are Not Universal

Several steps above are *checks*, not mandatory edits. Confirm each against the asset in front of
you rather than copying another port:

* **Hand colliders may already be correct.** GR1T2's finger links import as `CONVEX_MESH` and need
  the fix above; G1's TriHand links already import as `MESH`. Probe `shape_type` before editing.
* **Gravity handling depends on the base.** A fixed-base humanoid can use
  `MujocoRigidBodyPropertiesCfg(disable_gravity=True, gravcomp=1.0)`. A *locomanipulation* robot
  walks, so its lower-body policy needs real ground contact -- copying the fixed-base trick breaks
  locomotion.
* **Solver profile follows the task, not the robot.** Use the locomotion profile (pyramidal cone,
  `impratio=1`) for walking tasks and the dexterous profile (elliptic, `impratio=10`) for
  fixed-base manipulation, then raise the contact budget for the extra bodies.

## Contact Sensors: Validate Patterns Against Newton Labels

`ContactSensorCfg.prim_path` is a regex. PhysX resolves it against USD prim paths; **Newton
matches it against the model's body labels**, which preserve the asset's intermediate grouping
prims. A pattern written for PhysX can match nothing under MJWarp, and sensor init then fails the
whole env build:

```
ValueError: No bodies matched the sensing object pattern(s).
```

Real example: the G1 hands nest under a grouping prim, so the Newton label is
`/Robot/left_hand/left_hand_index_0_link` while the task's pattern was
`/Robot/left_hand_[^/]*_link`. `[^/]*` cannot cross a path separator, so nothing matched. Preset
the corrected pattern for `newton_mjwarp` and leave the PhysX one alone.

Dump the labels before writing a pattern:

```python
from isaaclab_newton.physics import NewtonManager as NM

print([lbl for lbl in NM.get_model().body_label if "hand" in lbl])
```

Note the articulation's `data.body_names` are short names and will *not* reveal this -- they look
matchable when the labels are not.

## Policy-Driven Action Terms Must Detach

Newton writes joint targets through Warp kernels, which reject a torch tensor that requires grad:

```
RuntimeError: Can't get __cuda_array_interface__ on Variable that requires grad
```

Any action term that runs a pretrained policy (a locomotion policy wrapped by a locomanipulation
task, for instance) must run inference under `torch.no_grad()` or detach its output. PhysX writes
through torch and never trips this, so the bug is invisible until the port. It surfaces on the
first `env.step`, not at construction.

## Standalone Probes Need Cameras When The Env Has One

Launching a probe with `AppLauncher(args_cli, enable_cameras=False)` against an env that carries a
camera fails deep in the render graph with a message that never mentions cameras:

```
ValueError: Invalid object in Py_Graph in getWrappedGraphFromNode
```

Use `enable_cameras=True` in probes for camera-bearing envs, or strip the camera from the cfg.

## Benchmarking Without Fooling Yourself

Use the CI replay harness as the metric and respect its variance.

* **n=20 minimum.** `reset_object` randomises the object by +/-1 cm and Newton's
  `deterministic_mode` is `NOT_GUARANTEED`. One config scored 1/3 then 0/6 at identical settings;
  another looked like a 4x win and was pure noise.
* **One process at a time.** The harness timeout is wall-clock. GPU contention pushed a PhysX
  baseline from 60% down to 33% and nearly redirected the whole investigation.
* **Do not change the harness timeout while using it to measure.** Check what CI actually passes
  (`MAX_REPLAY_DURATION_S`, often 60 s) rather than the script default of 600 s.
* **Size any cap off measured success time.** Successes took about 21 s, so 30 s is safe while
  20 s would have truncated every one of them.
* **Verify a change took effect before benchmarking it.** A mass fix that silently did nothing
  produced a 20-run result that meant nothing.
* **`timeout` conflates "slow" with "never completed."** Successes finish in roughly 230
  iterations; a failure that runs 8500 has exhausted the capture and idled, not run slowly.

The replay agent needs `setup_preset_cli` plus `resolve_task_config` plus
`sys.argv = [sys.argv[0]] + hydra_args` before it will accept `physics=` at all, and its
"at least 2 renders" assertion must be relaxed for `--viz none` benchmarking.

## Diagnosing a NaN

**Never test `value > threshold` to detect divergence.** `nan > x` is `False`, so the check
passes silently and the NaN looks like healthy physics. Count non-finite entries instead.

Bisect in this order; each step is one ninety-second standalone run:

1. Is state clean after `reset()`? If not, it is initialization.
2. Are actions finite? Print min, max and the quaternion norms.
3. Do IK targets match PhysX for the same action? Byte-identical means the controller is fine.
4. Does a zero action step cleanly? A uniform one? The exact recorded frame?
5. Bisect the action vector: arm block against hand block.
6. Compare against a simpler task (Cartpole, Franka lift) to see whether the failure is general
   or specific to this asset.

Beware clamping when interpreting synthetic actions: a uniform `+1.0` on joints limited to
`[-1.57, 0]` clamps to 0 and silently tests the *zero* case instead of a large one.

## Validation Checklist

1. `physics=newton_mjwarp` resolves and the model compiles.
2. Probe confirms mass and inertia match PhysX within a few percent.
3. Probe confirms every hand link has colliders, and that they are `MESH` not `CONVEX_MESH`.
4. Objects rest on support surfaces instead of falling through them (step with zero actions and
   watch the height settle).
5. Object and robot both carry authored friction.
6. A hard `sim.reset()` followed by a commanded motion stays finite.
7. In teleop, the two hands' fingertips touch without visible interpenetration.
8. In teleop, the object can be grasped, carried and released without sticking.
9. Contact-sensor patterns match actual Newton body labels, not just USD paths.
10. Any policy-driven action term runs under `no_grad` or detaches.
11. Replay success at n=20 sits within a stated margin of the PhysX baseline, both measured one
    process at a time with the CI timeout.

## References

* [Asset migration guide](../../docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst)
* [MJWarp solver profiles](../../docs/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.rst)
* [PhysX and Newton policy transfer](../../docs/source/how-to/transfer_policies_between_physx_and_newton.rst)
* NIST factory assets, the reference for detailed meshes on MJWarp (SDF plus authored materials):
  `source/isaaclab_tasks/isaaclab_tasks/contrib/nist/factory_assets_cfg.py`
* Franka lift, the reference for mimic-joint handling (`panda_finger2_passive`):
  `source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka/franka_env_cfg.py`
* Validated graspable primitives, `ObjectCfg` in
  `source/isaaclab_tasks/isaaclab_tasks/core/lift/lift_env_cfg.py`
