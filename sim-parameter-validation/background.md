# Background and evidence for Isaac Lab parameter simulation validation

This document contains the implementation research and existing-test inventory supporting the
[simulation-validation test-suite design](plan.md). It is evidence for the design, not the normative definition
of scope or acceptance criteria.

## Evidence classification

- **Analytical:** simulation output is compared with a closed-form equation, conservation law, or kinematic
  identity.
- **Behavioral:** the simulation checks a meaningful direction, ordering, equilibrium, or decay property, but
  not a complete closed-form trajectory.
- **Component:** a force law or controller is checked directly without validating the resulting simulated
  trajectory.
- **Storage/notification:** construction, conversion, aliasing, or solver refresh is checked without a physical
  observable.

Storage/notification evidence is necessary for runtime support but is not a substitute for an Isaac Lab
black-box test.

## Existing Isaac Lab implementation status

`source/isaaclab_newton/test/assets/test_articulation_kamino_joint_dynamics.py` is the reference partial
implementation of `PROFILE-DOF`, `FIX-DOF-STEP`, `FIX-LIMIT-POS`, and partial `FIX-PASSIVE`. Its Phase 0
classification is:

| Parameter ID | Kamino disposition (USD / Python / runtime) | Backend test | Physical evidence |
|---|---|---|---|
| `DRIVE-01` | `I / I / I` | `test_drive_01_stiffness_single_step` | Analytical joint position and velocity after one step |
| `DRIVE-02` | `I / I / I` | `test_drive_02_damping_single_step` | Analytical joint position and velocity after one step |
| `JOINT-07` | `I / I / I` | `test_joint_07_armature_single_step` | Analytical reduction in acceleration from added joint-space inertia |
| `CMD-01` | `N / N / I` | `test_cmd_01_feedforward_torque_implicit_single_step`, `test_cmd_01_feedforward_torque_explicit_single_step` | Analytical effort-driven joint position and velocity after one step |
| `CMD-02` | `N / N / I` | `test_cmd_02_velocity_reference_single_step` | Analytical pure-velocity-target response after one step |
| `JOINT-04` | `I / N / I` | `test_joint_04_position_limit` | USD and in-place runtime upper-limit trajectories; `runtime-error` asserts topology-change rejection |
| `JOINT-08` | `I / T / T` | `test_joint_08_passive_damping_usd_single_step` | Analytical single-step velocity decay from USD-authored passive damping; cfg/runtime blocked on [IsaacLab#6517](https://github.com/isaac-sim/IsaacLab/issues/6517) |

The stiffness, damping, and armature cases cover revolute and prismatic joints through USD,
`ImplicitActuatorCfg`, and runtime writers. The command rows have only a runtime authoring path; the `authoring`
axis in the feed-forward tests varies the surrounding implicit-drive configuration rather than the command
path. The procedural-build test is fixture validation and does not represent a parameter cell. Canonical
parameter IDs appear in pytest node IDs and assertion diagnostics.

Its `DRIVE-02` cases validate implicit drive damping through `write_joint_damping_to_sim_index`; they do not
cover the separate passive-damping contract in `JOINT-08`. Passive damping is authored in USD through
`newton:damping` and is dynamically equivalent to drive damping with a zero velocity reference. Python override
and runtime paths for passive damping remain `T` until
[IsaacLab#6517](https://github.com/isaac-sim/IsaacLab/issues/6517) exposes them separately from implicit drive
damping. Kamino runtime may change existing finite limit values in place (`I`); runtime writes that introduce or
remove a limit must raise instead of changing solver topology.
The test pins `dt = 1/120 s`, one Kamino Euler substep, CUDA graphs off, and `rtol = 5e-3`,
`atol = 2e-4`; these settings define the initial Kamino profile in the design.

Existing backend asset tests for state writes, mass, material properties, limits, collider offsets, and
notifications are evidence inputs. A test is promoted to implemented physical coverage only after confirming
that it invokes an Isaac Lab authoring path and asserts the resulting physical behavior rather than only reading
the value back.

## Existing Newton physical-verification coverage

The inventory below records the Newton fixtures that are directly relevant to the design.

### Newton fixture inventory

| Physical setup | Parameters exercised | Observable or reference | Newton tests | Solver coverage |
|---|---|---|---|---|
| Free sphere under uniform gravity | Gravity, initial pose | \(x=x_0+\tfrac12gt^2\), \(v=gt\) | `test_physics_verification.py::test_free_fall` | Featherstone, MuJoCo CPU/MJWarp, XPBD, SemiImplicit |
| Free projectile | Initial pose, initial linear velocity, gravity | Parabolic position and linear velocity | `test_physics_verification.py::test_projectile_motion` | Featherstone, MuJoCo CPU/MJWarp, XPBD, SemiImplicit |
| Free box under force or torque | Mass, inertia tensor, applied effort | \(v=F\Delta t/m\), \(\omega=\tau\Delta t/I\) | `test_body_force.py::test_floating_body` | Featherstone, MuJoCo CPU/MJWarp, XPBD, SemiImplicit |
| Free box with non-zero center of mass | Center-of-mass position, linear/angular velocity, force, torque | Center-of-mass translation and rotation invariants | `test_body_force.py::test_force_no_rotation`, `test_body_force.py::test_combined_force_torque`, `test_body_velocity.py` | Featherstone, MuJoCo CPU/MJWarp, XPBD, SemiImplicit |
| Runtime gravity changes on a free box | Gravity vector, runtime notification | Falling, constant-velocity, upward, and horizontal acceleration phases | `test_runtime_gravity.py::test_runtime_gravity_bodies` | XPBD, SemiImplicit, MuJoCo CPU/MJWarp, Kamino |
| Revolute and prismatic single-DOF bodies | Mass, inertia, feed-forward joint effort | \(v=Ft/m\), \(\omega=\tau t/I\) | `test_physics_verification.py::test_joint_actuation` | Featherstone, MuJoCo CPU/MJWarp, XPBD |
| Single-DOF PD drive | Joint state, targets, drive stiffness and damping, target mode | One-step PD dynamics and target convergence | `test_joint_drive.py`, `test_joint_controllers.py` | Primarily MuJoCo/MJWarp; target convergence also covers other registered solvers |
| Revolute joint with initial velocity | Passive joint damping | Damped response decays while the zero-damping control preserves speed | `test_joint_damping.py` | Featherstone, SemiImplicit, MJWarp, Kamino |
| Revolute drive with a low effort cap | Effort limit, drive gains | Clamped torque followed by one-step angular response | `test_joint_controllers.py::test_effort_limit_clamping` | MuJoCo/MJWarp |
| Pendulum | Gravity, mass, inertia, center-of-mass lever arm, joint geometry | Small-angle period, trajectory, and total energy | `test_physics_verification.py::test_pendulum_period`, `test_energy_conservation` | Featherstone, MuJoCo CPU/MJWarp, SemiImplicit; XPBD period only |
| Pendulum reaction force | Joint frames, mass, gravity, angular velocity | \(F=mg\) and \(F_c=m\omega^2r\) | `test_parent_force.py` | Featherstone, MJWarp |
| Boxes on inclined planes | Coulomb friction | Static threshold \(\theta_c=\arctan(\mu)\) | `test_rigid_friction_ramp.py::test_friction_ramp` | XPBD, MuJoCo CPU/MJWarp, VBD |
| Sliding boxes on level ground | Coulomb friction, initial velocity | Stopping distance \(d=v_0^2/(2\mu|g|)\) | `test_rigid_friction_ramp.py::test_friction_stopping_distance` | XPBD, MuJoCo CPU/MJWarp, VBD |
| Sphere dropped onto a plane | Restitution | Rebound height \(h_r=e^2h_0\) | `test_physics_verification.py::test_restitution`, `test_restitution_kamino` | XPBD, Kamino |
| Elastic and inelastic dropped spheres | Contact damping as MuJoCo's restitution control | Full versus near-zero rebound | `test_physics_verification.py::test_restitution_mujoco` | MuJoCo CPU/MJWarp |
| PD/PID actuator components | Actuator stiffness/damping, feed-forward effort, delay, effort and motor limits | Controller and clamping force equations | `test_actuators.py` | Component tests; not solver trajectories |
| Non-aligned Kamino joint frames | Parent and child joint frames | Frame conversion identities; finite gravity-driven motion | `test_solver_kamino_joint_frames.py` | Kamino; analytical conversion plus behavioral stepping |

Newton also has pendulum conservation, momentum, inverse-dynamics, mass-matrix, loop-constraint, deformable,
cable, and hydroelastic verification tests. They strengthen Newton's broader solver verification but do not
exercise additional Isaac Lab parameters in the current scope, so they are not proposed as primary fixtures.

### Parameter-to-evidence map

| Parameter | Closest Newton evidence | Evidence strength | Remaining Isaac Lab or backend gap |
|---|---|---|---|
| Gravity vector | Free fall; projectile; runtime gravity | Analytical at build time; behavioral at runtime | Exercise USD, Python override, and runtime write for each backend |
| Initial body pose | Projectile initial position | Analytical at construction | Distinguish reset default from live-state write |
| Initial body spatial velocity | Projectile and center-of-mass motion | Analytical | Distinguish authored reset velocity from current-state velocity |
| Mass | Force-driven free/prismatic body | Analytical | Repeat through all supported authoring paths |
| Inertia tensor | Torque-driven free/revolute body | Analytical | Vary individual diagonal components and joint axis |
| Center-of-mass position | Center-of-mass force/velocity tests; pendulum lever arm | Analytical | Add Isaac Lab runtime update and derived-frame refresh checks |
| Shape transform relative to body | Contact fixtures use transforms; Kamino notify test maps storage | Storage/notification only for parameter variation | Add first-contact or resting-pose control pair |
| Shape scale or dimensions | Contact fixtures use dimensions; Kamino notify maps scale | Storage/notification only for parameter variation | Add known contact-time or resting-height response |
| Collision radius | Kamino notify mapping | Storage/notification only | Add paired first-contact-distance response |
| Combined contact friction `mu` | Inclined-plane threshold and stopping distance using Newton's `shape_material_mu` | Analytical Coulomb coefficient on MJWarp; Kamino supports combined contact `mu` | Add Isaac Lab `MAT-03` coverage through USD, Python, and runtime paths on both Newton backends; each backend requires both `FIX-FRICTION-STATIC` and `FIX-FRICTION-DYNAMIC` |
| Static friction (PhysX) | Inclined-plane threshold | Analytical where distinct static coefficient is exposed | Exercise PhysX `MAT-01` through all authoring paths |
| Dynamic friction (PhysX) | Stopping distance | Analytical where distinct dynamic coefficient is exposed | Exercise PhysX `MAT-02` through all authoring paths |
| Restitution | Sphere rebound | Analytical for XPBD and Kamino ([newton-physics/newton#3588](https://github.com/newton-physics/newton/pull/3588)); behavioral MuJoCo control | Add Isaac Lab `FIX-RESTITUTION` coverage through USD, Python, and runtime paths |
| Contact margin | Kamino notify and contact-geometry tests | Storage/notification or geometry only | Add slow-approach first-contact response |
| Contact gap | Kamino notify and contact-geometry tests | Storage/notification or geometry only | Add slow-approach first-contact response |
| Parent joint frame | Reaction-force fixtures; Kamino frame conversion | Partial analytical/behavioral | Add known-effort motion-direction test |
| Child joint frame | Reaction-force fixtures; Kamino frame conversion | Partial analytical/behavioral | Add known-effort motion-direction test |
| Joint position/state | Joint controller initial state and targets | Analytical or target-convergence behavior | Separate reset state from runtime state write |
| Joint velocity/state | Joint damping and velocity-target tests | Behavioral; analytical in unforced free motion | Separate reset state from runtime state write |
| Lower and upper position limits | `test_joint_limits.py`; Kamino notify; Kamino `test_joint_04_position_limit` | Kamino USD and in-place runtime paths are behavioral; topology-change runtime writes error | Replicate Kamino USD and runtime coverage on MJWarp and PhysX; no Isaac Lab Python override is exposed |
| Velocity limit | MuJoCo property tests note no matching physical enforcement | Storage only on MJWarp; Kamino tracked by [vastsoun/newton#397](https://github.com/vastsoun/newton/issues/397) | MJWarp `JOINT-05` is an accepted Isaac Lab out-of-scope gap; add sustained-drive coverage for PhysX and document MJWarp/Kamino matrix dispositions |
| Effort limit | One-step clamped drive response | Analytical | Extend beyond MuJoCo/MJWarp and through Isaac Lab writers |
| Armature | MuJoCo/Kamino property and notify tests | Kamino USD/config/runtime paths have analytical Isaac Lab coverage | Extend the one-step acceleration fixture to MJWarp and PhysX |
| Passive joint damping | Damped versus undamped revolute joint; Kamino `test_joint_08_passive_damping_usd_single_step` | Kamino USD path is analytical; upstream Newton evidence is behavioral | Extend Kamino cfg/runtime after [IsaacLab#6517](https://github.com/isaac-sim/IsaacLab/issues/6517); replicate USD coverage on MJWarp and PhysX |
| Joint friction coefficient | MuJoCo property/notify tests | Storage/notification only | Add breakaway or stopping response with drive disabled |
| Drive stiffness / position gain | One-step joint drive | Analytical upstream evidence; Kamino has analytical Isaac Lab coverage | Extend the shared fixture to MJWarp and PhysX |
| Drive damping / velocity gain | One-step joint drive | Analytical upstream evidence; Kamino has analytical Isaac Lab coverage | Extend to MJWarp and PhysX and keep distinct from passive damping |
| Internal actuation/target mode | Position/velocity controller responses and runtime mode switch | Upstream Newton evidence only | No public Isaac Lab writer; retain as out of scope until an API contract exists |
| Explicit actuator stiffness integration | Direct PD component force law and active-controller gain writes | Component and storage only | Validate controller effort and simulated trajectory through the Isaac Lab actuator writer |
| Explicit actuator damping integration | Direct PD component force law and active-controller gain writes | Component and storage only | Validate controller effort and simulated trajectory through the Isaac Lab actuator writer |
| Constraint properties | MuJoCo loop-constraint tests | Analytical for selected constraints, no generic property-write coverage | Add cases only for properties Isaac Lab exposes |
| Tendon properties | Tendon unit tests are not a generic runtime physical oracle | No applicable evidence in this design | Add cases only for supported runtime writes |

### Coverage conclusions

- Newton-MJWarp participates in most shared analytical free-body, articulation, friction, and MuJoCo contact
  fixtures.
- Kamino is covered by shared public simulation tests for runtime gravity and passive joint damping. Its notify
  tests and internal joint-frame/kinematics tests provide additional storage and component evidence. Upstream
  Kamino restitution verification in [newton-physics/newton#3588](https://github.com/newton-physics/newton/pull/3588)
  supplies the rebound oracle but does not replace the planned Isaac Lab black-box trajectories.
- Existing Newton tests strongly support the proposed free-body, single-DOF, friction, and restitution
  experiments. Physical limit behavior, velocity limits, armature variation, joint friction, shape geometry,
  collision radius, contact margin/gap, and actuator-layer gain updates remain the highest-priority new
  integration fixtures.
- Tolerances should be derived from integrator order, time step, and contact model. Newton's analytical tests are
  useful starting points, but their tolerances should not be copied unchanged across backends.

## Backend implementation evidence

### Newton-Kamino

The property mapping and runtime behavior below are based on
`newton/_src/solvers/kamino/tests/test_solver_kamino_notify.py`.

#### Property mapping

- Global simulation: gravity is stored in `gravity`.
- Body initial state: pose and spatial velocity are stored in `body_q` and `body_qd`.
- Body inertial properties:
  - Mass is stored in `body_mass`; Newton also updates `body_inv_mass`.
  - Inertia is stored in `body_inertia`; Newton also updates `body_inv_inertia`.
  - Center of mass is stored in `body_com`.
- Shape properties:
  - Transform is stored in `shape_transform`.
  - Scale is stored in `shape_scale`.
  - Collision radius is stored in `shape_collision_radius`.
  - Combined contact friction is stored in `shape_material_mu`.
  - Contact margin and gap are stored in `shape_margin` and `shape_gap`.
- Joint geometry: parent and child frames are stored in `joint_X_p` and `joint_X_c`.
- Joint degree-of-freedom properties:
  - State is stored in `joint_q` and `joint_qd`.
  - Limits are stored in `joint_limit_lower`, `joint_limit_upper`, `joint_velocity_limit`, and
    `joint_effort_limit`.
  - Armature, passive damping, and drive gains are stored in `joint_armature`, `joint_damping`,
    `joint_target_ke`, and `joint_target_kd`.
- Actuator target mode is stored in `joint_target_mode`.

The Kamino notify test establishes aliasing or refresh behavior for shape transform, scale, collision radius,
margin, and gap. It does not by itself establish Isaac Lab black-box coverage for combined contact `mu`
(`MAT-03`); Kamino applies `shape_material_mu` in its contact model and the planned friction fixtures should
validate both the inclined-plane threshold and stopping-distance responses. Kamino restitution is covered
upstream by the rebound-height verification in
[newton-physics/newton#3588](https://github.com/newton-physics/newton/pull/3588); positive contact gap remains
tracked separately in [vastsoun/newton#375](https://github.com/vastsoun/newton/issues/375).

#### Runtime behavior

- A center-of-mass update refreshes the derived body pose, joint frames, and shape offsets.
- Newton's Kamino notification layer can refresh finite joint-limit arrays in place. Changing a joint between
  limited and unlimited requires solver recreation; Isaac Lab's
  `write_joint_position_limit_to_sim_index` rejects that topology change with a documented `RuntimeError`.
  In-place edits to existing finite lower/upper values are supported and covered by
  `test_joint_04_position_limit`.
- Armature, passive damping, and drive gains can be updated only while they do not add or remove the joint's
  dynamic-constraint allocation.
- Switching between active position and velocity actuation can be updated in place. Switching between active
  and passive actuation changes the actuation partition and requires solver recreation.
- Newton defines `CONSTRAINT_PROPERTIES` and `TENDON_PROPERTIES`, but the Kamino notify test currently treats
  both as no-ops and does not identify individual runtime properties.

### Newton-MJWarp

#### Property and API mapping

- Scene gravity is stored in the Newton model gravity array. Isaac Lab runtime randomization writes the model
  value and emits `MODEL_PROPERTIES`.
- Body pose/velocity and inertial properties use Newton's body state, mass, inertia, and center-of-mass arrays.
  Public `set_masses_*`, `set_inertias_*`, and `set_coms_*` methods emit `BODY_INERTIAL_PROPERTIES`.
- Shape geometry uses Newton shape transform, scale, collision-radius, margin, and gap arrays. Combined contact
  friction uses one `shape_material_mu` array; restitution is stored separately. Runtime material and
  collider-offset paths emit `SHAPE_PROPERTIES`.
- Joint state, limits, armature, damping, friction, and implicit drive gains use the Newton joint arrays.
  Public joint-property writers emit `JOINT_DOF_PROPERTIES`.
- Newton-MJWarp joint friction is an absolute dry-friction force or torque, mapped to MuJoCo
  `dof_frictionloss`; it is not a unitless coefficient. Contact dynamic friction is not separately exposed.

#### Runtime behavior and gaps

- Runtime writes that alter model topology, collision capacity, or the number/type of constraints require
  reconstruction and must have an error-path test once the public API reports that condition.
- Velocity-limit physical enforcement is not established by current integration tests on MJWarp; `JOINT-05`
  remains `X` as an accepted Isaac Lab out-of-scope gap with no issue filed in this repository. Kamino
  velocity-limit gaps are tracked by [vastsoun/newton#397](https://github.com/vastsoun/newton/issues/397).
- Material and collider-offset paths currently rely on view bindings or event terms. Their v1 tests must invoke
  the public Isaac Lab path and let that path emit `SHAPE_PROPERTIES`; direct buffer writes plus manual notify
  are reference patterns only.
- Explicit actuator `kp`/`kd` writers patch the active controller arrays directly. These gains are not
  solver-owned model properties, so this path does not require an `ACTUATOR_PROPERTIES` model-change
  notification; the remaining evidence gap is an end-to-end effort and trajectory assertion.

### PhysX

This evidence covers `isaaclab_physx`, not `isaaclab_ovphysx`.

#### Property and API mapping

- Gravity is a scene-wide PhysX property set through the simulation view. Unlike Newton's model representation,
  the PhysX runtime path does not provide independent per-environment gravity.
- Body mass, inertia, and center of mass are written through the PhysX tensor view by the public
  `set_masses_*`, `set_inertias_*`, and `set_coms_*` methods.
- Contact materials expose static friction, dynamic friction, and restitution.
- Public rest and contact offsets map directly to PhysX rest/contact offsets; they are not named margin/gap in
  the PhysX oracle.
- Joint stiffness, damping, position/velocity/effort limits, armature, and friction components are written
  through the PhysX articulation view. PhysX joint friction may expose static, dynamic, and viscous components.
- PhysX does not use Newton `ModelFlags`; successful return from a PhysX tensor setter is not equivalent to a
  Newton model-change notification.

#### Runtime behavior and gaps

- Existing PhysX asset tests provide useful read-back and partial behavioral evidence for gravity, friction,
  restitution, state, limits, and mass, but most do not isolate all three authoring paths against one physical
  oracle.
- Scene-wide gravity tests use one environment or apply the same value to every environment.
- Shape topology and joint-frame changes after initialization are not public in-place updates.

## Collected implementation gaps and observations

- **Explicit actuator gains bypass model notification by design.** `write_actuator_stiffness_to_sim` and
  `write_actuator_damping_to_sim` patch active-controller `kp`/`kd` through
  `ArticulationView.set_actuator_parameter`. They do not call `add_model_change` because these controller-owned
  gains are not solver model properties.
- **Tendon writers do not notify.** The fixed and spatial tendon setters
  (`set_fixed_tendon_*`, `write_fixed_tendon_properties_to_sim_*`, and related methods) write buffers but emit
  no `TENDON_PROPERTIES` flag.
- **Unused flags:** `JOINT_PROPERTIES` (`1 << 0`), `BODY_PROPERTIES` (`1 << 2`),
  `CONSTRAINT_PROPERTIES` (`1 << 6`), `TENDON_PROPERTIES` (`1 << 7`), `ACTUATOR_PROPERTIES` (`1 << 8`), and
  `ALL` are defined in Newton but are not currently set by Isaac Lab. This does not include
  `JOINT_DOF_PROPERTIES`, which Isaac Lab does emit for joint-property writers.
- **`SolverNotifyFlags` is deprecated.** Per `newton/_src/solvers/flags.py`, it is a deprecated alias for
  `newton.ModelFlags` (since Newton 1.3); their values are identical. The flag docstrings in `ModelFlags`
  (`enums.py`) are the authoritative list of the underlying arrays covered by each flag.
- The backend-specific files under `source/isaaclab_newton/test/assets/`, including `test_articulation.py`,
  `test_rigid_object.py`, and `test_rigid_object_collection.py`, also call `add_model_change` directly for
  `SHAPE_PROPERTIES` and `MODEL_PROPERTIES` when exercising material and gravity writes through raw Warp
  bindings. These are references for the intended write-to-notify pattern, not qualifying public-API coverage.
