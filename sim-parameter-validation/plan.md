## Isaac Lab parameter simulation validation

**Status:** Draft design, implementation in progress
**Backends in v1:** `isaaclab_physx`, Newton-MJWarp, Newton-Kamino
**Deferred backends:** `isaaclab_ovphysx`
**Last updated:** 2026-07-21

The goal of this document is to outline a set of tests that validate that Isaac Lab correctly sets physical
parameters in its simulation backends. The simulator should be treated as a black box: each test should use a
simple simulation experiment to verify that setting a parameter produces the expected physical effect.

Newton calls tests against closed-form mechanics, conservation laws, and kinematic identities *verification*
rather than real-world validation. This document uses "physical validation" for the Isaac Lab integration goal,
but preserves that distinction when citing Newton tests. Existing Newton tests establish useful solver behavior
and reference oracles; they do not establish that Isaac Lab's USD, Python configuration, and runtime-write paths
set the intended values. The Isaac Lab tests must exercise those paths directly.

### Terminology

- **Physical-behavior test:** an Isaac Lab integration test that observes simulated motion, contact, or
  equilibrium after setting a parameter.
- **Storage/notification test:** a test that checks a backend buffer or model-change notification without a
  physical observable. This is supporting evidence, not physical coverage.
- **USD-authored:** the value is present on the USD stage before the asset is loaded.
- **Python override:** Isaac Lab schema, spawn, asset, or actuator configuration authors or overrides the value
  before simulation initialization.
- **Runtime write:** a public Isaac Lab API updates an initialized simulation model.
- **Reset default:** state restored by the asset or environment reset procedure.
- **Live state:** current simulator state changed by a state writer.
- **Joint passive damping:** an uncommanded dissipative joint property.
- **Implicit drive gain:** stiffness or damping consumed directly by the physics solver's joint drive.
- **Explicit actuator gain:** gain used by an Isaac Lab or Newton actuator controller to compute effort.

### Requirements and success criteria

- **REQ-01:** Every supported parameter/backend/authoring-path cell in the coverage matrix must have a
  physical-behavior test.
- **REQ-02:** A runtime write that requires model or solver reconstruction must fail with a documented exception
  or warning. Silent acceptance is a defect and must not be recorded as supported coverage.
- **REQ-03:** Storage, read-back, and notification checks may support diagnosis but do not satisfy physical
  coverage.
- **REQ-04:** Each physical test must identify its Isaac Lab API, controlled fixture, physical observable,
  reference oracle, simulation profile, and tolerance.
- **REQ-05:** State tests must distinguish reset defaults from live-state writes and must identify whether pose
  and velocity refer to the link frame or center-of-mass frame.
- **REQ-06:** Backend-specific semantics must use backend-specific oracles rather than forcing nominally similar
  properties into one interpretation.
- **REQ-07:** Known unsupported or defective cells must be represented explicitly and linked to a test or issue
  that will detect when support changes.

The v1 design is complete when every in-scope matrix cell has one of the dispositions defined below, every
physical-test cell maps to a fixture contract, and every test has a fully specified execution profile without
relying on undocumented defaults.

### Scope

The tests should cover the following simulation backends:

- PhysX through `isaaclab_physx`
- Newton-MJWarp
- Newton-Kamino

The tests should cover the following mechanisms for setting parameters:

- The parameter is authored in USD.
- The parameter is overridden through a Python configuration in Isaac Lab.
- The parameter is updated after the simulation has started.

Not every parameter supports every mechanism. In particular, a runtime update may require solver recreation if
it changes model topology or constraint capacity. In that case a test should verify the documented error.

`isaaclab_ovphysx` is explicitly deferred from v1. It uses the PhysX solver but has a different view and
material integration layer, so passing `isaaclab_physx` tests must not be interpreted as OVPhysX coverage.

#### Non-goals for v1

- Solver verification that does not pass through an Isaac Lab authoring or runtime-write path.
- Cross-backend trajectory equality; each backend is compared with its own documented physical model.
- Performance benchmarking, policy-training parity, deformables, particles, and task-level domain
  randomization.
- Generic constraint and tendon validation. These remain deferred until Isaac Lab exposes a supported runtime
  contract and the backend notification behavior is defined.

### Parameters to cover

The following coverage is backend-independent. Each backend should test the same physical behavior wherever it
supports the parameter. Backend-specific storage, notification, and runtime-update constraints are documented
separately below.

#### Global simulation properties

- Gravity vector

#### Body initial state

- Initial body pose
- Initial body center-of-mass spatial velocity

These properties affect reset defaults. Tests should distinguish them from writes to the current simulation
state.

#### Body inertial properties

- Mass
- Inertia tensor and inertial-frame orientation
- Center-of-mass position

Internal derived values such as inverse mass and inverse inertia should be validated through physical behavior
rather than treated as independent user parameters.

#### Shape and material properties

- Shape transform relative to its body
- Shape scale or dimensions
- Collision radius
- Contact static friction
- Contact dynamic or combined friction, depending on backend
- Restitution
- Public rest/contact offsets and their backend margin/gap mapping

#### Joint geometry properties

- Parent joint frame
- Child joint frame

Joint-frame tests should validate the resulting joint axis and body motion, not only inspect stored transforms.

#### Joint degree-of-freedom properties

- Joint position/state
- Joint velocity/state
- Lower and upper position limits
- Velocity limit
- Effort limit
- Armature
- Passive joint damping
- Backend-specific joint dry-friction property
- Drive stiffness / position gain
- Drive damping / velocity gain

The tests must distinguish passive joint damping from drive damping.

#### Actuator properties

- Explicit actuator stiffness
- Explicit actuator damping

Newton's internal actuation/target mode is not in v1 because Isaac Lab does not expose a public authoring or
runtime writer for it.

#### Constraint and tendon properties

Deferred from v1. Add individual black-box cases only after Isaac Lab exposes a supported contract and each
backend documents whether the update is in-place or requires reconstruction.

### Canonical parameter registry and coverage matrix

This matrix is the canonical v1 registry. The three letters in each backend cell represent
**USD / Python override / runtime write**:

- **I:** a black-box Isaac Lab integration test is already implemented.
- **T:** a physical-behavior test is required.
- **E:** an error-path test is required because the update is not supported in place.
- **N:** Isaac Lab does not expose that authoring path.
- **D:** intentionally deferred from v1.
- **X:** a known defect or semantic mismatch blocks valid coverage; retain a strict expected-failure or issue
  reference until it is fixed.

An `I` disposition records current coverage only when the existing test uses a physical observable. Existing
storage-only tests remain `T`.

| ID | Parameter | PhysX | Newton-MJWarp | Newton-Kamino |
|---|---|---|---|---|
| SIM-01 | Gravity vector | T / T / T | T / T / T | T / T / T |
| STATE-01 | Initial/reset link pose | T / T / T | T / T / T | T / T / T |
| STATE-02 | Initial/reset COM spatial velocity | T / T / T | T / T / T | T / T / T |
| BODY-01 | Mass | T / T / T | T / T / T | T / T / T |
| BODY-02 | Inertia tensor and inertial-frame orientation | T / T / T | T / T / T | T / T / T |
| BODY-03 | Center-of-mass position | T / T / T | T / T / T | T / T / T |
| SHAPE-01 | Shape transform relative to body | T / T / N | T / T / N | T / T / N |
| SHAPE-02 | Shape scale or dimensions | T / T / E | T / T / E | T / T / E |
| SHAPE-03 | Collision radius | T / T / N | T / T / N | T / T / N |
| MAT-01 | Contact static friction | T / T / T | T / T / T | D / D / D |
| MAT-02 | Contact dynamic friction | T / T / T | D / D / D | D / D / D |
| MAT-03 | Combined contact friction `mu` | N / N / N | T / T / T | X / X / X |
| MAT-04 | Restitution | T / T / T | T / T / T | X / X / X |
| CONTACT-01 | Rest/contact offset mapped to margin/gap | T / T / T | T / T / T | T / T / T |
| JOINT-01 | Parent and child joint frames | T / T / N | T / T / N | T / T / N |
| JOINT-02 | Reset/live joint position | N / T / T | N / T / T | N / T / T |
| JOINT-03 | Reset/live joint velocity | N / T / T | N / T / T | N / T / T |
| JOINT-04 | Lower and upper position limits | T / T / T | T / T / T | I / T / X |
| JOINT-05 | Velocity limit | T / T / T | X / X / X | X / X / X |
| JOINT-06 | Effort limit | T / T / T | T / T / T | X / X / X |
| JOINT-07 | Armature | T / T / T | T / T / T | I / I / I |
| JOINT-08 | Passive joint damping | T / T / T | T / T / T | T / T / T |
| JOINT-09 | Joint dry-friction force/torque | T / T / T | T / T / T | X / X / X |
| DRIVE-01 | Implicit drive stiffness | T / T / T | T / T / T | I / I / I |
| DRIVE-02 | Implicit drive damping | T / T / T | T / T / T | I / I / I |
| CMD-01 | Feed-forward joint effort | N / N / T | N / N / T | N / N / I |
| CMD-02 | Joint velocity target | N / N / T | N / N / T | N / N / I |
| ACT-01 | Explicit actuator stiffness | N / T / T | N / T / X | N / T / X |
| ACT-02 | Explicit actuator damping | N / T / T | N / T / X | N / T / X |
| ACT-03 | Runtime actuation/target mode | N / N / N | N / N / N | N / N / N |
| DEFER-01 | Generic constraint properties | D / D / D | D / D / D | D / D / D |
| DEFER-02 | Fixed and spatial tendon properties | D / D / D | D / D / D | D / D / D |

The matrix intentionally separates contact friction from joint friction. Newton-MJWarp exposes one contact
coefficient, `shape_material_mu`, and ignores a distinct dynamic-friction range. Newton joint friction is an
absolute dry-friction force or torque `[N or N·m, depending on joint type]`, despite the `coefficient` name in
the common Isaac Lab API.

`ACT-03` is retained to document the gap in the original scope, but it is not v1 coverage: Isaac Lab currently
has no public writer for Newton's internal `joint_target_mode`. Likewise, the `X` runtime-limit cells record
current silent or ineffective writes; they must become `E` when a documented error exists or `T` when physical
support is implemented.

#### `X`-cell issue register

The register below covers every `X` cell in the matrix. An issue is qualifying only when its scope matches the
backend, parameter, and authoring path. Related issues provide implementation context but do not satisfy
`REQ-07`.

| Parameter ID | Backend and `X` authoring paths | Qualifying issue | Related context |
|---|---|---|---|
| `MAT-03` | Newton-Kamino: USD, Python, runtime | TODO: exact Kamino contact-friction issue required | None |
| `MAT-04` | Newton-Kamino: USD, Python, runtime | TODO: exact Kamino restitution-support issue required | [vastsoun/newton#375](https://github.com/vastsoun/newton/issues/375) tracks restitution artifacts with positive contact gap |
| `JOINT-04` | Newton-Kamino: runtime | TODO: exact runtime position-limit issue required | [vastsoun/newton#104](https://github.com/vastsoun/newton/issues/104) tracks the general model-change integration |
| `JOINT-05` | Newton-MJWarp: USD, Python, runtime | TODO: exact upstream Newton velocity-limit issue required | None |
| `JOINT-05` | Newton-Kamino: USD, Python, runtime | [vastsoun/newton#397](https://github.com/vastsoun/newton/issues/397) | [newton-physics/newton#161](https://github.com/newton-physics/newton/issues/161) added model storage; does not establish Kamino enforcement |
| `JOINT-06` | Newton-Kamino: USD, Python, runtime | [vastsoun/newton#398](https://github.com/vastsoun/newton/issues/398) | [newton-physics/newton#161](https://github.com/newton-physics/newton/issues/161) added model storage; does not establish Kamino enforcement |
| `JOINT-09` | Newton-Kamino: USD, Python, runtime | [vastsoun/newton#383](https://github.com/vastsoun/newton/issues/383) | None |
| `ACT-01`, `ACT-02` | Newton-MJWarp: runtime | TODO: exact Isaac Lab or upstream Newton actuator-update issue required | None |
| `ACT-01`, `ACT-02` | Newton-Kamino: runtime | TODO: exact Isaac Lab/Kamino actuator-update issue required | [vastsoun/newton#104](https://github.com/vastsoun/newton/issues/104) tracks general model-change integration |

[vastsoun/newton#385](https://github.com/vastsoun/newton/issues/385) is closed historical evidence for allocating
Kamino's implicit joint-dynamics equations when gains may change at runtime. It explains the pre-allocation used
by the existing `DRIVE-01`, `DRIVE-02`, and `JOINT-07` tests, but it is not a qualifying issue for the explicit
actuator `X` cells.

#### Temporary Newton target-mode workaround

[IsaacLab#6649](https://github.com/isaac-sim/IsaacLab/issues/6649) tracks that `ImplicitActuatorCfg` writes
Newton stiffness and damping without authoring a compatible `joint_target_mode`. Newton infers this mode while
importing the USD drive, so a zero-gain USD drive remains in `EFFORT` mode even when Python configuration or a
runtime write later supplies non-zero gains. Kamino then stores and reports the new gains but omits the inactive
position or velocity term from its dynamics; MJWarp cannot create the missing target-actuator topology after
solver construction.

Until the issue is resolved, the Kamino single-DOF test authors a non-zero USD stiffness and/or damping for
Python-configuration and runtime cases solely to establish the required target mode before solver construction.
The configured or runtime value remains the parameter under test. Runtime armature cases similarly author a
smaller non-zero armature to preserve dynamic-constraint topology. Remove these seed values when
`ImplicitActuatorCfg` explicitly establishes the target mode, then rerun all three authoring paths from a
zero-gain USD baseline. This workaround is test scaffolding, not evidence that an otherwise incompatible USD
drive is supported.

For `CMD-02`, the test authors zero stiffness and non-zero damping so Newton imports the drive in pure
`VELOCITY` mode. This validates the public velocity-target command without requiring
`force_position_velocity_actuation=True` on `builder.add_usd()`. It does not establish that the importer selects
combined `POSITION_VELOCITY` mode when both gains are non-zero; that remains part of the target-mode integration
tracked by [IsaacLab#6649](https://github.com/isaac-sim/IsaacLab/issues/6649).

### Planned physical-behavior tests

#### Fixed-base single-DOF articulation

Joint degree-of-freedom and actuator parameters will be validated with a procedurally constructed, fixed-base,
single-DOF articulation. Separate cases will use revolute and prismatic joints. The moving body will have an
explicitly authored mass, diagonal inertia tensor, and center of mass on the joint axis, making the effective
joint-space inertia known. Gravity and collisions will be disabled so that the observed joint motion is caused
only by the parameter or command under test.

For drive stiffness, drive damping, armature, feed-forward effort, and velocity targets, the test will start the
joint from rest, advance the simulation by one step, and compare the resulting joint position and velocity with
the expected single-DOF dynamics. Each supported parameter-authoring path (USD, Python actuator configuration,
and runtime write) will be tested separately. Position limits will instead use a multi-step response: a drive
will command motion beyond the limit, and the bounded trajectory will be compared with a control case whose
limit lies beyond the commanded target.

For a prismatic joint with effective mass \(m_\mathrm{eff}\), the one-step reference is
\(v_1 = v_0 + F\,\Delta t / m_\mathrm{eff}\). For a revolute joint with effective inertia
\(I_\mathrm{eff}\), it is \(\omega_1 = \omega_0 + \tau\,\Delta t / I_\mathrm{eff}\). A position/velocity
drive supplies \(\tau = k_p(q_t-q) + k_d(\dot q_t-\dot q)\), subject to the configured effort limit. These are
the same reference equations used by Newton's joint-actuation, joint-drive, and effort-clamping tests. Armature
is included in the effective joint-space inertia and therefore must reduce the acceleration under otherwise
identical effort.

Variants of the same fixture can cover additional properties without introducing contacts or multi-body
coupling. A prismatic joint converts body mass and gravity projected onto the joint axis into a directly
observable linear acceleration. A revolute joint similarly exposes the body's moment of inertia; moving its
center of mass away from the joint axis also permits a gravity-driven center-of-mass test. Initial joint state,
joint frames, passive losses, command limits, and actuation modes can all be isolated by choosing an appropriate
unforced or driven response. These variants will continue to change one parameter at a time and compare against
an otherwise identical control case.

| Parameter or command | Physical-behavior test |
|---|---|
| Gravity vector | Unactuated acceleration with gravity projected onto the prismatic joint axis |
| Body mass | Prismatic-joint acceleration under a known joint effort |
| Body inertia tensor | Revolute-joint acceleration under a known joint effort |
| Body center-of-mass position | Gravity-driven revolute motion with a known center-of-mass offset |
| Parent and child joint frames | Direction and magnitude of body motion under a known joint effort |
| Joint position/state | Unforced step after writing a non-zero joint position |
| Joint velocity/state | Unforced step after writing a non-zero joint velocity |
| Drive stiffness / position gain | Single-step position-target response from rest |
| Drive damping / velocity gain | Single-step position-target response from rest with fixed stiffness |
| Armature | Single-step position-target response from rest with fixed stiffness |
| Lower and upper position limits | Multi-step driven motion with active-limit and inactive-limit cases |
| Velocity limit | Sustained drive beyond the configured maximum velocity |
| Effort limit | Acceleration under a commanded effort above the configured maximum |
| Passive joint damping | Deceleration from a known initial velocity with the drive disabled |
| Joint friction coefficient | Breakaway or deceleration response with the drive disabled |
| Explicit actuator stiffness | Single-step position-target response from rest |
| Explicit actuator damping | Single-step velocity-target response from rest |
| Feed-forward joint effort target | Single-step acceleration from rest, with and without implicit drive dynamics |
| Joint velocity target | Single-step velocity-target response from rest with non-zero drive damping |

#### Free-body initial-state and inertial fixture

A collision-free body will cover initial body pose, spatial velocity, and body inertial properties independently
of articulation state. With uniform gravity, its position and velocity will be compared at several checkpoints
with
\(x(t)=x_0+v_0t+\tfrac12gt^2\) and \(v(t)=v_0+gt\). A zero-gravity variant will apply a known force or torque
and compare the resulting velocity with \(v_1=v_0+F\Delta t/m\) or
\(\omega_1=\omega_0+I^{-1}\tau\Delta t\).

For center-of-mass tests, the body will have a non-zero authored offset. Pure angular velocity with zero linear
center-of-mass velocity should rotate the body without translating its center of mass; a pure force applied at
the center of mass should translate it without inducing rotation. Reset-default tests will first disturb the
current state and then reset, so an authored initial pose or velocity cannot be confused with a write to the live
state.

#### Contact and shape fixtures

Contact parameters require separate fixtures because the single-DOF articulation intentionally disables
collisions:

- **Static friction:** place a box on an inclined plane. It should remain at rest below the critical angle
  \(\theta_c=\arctan(\mu_s)\) and slide above it, using a dead band around \(\theta_c\).
- **Dynamic friction:** give a box initial speed \(v_0\) on level ground and compare its stopping distance with
  \(d=v_0^2/(2\mu_d|g|)\). Use matching material values on both contacting shapes or account explicitly for the
  backend's material-combination rule.
- **Restitution:** drop a sphere from height \(h_0\) and compare its first rebound height with
  \(h_r=e^2h_0\). Disable friction and choose contact settings that do not add appreciable damping beyond the
  restitution model.
- **Shape transform and dimensions:** compare first-contact time or resting pose for otherwise identical bodies
  whose local shape offset, scale, dimensions, or collision radius differ by a known amount.
- **Contact margin and gap:** use two slowly approaching shapes and measure the separation at first contact.
  Treat these as collision-generation parameters, not as restitution or penetration-depth controls.

The friction and restitution cases should use multi-step trajectories and tolerances tied to the time step and
contact model. Shape geometry, margin, and gap cases may use paired control scenes when a backend-independent
closed form is unavailable.

#### Fixture contracts

Every matrix cell marked `T` or `I` maps to one of these contracts. A backend implementation may split a
contract into multiple pytest cases, but it must preserve the listed controlled variables and observable.

| Fixture ID | Parameters | Observable and oracle | Required controls |
|---|---|---|---|
| FIX-FREE-FALL | SIM-01, STATE-01, STATE-02 | Link/COM position and velocity at fixed checkpoints; backend's discrete gravity update | Collision-free body; authored and live-state variants |
| FIX-WRENCH-LIN | BODY-01 | COM velocity increment under known force, `Delta v = F dt / m` for the pinned discrete step | Zero gravity, zero initial velocity |
| FIX-WRENCH-ANG | BODY-02 | Angular-velocity increment from `I_world^-1 tau`; include one non-spherical inertia and rotated inertial frame | Zero gravity and force; torque about at least two axes |
| FIX-COM | BODY-03 | Pure COM force produces translation without rotation; gravity torque follows `r_com x mg` | Paired zero-offset body |
| FIX-DOF-STEP | DRIVE-01, DRIVE-02, JOINT-07, CMD-01, CMD-02 | Joint position and velocity after one step using the backend-specific implicit or explicit update | Fixed-base revolute and prismatic variants; gravity/collision off |
| FIX-JOINT-STATE | JOINT-02, JOINT-03 | State immediately after write and after one unforced step | Separate reset-default and live-write cases |
| FIX-JOINT-FRAME | JOINT-01 | World-space motion axis and displacement under known effort | Identity-frame control and one rotated/translated frame at a time |
| FIX-LIMIT-POS | JOINT-04 | Maximum and settled joint position for active and inactive limits | Same drive and initial state; lower and upper limits tested separately |
| FIX-LIMIT-VEL | JOINT-05 | Sustained maximum speed and braking response after starting above the limit | Effort high enough that velocity, not effort, is limiting |
| FIX-LIMIT-EFFORT | JOINT-06 | Initial acceleration under sub-limit and over-limit commands | Known effective inertia; drive terms disabled or included in oracle |
| FIX-PASSIVE | JOINT-08, JOINT-09 | Velocity decay or breakaway threshold using backend-specific damping/friction law | Drive disabled; zero-loss paired control |
| FIX-ACTUATOR | ACT-01, ACT-02 | Controller effort and resulting trajectory after gain update | Separate explicit actuator and implicit drive cases |
| FIX-FRICTION-STATIC | MAT-01, MAT-03 | Rest/sliding classification on both sides of the critical incline angle | Identical material values on both shapes unless combination is under test |
| FIX-FRICTION-DYNAMIC | MAT-02, MAT-03 | Stopping distance and velocity decay under the backend's contact law | Level plane, zero restitution, known initial speed |
| FIX-RESTITUTION | MAT-04 | First post-impact apex or normal-velocity ratio | Zero friction; first impact only |
| FIX-SHAPE-CONTACT | SHAPE-01, SHAPE-02, SHAPE-03 | First-contact step and resting separation relative to a control shape | Slow approach; fixed body poses and velocities |
| FIX-CONTACT-OFFSET | CONTACT-01 | First generated-contact step and equilibrium separation | Paired baseline; report public rest/contact offsets and mapped margin/gap |

The measurement API and frame are part of the contract. Pose and velocity assertions must state `link` or
`COM`; joint-frame assertions use world-space motion; contact assertions use a backend-supported contact
observable rather than inferring contact from a solver array alone.

#### Deterministic simulation profiles

- **PROFILE-DOF:** one environment, float32, `dt = 1/120 s`, one substep, gravity and collisions disabled,
  fixed base, explicit mass/inertia, and the backend integrator named in the test. Newton-Kamino uses
  `integrator="euler"`, zero bilateral Baumgarte terms for one-step tests, and CUDA graphs disabled.
- **PROFILE-FREE:** one environment, float32, `dt = 1/120 s`, one substep, collisions disabled, and no damping,
  gyroscopic, sleep, or stabilization effects unless included in the oracle.
- **PROFILE-CONTACT:** one environment, float32, a backend-pinned integrator and substep count, sleep disabled,
  zero unrelated damping, and a fixed seed. The test records the approach speed and detects contact at step
  resolution.
- **PROFILE-RUNTIME:** run a baseline phase, restore an identical controlled state, perform exactly one public
  runtime write, allow the API to issue its required notification, and run the changed phase. Do not manually
  notify in a test of a public writer.

Tests use deterministic authored values rather than random samples. If randomization machinery is the public
write path under test, set the seed to `0` and assert the sampled value as well as its physical effect. GPU
tests must be repeatable across three consecutive runs before their tolerance is accepted.

#### Tolerance policy

Tolerances are attached to fixture/backend pairs, not copied between backends:

| Fixture class | Initial acceptance rule |
|---|---|
| Kamino one-step DOF | `rtol = 5e-3`, `atol = 2e-4`, matching the existing float32 P-ADMM single-step test |
| Other one-step analytical cases | `abs(error) <= max(50 eps scale, 10 solver_residual scale, truncation_bound)`; if the backend exposes no residual, establish `truncation_bound` with a `dt` versus `dt/2` convergence check |
| Free-body checkpoints | Velocity uses the one-step analytical rule; position includes the pinned integrator's accumulated truncation bound rather than comparing blindly with the continuous equation |
| Contact event time | Measured event differs from the predicted/control event by at most one simulation step |
| Contact separation | `abs(error) <= max(2 mm, 2 v_approach dt)` unless the backend documents a larger solver tolerance |
| Static friction | Test on both sides of a dead band of at least `max(2 degrees, 5% of theta_c)`; no assertion exactly at the threshold |
| Dynamic friction | Stopping-distance relative error at most `10%`, plus one-step distance `v dt` |
| Restitution | First-apex relative error at most `10%`, plus the height uncertainty caused by one-step impact timing |
| Paired controls | The changed case must exceed both the analytical tolerance and five times the repeated-run control spread |

Any tolerance looser than these initial rules requires a backend-specific rationale in the test. Failures must
report the parameter ID, backend, authoring path, API, profile, `dt`, substeps, measured value, expected value,
absolute/relative error, and tolerance.

#### Test execution pattern

Each fixture should have one canonical scene and observable, then be parameterized over the supported authoring
paths rather than reimplemented three times:

1. **USD-authored:** author the parameter in the asset, load it through Isaac Lab, reset, and run the physical
   measurement.
2. **Python override:** load the same baseline asset, apply the Isaac Lab configuration override before
   simulation starts, reset, and repeat the same measurement.
3. **Runtime update:** run a baseline phase, update one parameter through the Isaac Lab writer, perform the
   required notification, and run a second phase from a controlled state. Compare both phases with their
   respective references.

For runtime changes, the test must not infer success solely from a Newton array value. It must observe the
changed trajectory, contact event, or equilibrium. If a requested runtime change requires solver recreation,
do not recreate the solver and rerun the physical measurement. The public write must raise the documented
exception or warning,
which is tested as an `E` matrix cell. A silent or ineffective write is an `X` defect, not an expected success.
If the change is unsupported because it changes topology or constraint capacity, assert the documented error
and do not classify that case as physical coverage.

All fixtures should report the authored path, backend, time step, substep count, measured quantity, expected
quantity, and tolerance on failure. Use deterministic initial states and disable unrelated effects such as
collisions, gravity, damping, or drives unless they are part of the parameter under test.

### Supporting evidence

The existing-test inventory, parameter-to-evidence map, backend storage and notification mappings, current
implementation status, and collected integration gaps live in the
[background and evidence document](background.md).

### Isaac Lab authoring-path mapping

The test adapters use these public integration paths. Backend-private arrays may be inspected for diagnostics
but are not the action under test.

| Parameter family | USD/Python path | Runtime path |
|---|---|---|
| Gravity | Authored scene gravity or `SimulationCfg.gravity` | `randomize_physics_scene_gravity` |
| Reset pose/velocity | USD state plus asset `init_state` configuration | `write_root_link_pose_to_sim_index`, `write_root_com_velocity_to_sim_index`, `write_joint_position_to_sim_index`, and `write_joint_velocity_to_sim_index` |
| Mass/inertia/COM | USD Mass API or Isaac Lab mass-property schema | `set_masses_index`, `set_inertias_index`, and `set_coms_index` |
| Shape dimensions | USD geometry or spawn/schema configuration | `randomize_rigid_body_scale` before simulation only; an after-start call is an error-path case |
| Materials | USD material binding or material configuration | `randomize_rigid_body_material` |
| Collider offsets | USD/PhysX/Newton collision schemas | `randomize_rigid_body_collider_offsets` |
| Joint limits/properties | USD joint/drive schemas and `ImplicitActuatorCfg` | `write_joint_position_limit_to_sim_index`, `write_joint_velocity_limit_to_sim_index`, `write_joint_effort_limit_to_sim_index`, `write_joint_armature_to_sim_index`, `write_joint_stiffness_to_sim_index`, and `write_joint_damping_to_sim_index` |
| Joint commands | Not persistent authored parameters | `set_joint_effort_target_index`, `set_joint_position_target_index`, and `set_joint_velocity_target_index`, followed by `write_data_to_sim` |
| Explicit actuator gains | Actuator configuration | `write_actuator_stiffness_to_sim` and `write_actuator_damping_to_sim` |

Index writers are the canonical physical-test path. Their mask counterparts remain covered by focused backend
API tests unless a mask-only graphed pipeline has distinct physical behavior.

### Backend test constraints

- **PhysX:** gravity cases use one environment or one scene-wide value. Contact cases pin material combination
  modes and use matching values on both shapes unless combination itself is under test. Joint-friction cases
  identify whether static, dynamic, or viscous friction is exercised.
- **Newton-MJWarp:** contact cases use the single combined `mu` interpretation. Runtime material and
  collider-offset tests invoke the public Isaac Lab event/API path and must not substitute a raw buffer write
  plus manual notification. Joint-friction cases use absolute dry-friction force/torque semantics.
- **Newton-Kamino:** blocked limit, friction, restitution, and actuator cells remain `X`; tests must not encode
  silent or ineffective writes as expected behavior. Single-step joint cases use the pinned Kamino profile and
  oracle defined above.

### Proposed test architecture

Keep backend launch and adapter code in the backend packages while sharing only backend-neutral fixture
construction and oracle logic:

```text
source/isaaclab/test/physics/parameter_validation/
  fixtures.py              # procedural scene descriptions and controlled values
  oracles.py               # pure discrete mechanics and tolerance helpers
source/isaaclab_physx/test/physics/parameter_validation/
  conftest.py              # PhysX profiles and public API adapters
  test_free_body.py
  test_joint_dynamics.py
  test_contact_parameters.py
source/isaaclab_newton/test/physics/parameter_validation/
  conftest.py              # MJWarp/Kamino profiles and public API adapters
  test_free_body.py
  test_joint_dynamics.py
  test_contact_parameters.py
```

The existing Kamino test may remain in its current location, but new shared coverage should use the layout
above. A small case descriptor should carry `parameter_id`, `backend`, `authoring_path`, `profile`, `act`,
`observe`, `predict`, and `tolerance`; it is test infrastructure, not a public Isaac Lab API.

Tests are parameterized over authoring path only when the path invokes a genuinely different Isaac Lab
integration route. Index and mask writers require focused selection/coherence tests, but one physical test per
runtime property may use the index writer as the canonical path. Mask-writer correctness remains part of the
backend asset API suites.

### Traceability

The matrix parameter ID is the stable key used in test IDs and failure messages. Traceability is:

`requirement -> parameter ID -> matrix disposition -> fixture ID -> backend test -> upstream evidence`

| Requirement | Design artifact | Test evidence required |
|---|---|---|
| REQ-01 | Coverage matrix and fixture contracts | One physical test for every `T`; existing test path for every `I` |
| REQ-02 | `E` and `X` matrix cells plus linked backend evidence | Exception/warning assertion for `E`; strict expected-failure or linked issue for `X` |
| REQ-03 | Linked background evidence and implementation-status inventory | Physical observable in addition to any buffer/read-back assertion |
| REQ-04 | Fixture contracts, profiles, and tolerance policy | Case descriptor and diagnostic assertion context |
| REQ-05 | STATE-01/02 and JOINT-02/03 rows | Separate reset/live and link/COM cases |
| REQ-06 | Coverage-matrix friction taxonomy and linked backend evidence | Backend-specific oracle selected by fixture adapter |
| REQ-07 | Gaps, risks, and unresolved decisions | Explicit deferred/blocked test or issue reference |

Upstream Newton tests are oracle evidence and should be cited with the Newton version or commit used by Isaac
Lab. They are not Isaac Lab integration coverage. In-repo references must include the backend package in their
path.

Route issue references by ownership: Kamino solver defects to
[vastsoun/newton](https://github.com/vastsoun/newton/issues), upstream non-Kamino Newton defects to
[newton-physics/newton](https://github.com/newton-physics/newton/issues), and Isaac Lab adapter/public-API
defects to [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab/issues).

### Implementation phases

1. **Phase 0 — establish the baseline:** classify the existing Kamino single-DOF test under the IDs in this
   document and add issue references for its `X` cells.
2. **Phase 1 — non-contact analytical core:** implement free-body, wrench, COM, state, and single-DOF cases
   across the scoped backends.
3. **Phase 2 — limits, frames, and passive effects:** add position/velocity/effort limits, joint frames,
   passive damping, and backend-specific joint friction.
4. **Phase 3 — contact parameters:** add friction, restitution, geometry, and offset fixtures after their
   backend-specific contact semantics are resolved.
5. **Phase 4 — blocked runtime contracts:** convert `X` cells to `E` or `T` as backend defects are resolved.
   Deferred constraints, tendons, and OVPhysX require a separate design revision.

Phase 0 test classification is implemented for `DRIVE-01`, `DRIVE-02`, `JOINT-04`, `JOINT-07`, `CMD-01`, and
`CMD-02`. The `X`-cell register is complete, but Phase 0 does not fully satisfy `REQ-07` while qualifying issues
are still missing for Kamino contact/restitution, Kamino runtime position limits, MJWarp velocity limits, and
runtime actuator gain updates.

The appropriate CI selection, gating policy, and scheduling are intentionally left to the implementation
change. Contact tests are likely to be less reliable in CI because their thresholds depend on integrator,
device, and step-resolution behavior. Their stability should be measured with repeated runs before deciding
where or whether they gate changes.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Contact thresholds are flaky across devices | Use paired controls, step-resolution bounds, fixed profiles, and three-run tolerance qualification |
| A test passes by reading the value it wrote | Require a trajectory/contact/equilibrium assertion for all `T` and `I` cells |
| Continuous equations disagree with the discrete integrator | Pin the integrator and use its discrete update or an explicit truncation bound |
| Runtime notifications silently leave derived data stale | Compare baseline and changed phases through the public writer; classify silent writes as `X` |
| One name hides different backend semantics | Keep separate matrix rows and backend-specific oracles for contact friction, joint friction, and actuator gains |
| Contact behavior is flaky in CI across devices or solver versions | Keep one environment, use paired controls and repeated-run qualification, and decide CI gating separately |
| Upstream Newton behavior changes | Pin evidence to the Isaac Lab dependency version and make strict expected failures turn into reviewable XPASS results |

### Unresolved decisions and exit criteria

These decisions do not block the document structure, but each blocks promotion of the named matrix cells:

| Decision | Owner role | Exit criterion |
|---|---|---|
| Kamino runtime position limits: support or explicit rejection | Newton/Kamino integration maintainer | Physical limit test passes or public writer emits documented error |
| MJWarp/Kamino restitution support | Newton contact maintainer | Backend-specific rebound oracle and qualifying Isaac Lab test |
| Newton explicit actuator gain notification | Newton actuator integration maintainer | Runtime writer updates the active controller and `FIX-ACTUATOR` passes |
| Kamino joint-friction mapping | Newton/Kamino integration maintainer | Dry-friction semantics implemented or public API documents a distinct viscous parameter |
| MJWarp velocity-limit enforcement | Newton/MJWarp integration maintainer | Sustained-drive and above-limit braking behavior are documented and tested |
| Generic constraints and tendons | Asset API maintainers | Public authoring/runtime contract plus backend reconstruction/error semantics |
| OVPhysX inclusion | OVPhysX maintainers | Separate backend mapping and support scope approved for a follow-up design |

When an exit criterion is met, update the matrix disposition, backend note, fixture mapping, and implementation
status in the same change. No unresolved decision may be hidden behind a skip without an issue or explicit
deferred disposition.
