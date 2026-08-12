# Joint and Actuator Property Ownership Design

## Context

PR 6839 gives each articulation an `ActuatorCollection` and aggregates compatible
Lab actuator groups into private execution batches. The execution path is now a
natural owner for actuator-model parameters, but construction still carries a
legacy ambiguity:

- `ActuatorBaseCfg` mixes actuator-model parameters with joint/solver overrides.
- `ActuatorBase` retains resolved armature, friction, solver effort limits, and
  solver velocity limits after those values have been written to the backend.
- Aggregated executors concatenate those joint properties even though actuator
  `compute()` does not consume them.
- `effort_limit_sim` and `velocity_limit_sim` name a destination rather than the
  physical object that owns the value.
- A full-articulation group may retain a view of a backend joint buffer while an
  indexed group receives a copy, making ownership depend on joint selection.

This design separates joint state from actuator state while retaining actuator
groups as a convenient configuration-time joint selector.

## Decision

The articulation is the sole runtime owner of joint and solver properties.
Ordinary actuator objects own only parameters used by their actuator model,
commands, scratch state, and telemetry.

Joint properties present in an actuator configuration are construction-time
overrides. Construction resolves them for the group's joints, writes them through
the articulation's backend-neutral joint-property interface, and does not retain
them on the resulting ordinary actuator object.

`ImplicitActuator` is the deliberate exception. The simulation solver executes
that actuator, so its drive stiffness, damping, and effort limit bind to the
corresponding live articulation state rather than independent actuator-owned
copies. The joint velocity limit remains articulation-owned and is available
through `ArticulationData` and the deprecated compatibility accessor.

## Property domains

| Domain | Canonical values | Runtime owner |
|---|---|---|
| Joint/solver | Joint effort and velocity limits, armature, static/dynamic/viscous friction | `ArticulationData` and the active backend |
| Explicit actuator model | Effort clipping, rated joint-side velocity, stiffness, damping, saturation, delay, network state, transmission parameters | The named actuator, its private aggregated executor, or a backend-native controller |
| Implicit solver drive | Stiffness, damping, and effort limit | Articulation state, projected through `ImplicitActuator` |
| Commands and telemetry | Raw and processed commands, computed and applied effort, soft-limit and gear-ratio compatibility projections | `ActuatorCollection` and execution actuators |

The same noun may legitimately appear in both the joint and actuator domains. An
explicit motor's `effort_limit` clips the model output; `joint_effort_limit`
constrains effort in the solver. An explicit motor's `velocity_limit` describes
its rated joint-side speed and may shape its torque curve;
`joint_velocity_limit` requests a solver constraint whose enforcement remains
backend-dependent.

Armature and friction remain valid configuration-time joint overrides, but they
are not actuator execution parameters and are not stored on ordinary runtime
actuators.

## Configuration API and deprecation

`ActuatorBaseCfg` retains the convenience of applying joint-property overrides
using the actuator group's joint expressions. Its canonical limit fields become:

- `effort_limit`: actuator-model effort limit.
- `velocity_limit`: actuator-model rated joint-side velocity.
- `joint_effort_limit`: joint/solver effort limit.
- `joint_velocity_limit`: joint/solver velocity limit.

The existing `effort_limit_sim` and `velocity_limit_sim` fields remain deprecated
configuration aliases for `joint_effort_limit` and `joint_velocity_limit` through
the 3.x release line. They do not become properties on `ActuatorCollection`.

Alias resolution follows one rule:

- When only the deprecated alias is set, emit `DeprecationWarning` and forward
  its value to the canonical `joint_*` field.
- When both names are set to equivalent values, emit the warning and use the
  canonical field.
- When both names are set to different values, raise `ValueError` during
  articulation construction.

Warnings identify the exact replacement and their planned removal in 4.0. The
deprecated names remain serializable and accepted by existing asset
configurations during the migration cycle.

`ImplicitActuatorCfg` applies two compatibility rules:

- `effort_limit` retains its existing deprecated implicit behavior as an alias
  for `joint_effort_limit`. Conflicting values fail during construction.
- `velocity_limit` retains its existing soft-limit behavior and does not set the
  solver clamp. When used on an implicit actuator, its warning states that
  `joint_velocity_limit` is required to request a solver limit. It is not
  silently reinterpreted during the 3.x line.

When an implicit configuration leaves `velocity_limit` unset, its soft-limit
projection retains current behavior by taking a one-time copy of the resolved
`joint_velocity_limit`. When both are configured, they remain independent. Later
joint-limit writes do not alter the soft-limit projection.

## Construction data flow

For every non-overlapping actuator group, construction performs these phases:

1. Resolve the group's articulation joint indices.
2. Read authored joint defaults from the articulation.
3. Resolve joint-property overrides independently from actuator-model defaults.
4. Write resolved joint properties through the articulation control interface.
5. Construct the actuator with only its model parameters and required runtime
   state.
6. Aggregate compatible stateless actuators using only the concrete class's
   declared execution parameters.

Explicit actuators retain current default precedence without sharing storage:

- An unset actuator `effort_limit` takes a one-time copy of the authored joint
  effort limit. It does not inherit a configured `joint_effort_limit` override.
- An unset actuator `velocity_limit` takes a one-time copy of the resolved
  `joint_velocity_limit`, including a configured override.
- An unset explicit `joint_effort_limit` remains the existing large default that
  avoids a second solver clamp, while an unset `joint_velocity_limit` retains the
  authored joint value.

Later randomization of `ArticulationData` joint limits must not change these
explicit actuator-model defaults.

The resolved joint-property payload belongs to construction. The existing
`ActuatorJointProperties` extension contract may continue to carry defaults from
a backend into collection construction. The joint-property writer is changed to
accept the resolved construction payload and group indices directly rather than
reading solver properties back from a completed actuator. Those values must not
survive as fields on ordinary actuator groups or execution batches.

## Runtime actuator contract

`robot.actuators` remains a mapping from configured group names to actuator
objects. It exposes no collection-wide `*_sim` or `joint_*` parameter view.

For an ordinary Lab or backend-native explicit actuator, a named group exposes
its model parameters and outputs, apart from the temporary deprecated joint
projections described below. Aggregated named groups remain stable views into
the private executor's model tensors. Solver-only values are removed from
execution-parameter declarations and are never concatenated into an executor.

Live joint and solver state is read through `robot.data`, including
`joint_effort_limits`, `joint_vel_limits`, `joint_armature`, and the supported
joint-friction properties. Runtime mutation continues through articulation joint
writers and existing randomization helpers.

The joint-only `ActuatorBase` attributes remain for the deprecation cycle as
compatibility accessors on individual named groups:

- `effort_limit_sim`
- `velocity_limit_sim`
- `armature`
- `friction`
- `dynamic_friction`
- `viscous_friction`

They emit `DeprecationWarning` and resolve the selected joints from the current
articulation buffers on each access. Contiguous selections may return a direct
view; arbitrary indexed selections may return a materialized projection, but no
persistent actuator-owned mirror is allocated. Their replacements are the
corresponding `robot.data` joint properties, including `joint_effort_limits`,
`joint_vel_limits`, `joint_armature`, and backend-supported friction properties.
No canonical `joint_*` accessor is added to `ActuatorCollection` or to ordinary
actuator groups. Direct mutation of a materialized compatibility projection is
not a supported solver write; users write through the articulation API.

## Implicit actuator binding

`ImplicitActuator` has no independent controller implementation: the backend
solver drive is the actuator. Its `stiffness`, `damping`, and `effort_limit`
runtime values therefore resolve from the articulation's live joint-drive
state. The bare `velocity_limit` retains the existing soft-limit compatibility
semantics during the 3.x line; it is not a live solver velocity-limit view.
`velocity_limit_sim` remains a deprecated projection of
`ArticulationData.joint_vel_limits` until 4.0.

A private implicit binding stores the canonical articulation arrays and the
group-to-articulation joint mapping. Full, contiguous, or strided selections use
framework views where representable. Arbitrary indexed selections are logical
projections: public reads gather the current values, while the execution kernel
uses the articulation-wide backing arrays and joint mapping directly. The hot
path does not allocate or gather an intermediate parameter tensor.

The binding has the following invariants:

- The articulation has one canonical logical value for each property.
- Reads through the implicit actuator reflect runtime articulation writes.
- Runtime setters and randomization helpers write through the articulation and
  are reflected by subsequent implicit actuator reads and execution.
- Reset and backend buffer rebinds refresh the binding when required; they do
  not create actuator-owned mirrors.
- Implicit telemetry uses these live gains and limits when approximating effort.

The canonical logical owner is the user-order `ArticulationData` property.
Backends may retain solver-order buffers, native model arrays, or synchronization
mirrors required by their implementation. The actuator binding targets the
backend-neutral articulation property and does not expose backend storage.

Armature and friction are still ordinary joint properties. They are not
canonical `ImplicitActuator` parameters merely because their construction
overrides were grouped with its configuration; only the deprecated non-owning
compatibility projections remain during the migration cycle.

## Backend-native actuators

Native Newton controller parameters remain owned by Newton's controller storage.
Lab-side named-group parameters that support runtime mutation are group-shaped
projections of that storage, not a second canonical copy. Gain reads resolve the
current controller value, while gain randomization and writes route through the
collection's native backend hook. The randomization path must not mutate a Lab
mirror and then forward it. Immutable controller parameters may remain resolved
configuration snapshots on the facade because they have no public runtime write
contract. An indexed projection may materialize on public access, and direct
in-place mutation of such a materialized tensor is not a native-controller write.
Joint-property overrides are written independently to the Newton model. Native
controller storage must never become the owner of joint armature, joint friction,
or joint solver limits.

The ownership contract is backend-neutral even where enforcement differs. In
particular, `joint_velocity_limit` remains a requested solver value: PhysX and
supported Newton solvers may enforce it, while a backend that does not enforce
it must still report the configured joint state accurately and document the
limitation.

## Documentation and migration

The implementation updates the public actuator concepts guide, articulation
configuration guide, API docstrings, 3.0 migration guide, and changelog fragment.
The documentation must state:

- which parameters describe a joint/solver and which describe an actuator model;
- that joint fields in an actuator configuration are applied during
  construction rather than owned by the runtime actuator;
- that `ActuatorCollection` does not expose `*_sim` or joint-property buffers;
- that implicit actuators intentionally bind to live canonical
  `ArticulationData` properties;
- that actuator and joint limits may have different values and enforcement;
- how to replace `effort_limit_sim` and `velocity_limit_sim` with the canonical
  `joint_*` configuration names and the existing `ArticulationData` accessors.

Examples use the canonical names. Deprecated names appear only in migration and
alias tables, compatibility docstrings, and deprecation notes.

## Validation

Tests focus on ownership boundaries rather than duplicating actuator mathematics:

- Configuration aliases forward, warn, and reject conflicting values; legacy
  runtime joint-property accessors warn and return current articulation
  projections.
- Explicit actuator parameters do not share storage with articulation joint
  buffers, including full-articulation and indexed groups.
- Joint-property runtime randomization does not alter explicit actuator-model
  limits.
- Implicit actuator bindings read the current articulation values for contiguous
  and indexed groups, follow backend rebinds, and add no allocation to execution
  or telemetry kernels. Explicit public reads of indexed compatibility
  projections may materialize a result.
- Aggregated executors contain only the parameters declared by the concrete
  actuator and preserve existing numerical outputs.
- PhysX, Newton, and OVPhysX construction write the same resolved joint values as
  before the rename.

Existing actuator curve, command routing, and backend integration tests are
updated or consolidated where they already cover these paths. The change does
not add a parallel exhaustive suite.

## Non-goals

- A collection-wide API for joint properties.
- A new scene-wide actuator manager.
- Uniform solver enforcement of joint velocity limits.
- Renaming armature or backend-specific friction fields in this change.
- Changing explicit actuator equations or native Newton controller semantics.
- Moving every joint-property override into a new articulation configuration
  hierarchy.

## Success criteria

- Ordinary actuators retain no joint-only runtime properties.
- Aggregated executors contain no unused joint-property tensors.
- Explicit actuator-model values never alias live articulation joint buffers.
- Implicit actuator drive values resolve through the correct live articulation
  binding without a persistent actuator-owned mirror.
- The collection exposes no `*_sim` API.
- Existing configurations using `*_sim` continue to work with actionable
  deprecation warnings.
- Existing runtime access to joint-only actuator attributes continues through
  deprecated, non-owning projections.
- Public documentation makes the ownership boundary and migration unambiguous.
- Focused tests, backend integration tests, documentation generation, and all
  pre-commit hooks pass before the implementation is pushed.
