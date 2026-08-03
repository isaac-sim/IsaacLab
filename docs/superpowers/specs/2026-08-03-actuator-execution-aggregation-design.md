<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Stateless Actuator Execution Aggregation

## Context

An articulation currently creates one actuator instance for every named entry in
`ArticulationCfg.actuators`. `ActuatorCollection.compute()` then calls every
instance separately and scatters every result separately. Two groups using the
same stateless actuator implementation therefore pay separate dispatch costs,
even when their math differs only through per-joint parameter tensors.

Named actuator groups remain useful. They give users understandable
configuration and access boundaries such as `hips`, `knees`, and `feet`. Those
logical boundaries do not need to define the runtime execution boundaries.

The Newton actuator path already separates the two concepts: Isaac Lab keeps its
logical actuator groups while a private Newton adapter constructs and steps
execution actuators. The standard Isaac Lab path should adopt the same ownership
model without changing the existing actuator compute implementations.

## Goals

- Preserve named actuator groups as the public configuration and access API.
- Preserve the concrete type, group-sized tensors, configuration, joint names,
  and articulation joint indices returned by each mapping entry.
- Aggregate all mutually disjoint groups of the same supported exact concrete
  actuator class, regardless of differences in their per-joint numeric
  parameters.
- Execute one existing `compute()` call and one pair of output scatter launches
  per aggregate instead of per logical group.
- Preserve processed commands and actuator telemetry exactly for the initial
  stateless models.
- Keep unsupported, stateful, neural, and custom actuator models on the existing
  per-group execution path.
- Leave Newton-native actuator execution unchanged.

## Non-goals

- Do not aggregate neural-network actuator models.
- Do not aggregate delayed or remotized actuator models in the initial change.
- Do not add new fused Warp kernels or rewrite actuator mathematics.
- Do not aggregate subclasses merely because a supported class appears in their
  inheritance chain.
- Do not change the solver-side grouping of joint property writes.
- Do not make direct calls to `compute()` or `reset()` on collection-managed
  group entries part of the supported lifecycle API.

## Terminology

### Logical actuator group

A named, user-facing entry in `ArticulationCfg.actuators` and
`Articulation.actuators`. It owns group-specific configuration and metadata and
continues to appear as its configured concrete actuator type.

### Execution batch

A private `ActuatorCollection` entry that owns the live tensors and performs
actuator computation for one or more compatible logical groups. An execution
batch has one concrete actuator instance and one combined array of articulation
joint indices.

## Eligibility

Aggregation is an explicit capability of an exact concrete actuator class. The
initial supported classes are:

- `ImplicitActuator`
- `IdealPDActuator`
- `DCMotor`

All groups whose exact `type()` is the same supported class belong to one
execution batch for the articulation. Numeric parameters do not participate in
the compatibility decision. Stiffness, damping, effort limits, velocity limits,
solver limits, armature, friction values, and DC-motor saturation effort may all
differ across groups and joints.

Every joint in an aggregate must be controlled by exactly one logical group in
the complete collection. The current collection permits overlapping groups and
applies them sequentially, so their config order determines which processed
command is scattered last. Any group touching a multiply-controlled joint
retains per-group execution, including when the other controlling group uses a
different actuator class. This preserves ordered overwrite behavior and avoids
duplicate writes within one scatter launch. The structural safety condition is
independent of actuator parameter compatibility.

Eligibility must not be inherited accidentally. In particular,
`ActuatorNetMLP`, `ActuatorNetLSTM`, `DelayedPDActuator`, and
`RemotizedPDActuator` must not enter a `DCMotor` or `IdealPDActuator` batch.
Unsupported classes retain one execution entry per logical group.

The capability is exposed through a protected class-level aggregation hook. The
base implementation declines aggregation. Each supported exact class opts in and
builds an executor from already-resolved group instances. This keeps the feature
extensible without promising a new public custom-actuator protocol in the first
release.

## Collection architecture

`ActuatorCollection` continues to own its mapping of logical group names to
concrete `ActuatorBase` instances. It additionally owns a private ordered list of
nested `_ExecutionBatch` records. Each record contains:

- The concrete actuator used for computation.
- The logical group names represented by the executor.
- Their contiguous batch-local column slices.
- The combined articulation joint-index array used to gather and scatter.
- The mapping required to route runtime parameter writes from articulation joint
  indices to batch-local columns.

The execution list follows the first appearance of each logical group in config
order. Groups combined into a previously created aggregate do not create an
additional entry. This makes execution deterministic without making dictionary
ordering part of numerical semantics.

## Construction and tensor ownership

The collection first builds all logical group instances exactly as it does
today. This preserves config-to-USD resolution, resolution diagnostics, backend
property writes, and all existing validation.

After joint coverage is validated, the collection partitions eligible groups by
exact concrete class. A supported class builds an aggregate only when at least
two groups are present; a single group remains its own execution instance.

The aggregate factory consumes resolved group tensors rather than resolving a
synthetic merged config again. It performs two phases:

1. Build and validate the complete private executor and all proposed group
   bindings without mutating public groups.
2. Commit the bindings only after every tensor has the expected device, dtype,
   shape, and contiguous batch-local slice.

This two-phase operation prevents a failed aggregation from leaving some public
groups rebound and others independent.

Groups are packed contiguously in config order. For example, a two-joint `hips`
group followed by a two-joint `knees` group occupies batch-local slices `0:2`
and `2:4`. Contiguous slices are required so group tensors remain writable Torch
views rather than advanced-indexing copies.

The aggregate owns one concatenated allocation for every persistent common
group-sized parameter tensor used by the supported models, including:

- model and solver effort limits;
- model and solver velocity limits;
- stiffness and damping;
- armature;
- static, dynamic, and viscous friction.

Each public group attribute is rebound to its slice of the corresponding
aggregate tensor. The public tensor shape remains `(num_envs, group_num_joints)`.
In-place mutation therefore affects the storage used by aggregate execution.
Runtime changes that must also synchronize collection-wide snapshots or backend
controllers continue to use the collection's explicit write methods.

Computed and applied effort require different ownership treatment. The existing
stateless `compute()` implementations replace `self.computed_effort` and
`self.applied_effort` with newly produced tensors on every call. Changing those
implementations to write in place would alter the hot path, so the aggregate does
not require persistent output allocations. After every aggregate compute, the
collection rebinds each public group's computed- and applied-effort attributes to
the appropriate contiguous slices of the executor's new output tensors. This
matches the existing behavior in which retaining an old effort-tensor reference
across compute calls does not guarantee that it remains current.

For `DCMotor`, the aggregate expands each group's scalar saturation effort into
batch-local per-joint values. It rebuilds the derived velocity-at-effort-limit
tensor and allocates batch-sized joint-velocity and zero-effort working buffers.
The existing element-wise `DCMotor.compute()` and clipping implementation then
broadcast and execute without modification.

The logical groups retain their original `cfg`, joint names, articulation joint
indices, and property-resolution tables. The private executor's construction
metadata is not exposed as user-facing configuration.

Replacing an entire aliased tensor attribute would detach that logical group
from the executor. Collection-managed actuator tensors therefore support
in-place mutation, while whole-attribute replacement is unsupported. The
documentation must direct runtime gain changes through the collection write
methods because those methods also update collection telemetry and native
controllers.

## Execution

On every standard Isaac Lab actuator step, the collection iterates over execution
batches rather than logical groups. For each entry it:

1. Gathers position, velocity, and effort commands for the entry's combined
   articulation joint indices.
2. Gathers current joint position and velocity with the same indices.
3. Calls the existing concrete actuator's `compute()` once.
4. Refreshes the logical groups' computed- and applied-effort views.
5. Scatters processed joint commands once.
6. Scatters computed effort, applied effort, gear-ratio telemetry when present,
   and soft velocity limits once.

The supported models use independent element-wise tensor operations with no
cross-joint reductions. Packing groups as columns therefore does not alter their
mathematical behavior. Output scattering restores articulation joint order, so
logical groups may cover interleaved articulation joints while remaining
contiguous in batch-local storage.

An unbatched entry follows the same execution-record path with its existing
actuator instance and joint indices. This avoids maintaining separate compute
loops for aggregated and non-aggregated models.

The initial aggregate classes are stateless. Their `reset()` methods are no-ops,
so collection reset behavior is unchanged. Unsupported stateful groups continue
to receive their existing per-group reset calls.

## Runtime writes

The collection builds an articulation-joint-to-execution-column routing map when
it creates the execution plan. Runtime stiffness and damping writes use this map
to update the relevant executor tensors for both aggregated and unbatched
entries. Because public group tensors alias those tensors, group reads reflect
the update immediately.

The same write continues to update the collection's articulation-ordered
resolved-gain buffer and calls the backend's native-controller write hook. This
keeps standard Python execution, public telemetry, and Newton-native execution
consistent.

Direct mutation of a group tensor changes standard aggregate execution through
the alias, but it does not perform backend synchronization. Documentation must
therefore continue to recommend the collection write methods whenever a value is
also consumed by a physics backend or native actuator controller.

## Newton-native behavior

When the backend reports that native actuator handling is active, the collection
does not construct Lab execution aggregates. Newton already owns separate
execution actuators, state buffers, reset behavior, and telemetry synchronization.
The Lab logical groups remain available to Newton's binding layer exactly as they
are today.

On PhysX with Newton actuators enabled, the existing USD parser continues to
combine identical resolved Newton signatures. This proposal does not change that
more conservative Newton grouping rule. Standard Lab aggregation and Newton
aggregation are independent implementations of the same logical-group versus
execution-batch distinction.

## Errors and fallback behavior

Unsupported classes take the existing per-group path without a warning. A
supported built-in class is expected to aggregate successfully. Failure to build
or validate its executor raises an exception that names the actuator class and
logical groups involved. It must not silently fall back because that would hide
an implementation defect and make performance depend on an unnoticed error.

Validation covers:

- all groups use the same device and number of environments;
- every packed tensor has the expected dtype and two-dimensional shape;
- batch-local slices are contiguous, non-overlapping, and cover the executor;
- combined articulation indices match the packed joint names and group slices;
- the public group binding shapes remain unchanged.

Existing articulation coverage validation remains unchanged. The execution-plan
builder counts joint use across every logical group and leaves any group touching
a multiply-controlled joint unbatched.

## Public API and documentation

The mapping contract remains source compatible:

```python
hips = robot.actuators["hips"]
assert isinstance(hips, DCMotor)
assert hips.stiffness.shape == (robot.num_instances, len(hips.joint_names))
```

The core actuator documentation must distinguish logical groups from private
execution batches. It must state that the articulation and collection own the
managed actuator lifecycle; users set commands and inspect or configure groups
but do not call a group's `compute()` or `reset()` directly.

The documentation must also explain that aggregation is an implementation detail:
users must not rely on how many execution batches are created. Adding support for
another actuator class must not require a user configuration change.

## Verification

Focused tests use groups with deliberately different values so they prove the
aggressive compatibility rule rather than merely exercising identical configs.
They cover:

1. Multiple `IdealPDActuator` groups with different gains and limits produce
   exactly the same processed commands and telemetry as independent execution.
2. Multiple `DCMotor` groups with different gains, velocity limits, effort
   limits, and saturation efforts produce exactly the same results.
3. Multiple `ImplicitActuator` groups preserve processed commands and approximate
   torque telemetry exactly.
4. In-place mutation of one logical group's tensor changes only its aggregate
   columns.
5. Logical group effort views are rebound to the aggregate's current outputs
   after every compute call.
6. Logical mapping values retain their concrete type, original metadata, and
   group-sized tensor shapes.
7. Mixed eligible, delayed, neural, and custom groups produce the expected
   execution plan.
8. Subclasses of supported models do not inherit aggregation eligibility.
9. Overlapping same- or different-class groups retain deterministic per-group
   execution.
10. Runtime gain writes update executor storage, logical group views, collection
   resolved-gain buffers, and the native write hook.
11. Native handling bypasses Lab aggregate construction and execution.
12. A compute-call spy observes one invocation per aggregated exact class and one
    invocation for each unsupported logical group.

All numerical equivalence assertions for the three stateless models use zero
absolute and relative tolerance. The tests also compare processed position,
velocity, and effort commands, not only actuator torque telemetry.

Before committing implementation, the focused tests and `./isaaclab.sh -f` must
pass. Direct dispatch instrumentation must confirm that multiple same-class
groups reduce to one compute dispatch and one pair of scatter launches per class.
This assertion belongs in the focused collection test rather than a permanent
performance benchmark.

## Success criteria

The change is complete when:

- public logical group access remains source compatible;
- every disjoint supported exact class executes at most once per articulation
  step;
- differing per-joint parameters remain exact and independently writable;
- supported batched outputs are exactly equal to independent execution;
- unsupported and Newton-native paths retain their existing behavior;
- documentation describes the logical versus execution grouping model; and
- focused tests and repository pre-commit checks pass.
