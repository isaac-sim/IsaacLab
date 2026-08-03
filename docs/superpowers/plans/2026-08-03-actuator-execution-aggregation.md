<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Stateless Actuator Execution Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute all disjoint logical groups of each supported stateless actuator class through one private actuator instance while preserving concrete, group-shaped public accessors.

**Architecture:** `ActuatorCollection` keeps its named concrete group objects and builds a private ordered list of execution batches. `ImplicitActuator`, `IdealPDActuator`, and `DCMotor` opt in through an exact-class protected hook that packs already-resolved per-joint tensors; the collection gathers, computes, refreshes group telemetry views, and scatters once per batch. Stateful, neural, custom, overlapping, and Newton-native paths retain their existing execution behavior.

**Tech Stack:** Python 3.11+, PyTorch, Warp, pytest, Isaac Lab `configclass`, pre-commit through `./isaaclab.sh`.

## Global Constraints

- Do not add required or optional dependencies.
- Aggregate by exact concrete class, never by `isinstance()` or inherited eligibility.
- Permit every per-joint numeric parameter to differ between aggregated groups.
- Keep `ImplicitActuator`, `IdealPDActuator`, and `DCMotor` as the only initial opt-ins.
- Keep delayed, remotized, neural-network, custom, and overlapping groups on per-group execution.
- Preserve `robot.actuators[name]` concrete type, config, joint names, articulation indices, and group-shaped tensors.
- Do not change existing actuator arithmetic or add fused Warp kernels.
- Do not construct Lab aggregates while backend-native actuator handling is active.
- Use exact numerical assertions with `rtol=0.0` and `atol=0.0` for stateless parity.
- Run regression tests before implementation and observe the expected failure.
- Run `./isaaclab.sh -f` before every commit; if it modifies files, review and stage them, then rerun it.
- Use conventional commit subjects in imperative mood without AI attribution.
- New files use the 2026 Isaac Lab SPDX copyright header.

---

### Task 1: Build stateless execution batches without changing group access

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_pd.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Test: `source/isaaclab/test/actuators/test_actuator_collection.py`

**Interfaces:**
- Consumes: Existing resolved `ActuatorBase` tensors and `ActuatorCollection._groups_by_class`.
- Produces: `ActuatorBase._supports_execution_aggregation: ClassVar[bool]`.
- Produces: `ActuatorBase._build_execution_actuator(cls, actuators: Sequence[ActuatorBase]) -> ActuatorBase`.
- Produces: Nested `ActuatorCollection._ExecutionBatch` with `actuator`, `group_names`, `group_slices`, `joint_indices`, and `joint_indices_wp`.
- Produces: `ActuatorCollection._execution_batches: list[ActuatorCollection._ExecutionBatch]`.
- Produces: `ActuatorCollection._make_execution_batch(group_names: tuple[str, ...], groups: tuple[ActuatorBase, ...], joint_indices: torch.Tensor, *, executor: ActuatorBase | None = None) -> ActuatorCollection._ExecutionBatch`.

- [ ] **Step 1: Make the fake backend resolve each group's joint expressions**

Replace the all-joints-only implementation in `FakeActuatorControl.find_joints()` with expression matching so tests can create disjoint and overlapping groups:

```python
import re

def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
    expressions = [name_keys] if isinstance(name_keys, str) else list(name_keys)
    matches = [
        (joint_id, joint_name)
        for joint_id, joint_name in enumerate(self._joint_names)
        if any(re.fullmatch(expression, joint_name) for expression in expressions)
    ]
    return [joint_id for joint_id, _ in matches], [joint_name for _, joint_name in matches]
```

- [ ] **Step 2: Write failing construction and eligibility tests**

Import the concrete actuator classes/configs and add helpers that deliberately use different parameters:

```python
from isaaclab.actuators import (
    DCMotor,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
)

def _ideal_cfg(joints: list[str], *, stiffness: float, damping: float, effort_limit: float):
    return IdealPDActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=100.0,
    )

def _dc_cfg(
    joints: list[str],
    *,
    stiffness: float,
    damping: float,
    effort_limit: float,
    velocity_limit: float,
    saturation_effort: float,
):
    return DCMotorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
    )
```

Add these behavioral tests:

```python
def test_same_stateless_class_builds_one_execution_batch_with_group_views():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=10.0, damping=1.0, effort_limit=20.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=30.0, damping=2.0, effort_limit=40.0),
        },
        control,
    )

    assert len(collection._execution_batches) == 1
    batch = collection._execution_batches[0]
    assert type(batch.actuator) is IdealPDActuator
    assert batch.group_names == ("hips", "knees")
    assert isinstance(collection["hips"], IdealPDActuator)
    assert collection["hips"].joint_names == ["joint_0", "joint_2"]
    assert collection["hips"].stiffness.shape == (2, 2)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 10.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))

    collection["hips"].stiffness.fill_(17.0)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 17.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))


def test_dc_motor_execution_batch_packs_different_saturation_efforts():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )

    batch = collection._execution_batches[0]
    assert type(batch.actuator) is DCMotor
    torch.testing.assert_close(
        batch.actuator._saturation_effort,
        torch.tensor([[60.0, 60.0, 120.0, 120.0]]).expand(2, -1),
    )


def test_stateful_subclasses_and_overlapping_groups_remain_unbatched():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    delayed = ActuatorCollection(
        {
            "first": DelayedPDActuatorCfg(
                joint_names_expr=["joint_0", "joint_1"], stiffness=1.0, damping=1.0, max_delay=0
            ),
            "second": DelayedPDActuatorCfg(
                joint_names_expr=["joint_2", "joint_3"], stiffness=2.0, damping=2.0, max_delay=0
            ),
        },
        control,
    )
    assert len(delayed._execution_batches) == 2

    overlapping = ActuatorCollection(
        {
            "first": _ideal_cfg(["joint_0", "joint_1"], stiffness=1.0, damping=1.0, effort_limit=10.0),
            "second": _ideal_cfg(["joint_1", "joint_2"], stiffness=2.0, damping=2.0, effort_limit=20.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2"]),
    )
    assert len(overlapping._execution_batches) == 2

    cross_class = ActuatorCollection(
        {
            "ideal_a": _ideal_cfg(["joint_0"], stiffness=1.0, damping=1.0, effort_limit=10.0),
            "dc": _dc_cfg(
                ["joint_1", "joint_2"],
                stiffness=2.0,
                damping=2.0,
                effort_limit=20.0,
                velocity_limit=10.0,
                saturation_effort=30.0,
            ),
            "ideal_b": _ideal_cfg(["joint_1"], stiffness=3.0, damping=3.0, effort_limit=30.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2"]),
    )
    assert len(cross_class._execution_batches) == 3
```

- [ ] **Step 3: Run the construction tests and verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_same_stateless_class_builds_one_execution_batch_with_group_views \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_dc_motor_execution_batch_packs_different_saturation_efforts \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_stateful_subclasses_and_overlapping_groups_remain_unbatched -v
```

Expected: FAIL because `ActuatorCollection` has no `_execution_batches` and groups still own independent tensors.

- [ ] **Step 4: Add the protected stateless aggregation factory**

In `actuator_base.py`, import `copy`, define the persistent parameter names, default capability, and common factory:

```python
_EXECUTION_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
    "effort_limit",
    "effort_limit_sim",
    "velocity_limit",
    "velocity_limit_sim",
    "stiffness",
    "damping",
    "armature",
    "friction",
    "dynamic_friction",
    "viscous_friction",
)

_supports_execution_aggregation: ClassVar[bool] = False

@classmethod
def _build_execution_actuator(cls, actuators: Sequence[ActuatorBase]) -> ActuatorBase:
    """Build one private executor from resolved logical actuator groups."""
    executor = copy.copy(actuators[0])
    executor._joint_names = [name for actuator in actuators for name in actuator.joint_names]
    for name in cls._EXECUTION_PARAMETER_NAMES:
        setattr(executor, name, torch.cat([getattr(actuator, name) for actuator in actuators], dim=1))
    executor.computed_effort = torch.zeros(
        executor._num_envs, len(executor._joint_names), device=executor._device
    )
    executor.applied_effort = torch.zeros_like(executor.computed_effort)
    return executor
```

Make `_EXECUTION_PARAMETER_NAMES` a member of `ActuatorBase`, keep the hook protected, and use modern annotations. Do not add the capability to subclasses through `isinstance()`.

In `actuator_pd.py`, opt in on all three exact classes by defining `_supports_execution_aggregation = True` in each class body. Override the factory on `DCMotor`:

```python
@classmethod
def _build_execution_actuator(cls, actuators: Sequence[ActuatorBase]) -> ActuatorBase:
    executor = super()._build_execution_actuator(actuators)
    executor._saturation_effort = torch.cat(
        [
            torch.full_like(actuator.effort_limit, float(actuator._saturation_effort))
            for actuator in actuators
        ],
        dim=1,
    )
    executor._vel_at_effort_lim = executor.velocity_limit * (
        1 + executor.effort_limit / executor._saturation_effort
    )
    executor._joint_vel = torch.zeros_like(executor.computed_effort)
    executor._zeros_effort = torch.zeros_like(executor.computed_effort)
    return executor
```

- [ ] **Step 5: Add normalized execution-batch records**

In `actuator_collection.py`, add a nested dataclass:

```python
@dataclass
class _ExecutionBatch:
    actuator: ActuatorBase
    group_names: tuple[str, ...]
    group_slices: tuple[slice, ...]
    joint_indices: torch.Tensor
    joint_indices_wp: wp.array
```

Implement `_joint_indices_as_torch()` so `slice(None)` becomes
`torch.arange(self.num_joints, dtype=torch.int32, device=self.device)`.
Implement `_make_execution_batch()` so it computes group slices in group order,
normalizes the executor's joint metadata, and creates one contiguous Warp index
array. For an unbatched group, pass that group as the executor.

- [ ] **Step 6: Partition groups into deterministic execution batches**

After `_validate_coverage()`, call `_build_execution_batches()`. Implement its
partitioning rules as follows:

```python
def _build_execution_batches(self) -> None:
    native_active = getattr(self._control, "native_active", False)
    batch_by_group: dict[str, ActuatorCollection._ExecutionBatch] = {}
    if not self._groups:
        self._execution_batches = []
        return
    group_joint_indices = {
        name: self._joint_indices_as_torch(group) for name, group in self._groups.items()
    }
    joint_use_count = torch.bincount(
        torch.cat(list(group_joint_indices.values())).to(dtype=torch.long),
        minlength=self.num_joints,
    )

    for actuator_type in self._groups_by_class:
        names = tuple(name for name, group in self._groups.items() if type(group) is actuator_type)
        groups = [self._groups[name] for name in names]
        joint_indices = [group_joint_indices[name] for name in names]
        supported = actuator_type.__dict__.get("_supports_execution_aggregation", False)

        if native_active or not supported:
            for name, group, indices in zip(names, groups, joint_indices):
                batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
            continue

        safe = [
            (name, group, indices)
            for name, group, indices in zip(names, groups, joint_indices)
            if torch.all(joint_use_count[indices.to(dtype=torch.long)] == 1)
        ]
        safe_names_set = {name for name, _, _ in safe}
        unsafe = [
            (name, group, indices)
            for name, group, indices in zip(names, groups, joint_indices)
            if name not in safe_names_set
        ]
        for name, group, indices in unsafe:
            batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
        if len(safe) < 2:
            for name, group, indices in safe:
                batch_by_group[name] = self._make_execution_batch((name,), (group,), indices)
            continue

        safe_names, safe_groups, safe_indices = zip(*safe)
        combined = torch.cat(safe_indices)
        executor = actuator_type._build_execution_actuator(safe_groups)
        executor._joint_indices = combined
        batch = self._make_execution_batch(safe_names, safe_groups, combined, executor=executor)
        self._validate_execution_batch(batch, safe_groups)
        self._bind_execution_batch_parameters(batch, safe_groups)
        for name in safe_names:
            batch_by_group[name] = batch

    seen: set[int] = set()
    self._execution_batches = []
    for name in self._groups:
        batch = batch_by_group[name]
        if id(batch) not in seen:
            self._execution_batches.append(batch)
            seen.add(id(batch))
```

- [ ] **Step 7: Validate and atomically bind aggregate parameter storage**

Implement `_validate_execution_batch()` to check device, dtype, shapes, slice
coverage, and combined name/index counts without modifying public groups. Build
the complete list of proposed `(group, attribute, view)` bindings next. Only
after every proposed view passes validation, have
`_bind_execution_batch_parameters()` reassign each logical group's persistent
parameter, initial `computed_effort`, and initial `applied_effort` tensors to its
executor slice.

- [ ] **Step 8: Run construction tests and the existing collection file**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection.py -v
```

Expected: PASS, including the three new construction tests.

- [ ] **Step 9: Run pre-commit twice around staging**

Run:

```bash
./isaaclab.sh -f
git diff --check
git add source/isaaclab/isaaclab/actuators/actuator_base.py \
  source/isaaclab/isaaclab/actuators/actuator_pd.py \
  source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection.py
./isaaclab.sh -f
```

Expected: both pre-commit runs PASS; inspect and stage any formatter changes before the second run.

- [ ] **Step 10: Commit execution-batch construction**

```bash
git commit -m "Build stateless actuator execution batches"
```

---

### Task 2: Execute aggregate actuators with exact output parity

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Test: `source/isaaclab/test/actuators/test_actuator_collection.py`

**Interfaces:**
- Consumes: `ActuatorCollection._ExecutionBatch` and `_execution_batches` from Task 1.
- Produces: `ActuatorCollection._bind_execution_batch_outputs(batch: _ExecutionBatch) -> None`.
- Updates: `ActuatorCollection._scatter_actuator_output(actuator, control_action, joint_indices: wp.array | None = None) -> None` so execution batches reuse cached Warp indices.
- Produces: One `compute()` and one `_scatter_actuator_output()` call per execution batch.

- [ ] **Step 1: Add a helper that constructs an unbatched reference collection**

Use `monkeypatch.context()` to disable the exact class capability only while constructing the reference:

```python
def _make_unbatched_reference(monkeypatch, actuator_type, cfgs, control):
    with monkeypatch.context() as patch:
        patch.setattr(actuator_type, "_supports_execution_aggregation", False)
        return ActuatorCollection(cfgs, control)
```

Add a helper that assigns deterministic joint state and all three command buffers to a collection/control pair. Use nonzero, nonsymmetric values so clipping and joint reordering are exercised.

- [ ] **Step 2: Write failing exact-parity tests for all three supported classes**

For each model, construct an unbatched reference and aggregated collection with identical four-joint configs, state, and commands. Use different parameters in the two groups. After `compute()`, assert exact equality for:

```python
torch.testing.assert_close(
    actual.joint_command.position.torch,
    reference.joint_command.position.torch,
    rtol=0.0,
    atol=0.0,
)
torch.testing.assert_close(actual.joint_command.velocity.torch, reference.joint_command.velocity.torch, rtol=0.0, atol=0.0)
torch.testing.assert_close(actual.joint_command.effort.torch, reference.joint_command.effort.torch, rtol=0.0, atol=0.0)
torch.testing.assert_close(actual.computed_torque.torch, reference.computed_torque.torch, rtol=0.0, atol=0.0)
torch.testing.assert_close(actual.applied_torque.torch, reference.applied_torque.torch, rtol=0.0, atol=0.0)
torch.testing.assert_close(actual.soft_joint_vel_limits.torch, reference.soft_joint_vel_limits.torch, rtol=0.0, atol=0.0)
```

Name the tests `test_ideal_pd_aggregate_matches_independent_groups_exactly`,
`test_dc_motor_aggregate_matches_independent_groups_exactly`, and
`test_implicit_aggregate_matches_independent_groups_exactly`.

For the DC-motor inputs, include velocities on both sides of each group's corner velocity so the two different saturation efforts and limits are exercised.

- [ ] **Step 3: Add failing dispatch-count and output-view tests**

Wrap `IdealPDActuator.compute` and the collection instance's
`_scatter_actuator_output` with counters, run a collection containing three
disjoint Ideal-PD groups, and assert one call to each. One scatter-helper call
represents the existing processed-target and telemetry Warp launch pair. Then
verify that each logical group's `computed_effort` and `applied_effort` equals
the corresponding aggregate output slice after two consecutive compute calls
with different commands:

```python
assert compute_calls == 1
assert scatter_calls == 1
torch.testing.assert_close(
    collection["hips"].computed_effort,
    collection._execution_batches[0].actuator.computed_effort[:, :2],
    rtol=0.0,
    atol=0.0,
)
```

Store the first output object before the second call and assert the group attribute is rebound to the new executor output object rather than the stale tensor.

- [ ] **Step 4: Run the new compute tests and verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_ideal_pd_aggregate_matches_independent_groups_exactly \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_dc_motor_aggregate_matches_independent_groups_exactly \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_implicit_aggregate_matches_independent_groups_exactly \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_aggregate_computes_once_and_refreshes_group_outputs -v
```

Expected: FAIL because `compute()` still iterates over logical groups and does not refresh views from the private executor.

- [ ] **Step 5: Refresh logical group output views after aggregate compute**

Implement `_bind_execution_batch_outputs()` by rebinding every named logical
group's `computed_effort` and `applied_effort` to the executor's current tensor
slices. Do not change the actuator models to write outputs in place; their
existing assignment behavior is part of the hot path.

- [ ] **Step 6: Route collection computation through execution batches**

Replace the logical-group loop with:

```python
for batch in self._execution_batches:
    actuator = batch.actuator
    joint_indices = batch.joint_indices
    control_action = ArticulationActions(
        joint_positions=self.command.position.torch[:, joint_indices],
        joint_velocities=self.command.velocity.torch[:, joint_indices],
        joint_efforts=self.command.effort.torch[:, joint_indices],
        joint_indices=joint_indices,
    )
    control_action = actuator.compute(
        control_action,
        joint_pos=self._control.joint_pos.torch[:, joint_indices],
        joint_vel=self._control.joint_vel.torch[:, joint_indices],
    )
    self._bind_execution_batch_outputs(batch)
    self._scatter_actuator_output(actuator, control_action, batch.joint_indices_wp)
```

Add the optional `joint_indices` parameter to `_scatter_actuator_output()`. Use
the supplied cached Warp array for execution batches and retain
`_joint_indices_as_wp(actuator)` as the fallback when it is omitted. The kernel
inputs and outputs remain unchanged.

- [ ] **Step 7: Run exact-parity and full actuator unit tests**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators -v
```

Expected: PASS with zero-tolerance parity for all three stateless models.

- [ ] **Step 8: Run pre-commit twice around staging**

Run:

```bash
./isaaclab.sh -f
git diff --check
git add source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection.py
./isaaclab.sh -f
```

Expected: both runs PASS; review and stage formatter changes before rerunning.

- [ ] **Step 9: Commit aggregate execution**

```bash
git commit -m "Execute stateless actuator groups together"
```

---

### Task 3: Synchronize runtime gains and bypass aggregation for native execution

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Test: `source/isaaclab/test/actuators/test_actuator_collection.py`

**Interfaces:**
- Consumes: Execution-batch joint-index routing from Task 1.
- Produces: `ActuatorCollection._write_execution_parameter(attr: str, values: torch.Tensor, env_ids: torch.Tensor, joint_ids: torch.Tensor) -> None`.
- Preserves: `ActuatorControl.write_native_actuator_gain(attr, values, env_ids, joint_ids)` with native names `kp` and `kd`.

- [ ] **Step 1: Extend the fake backend to record native calls and expose native activation**

Add `native_gain_writes` to `FakeActuatorControl` and implement:

```python
def write_native_actuator_gain(self, attr, values, env_ids, joint_ids) -> None:
    self.native_gain_writes.append((attr, values.clone(), env_ids.clone(), joint_ids.clone()))
```

Add a `NativeFakeActuatorControl` subclass:

```python
class NativeFakeActuatorControl(FakeActuatorControl):
    @property
    def native_active(self) -> bool:
        return True

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return True
```

- [ ] **Step 2: Write failing gain-routing tests**

Build a two-group DC-motor aggregate, write stiffness to environment `1` and articulation joints `[0, 3]`, then assert all four destinations:

```python
values = torch.tensor([[71.0, 93.0]])
env_ids = torch.tensor([1], dtype=torch.long)
joint_ids = torch.tensor([0, 3], dtype=torch.long)
collection.write_actuator_stiffness_to_sim(stiffness=values, env_ids=env_ids, joint_ids=joint_ids)

assert collection["hips"].stiffness[1, 0] == 71.0
assert collection["knees"].stiffness[1, 1] == 93.0
assert collection.actuator_stiffness.torch[1, 0] == 71.0
assert collection.actuator_stiffness.torch[1, 3] == 93.0
assert control.native_gain_writes[-1][0] == "kp"
```

Repeat for damping with reversed joint request order `[3, 0]` to prove values are routed by articulation ID rather than batch position.

- [ ] **Step 3: Write a failing native-bypass test**

Construct two compatible DC-motor groups with `NativeFakeActuatorControl`. Assert there are two unaggregated execution records, replace `DCMotor.compute` with a function that raises, call `collection.compute()`, and verify no exception occurs because native handling returned `True` before the Lab loop.

Also perform a stiffness write and assert each public logical group's tensor changes, proving native bypass does not leave group accessors stale.

- [ ] **Step 4: Run the synchronization tests and verify they fail**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_runtime_gains_route_into_aggregate_and_native_hook \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_native_execution_bypasses_lab_aggregation_and_keeps_group_gains_current -v
```

Expected: FAIL because current writes update only the collection proxy/native hook and do not update logical or aggregate actuator tensors.

- [ ] **Step 5: Implement articulation-ID-to-executor gain routing**

Keep every execution record's `joint_indices` as a normalized int32 Torch tensor, including unbatched `slice(None)` groups. Add:

```python
def _write_execution_parameter(
    self,
    attr: str,
    values: torch.Tensor,
    env_ids: torch.Tensor,
    joint_ids: torch.Tensor,
) -> None:
    values = values.to(self.device, dtype=torch.float32)
    env_ids = env_ids.to(self.device, dtype=torch.long)
    joint_ids = joint_ids.to(self.device, dtype=torch.long)
    for batch in self._execution_batches:
        batch_joint_ids = batch.joint_indices.to(dtype=torch.long)
        requested_columns, batch_columns = torch.where(joint_ids[:, None] == batch_joint_ids[None, :])
        if requested_columns.numel() == 0:
            continue
        target = getattr(batch.actuator, attr)
        target[env_ids[:, None], batch_columns[None, :]] = values[:, requested_columns]
```

In `_write_actuator_gain()`, map native names to actuator attributes:

```python
actuator_attr = {"kp": "stiffness", "kd": "damping"}[attr]
self._write_execution_parameter(actuator_attr, values, env_ids, joint_ids)
```

Then retain the existing Warp write into the articulation-ordered collection buffer and the existing native-controller hook call. Because aggregated group tensors alias executor parameters, no additional group copy is needed. Native mode uses unaggregated execution records, so the same routing updates each logical actuator directly.

- [ ] **Step 6: Run synchronization, collection, and backend adapter tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py::TestRandomizeActuatorGainsViaEventsPhysx \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py::TestRandomizeActuatorGainsViaEventsNewton -v
```

Expected: PASS.

- [ ] **Step 7: Run pre-commit twice around staging**

Run:

```bash
./isaaclab.sh -f
git diff --check
git add source/isaaclab/isaaclab/actuators/actuator_collection.py \
  source/isaaclab/test/actuators/test_actuator_collection.py
./isaaclab.sh -f
```

Expected: both runs PASS; inspect and stage any formatter changes before rerunning.

- [ ] **Step 8: Commit runtime synchronization**

```bash
git commit -m "Synchronize aggregated actuator gains"
```

---

### Task 4: Document logical groups and execution batches

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `docs/source/overview/core-concepts/actuators.rst`
- Modify: `source/isaaclab/changelog.d/actuator-collection.minor.rst`

**Interfaces:**
- Consumes: Final execution behavior from Tasks 1-3.
- Produces: User-facing distinction between logical actuator groups and private execution batches.
- Produces: Lifecycle guidance that collection-managed groups are configured/inspected by users but computed/reset by the collection.

- [ ] **Step 1: Update the collection class docstring**

Document that named mapping entries are logical configuration/access groups, that compatible stateless groups may share a private executor, and that users must not depend on execution-batch count. State that direct `compute()` and `reset()` calls on collection-managed group values are unsupported because the collection owns lifecycle execution.

- [ ] **Step 2: Update the actuator concepts documentation**

In the runtime API section of `docs/source/overview/core-concepts/actuators.rst`, add a subsection titled `Logical groups and execution batches`. Include this concrete example:

```rst
Logical groups and execution batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Named entries such as ``hips`` and ``knees`` remain separate configuration and
access groups. Isaac Lab may execute disjoint groups of the same supported
stateless actuator class through one private actuator instance. Per-joint gains
and limits may differ; aggregation does not merge their public configuration or
change the shapes returned by ``robot.actuators["hips"]``.

Execution batching is an implementation detail. Do not call
:meth:`~isaaclab.actuators.ActuatorBase.compute` or
:meth:`~isaaclab.actuators.ActuatorBase.reset` directly on an actuator obtained
from the collection, and do not rely on the number of execution batches. Set
commands and perform lifecycle operations through the articulation and its
:class:`~isaaclab.actuators.ActuatorCollection`.
```

Keep the existing Newton-native section and explain that Newton owns a separate execution aggregation path.

- [ ] **Step 3: Update the existing changelog fragment**

Add under `Added`:

```rst
* Added execution aggregation for disjoint stateless actuator groups while
  preserving named group configuration and access.
```

- [ ] **Step 4: Run focused tests and documentation generation**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators -v
./isaaclab.sh -d
```

Expected: actuator tests PASS and documentation generation completes without new warnings or broken references.

- [ ] **Step 5: Confirm dispatch reduction with the permanent spy test**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py::test_aggregate_computes_once_and_refreshes_group_outputs -vv
```

Expected: PASS, proving three same-class logical groups invoke the model once. Record the before/after dispatch counts in the PR description as `3 -> 1` compute calls and `3 -> 1` pairs of scatter launches for that fixture.

- [ ] **Step 6: Run repository pre-commit before staging**

Run:

```bash
./isaaclab.sh -f
git diff --check
git status --short
```

Expected: PASS and only files belonging to this feature are modified.

- [ ] **Step 7: Stage documentation and any formatter output, then rerun pre-commit**

Run:

```bash
git add source/isaaclab/isaaclab/actuators/actuator_collection.py \
  docs/source/overview/core-concepts/actuators.rst \
  source/isaaclab/changelog.d/actuator-collection.minor.rst
./isaaclab.sh -f
```

Expected: PASS. Review and stage any generated or formatted file that belongs to this task before rerunning.

- [ ] **Step 8: Commit documentation**

```bash
git commit -m "Document actuator execution batching"
```

---

### Task 5: Perform final regression and branch verification

**Files:**
- Verify only; no planned source changes.

**Interfaces:**
- Consumes: All implementation, tests, documentation, and changelog changes from Tasks 1-4.
- Produces: Final evidence that the branch is ready for review and push.

- [ ] **Step 1: Run the complete focused actuator and collection suites**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators -v
```

Expected: PASS.

- [ ] **Step 2: Run the affected backend actuator integration tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py::TestIdealPDEquivalence \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py::TestDCMotorEquivalence \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py::TestMixedActuatorEquivalence \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py::TestMixedWithImplicitEquivalence \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py::TestRandomizeActuatorGainsViaEventsPhysx \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py::TestIdealPDEquivalence \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py::TestDCMotorEquivalence \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py::TestMixedActuatorEquivalence \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py::TestMixedWithImplicitEquivalence \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py::TestRandomizeActuatorGainsViaEventsNewton -v
```

Expected: PASS for every selected class. Do not run training tasks as part of this unit-level refactor.

- [ ] **Step 3: Run final repository pre-commit**

Run:

```bash
./isaaclab.sh -f
```

Expected: every hook PASS and no files modified.

- [ ] **Step 4: Inspect the final diff and commit history**

Run:

```bash
git status --short --branch
git diff origin/develop...HEAD --check
git diff origin/develop...HEAD --stat
git log --oneline --decorate -8
```

Expected: clean worktree, no whitespace errors, only intended files in the feature diff, and focused commits in the planned order.

- [ ] **Step 5: Prepare the PR summary without pushing**

Summarize:

- logical groups remain concrete and group-shaped;
- disjoint groups of each supported exact stateless class share one executor;
- differing numeric parameters are packed per joint;
- neural, stateful, custom, overlapping, and native paths remain separate;
- zero-tolerance parity tests and dispatch-count tests pass;
- documentation generation and pre-commit pass.

Do not push until the user explicitly requests it; repository instructions prohibit pushing to `origin` and require the fork/PR remote.
