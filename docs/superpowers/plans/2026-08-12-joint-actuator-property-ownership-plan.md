# Joint and Actuator Property Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate articulation-owned joint properties from actuator-owned model parameters, introduce canonical `joint_effort_limit` and `joint_velocity_limit` configuration fields, preserve deprecated compatibility access, and bind implicit execution directly to live articulation state.

**Architecture:** `ActuatorCollection` resolves joint-property overrides before constructing actuator models and sends a construction-only `ActuatorJointProperties` payload to the backend. Ordinary Lab actuators and private executors retain only model inputs; implicit telemetry reads articulation-wide drive arrays directly in one cached Warp launch. Deprecated configuration and group-level accessors forward to the canonical articulation state without adding collection-wide buffers.

**Tech Stack:** Python 3.11, PyTorch, NVIDIA Warp, IsaacLab `configclass`, PhysX, OVPhysX, Newton, pytest, Sphinx/RST.

## Global Constraints

- Do not add dependencies.
- `ActuatorCollection` must expose no `*_sim` or `joint_*` property buffers.
- `effort_limit_sim` and `velocity_limit_sim` remain accepted deprecated aliases through 3.x and are scheduled for removal in 4.0.
- Canonical configuration names are `joint_effort_limit` and `joint_velocity_limit`; actuator `effort_limit` and `velocity_limit` retain model semantics.
- Joint-property configuration remains grouped by actuator joint expressions, but the articulation is its sole runtime owner.
- Ordinary collection-bound actuators and aggregated executors must not retain armature, friction, or solver-limit tensors.
- Implicit stiffness, damping, and effort telemetry must read current `ArticulationData` values without per-step allocation or parameter gather.
- Implicit `velocity_limit` remains a soft-limit snapshot; it must not become a live solver-limit alias during 3.x.
- Native Newton controller gains remain controller-owned and must not be routed through solver joint gains.
- Preserve current explicit actuator default precedence and backend numerical behavior.
- Consolidate existing tests where possible; do not add exhaustive parallel suites or remote-checkpoint tests.
- Follow AGENTS.md API naming, Google docstring, SI-unit, changelog, pre-commit, and deprecation rules.

---

## File Structure

Core responsibilities:

- `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`: canonical configuration fields and deprecated aliases.
- `source/isaaclab/isaaclab/actuators/actuator_base.py`: actuator-model storage and deprecated non-owning joint-property projections.
- `source/isaaclab/isaaclab/actuators/actuator_control.py`: backend-neutral articulation property access and construction-payload writer contract.
- `source/isaaclab/isaaclab/actuators/actuator_collection.py`: copied-config normalization, joint-property resolution, construction sequencing, bindings, and executor construction.
- `source/isaaclab/isaaclab/actuators/actuator_pd.py`: per-class execution parameter declarations and implicit facade.
- `source/isaaclab/isaaclab/actuators/actuator_kernels.py`: allocation-free articulation-wide implicit telemetry kernel.
- `source/isaaclab/isaaclab/envs/mdp/events.py`: implicit and native gain randomization routing.

Backend responsibilities:

- `source/isaaclab_{physx,newton,ovphysx}/isaaclab_*/assets/articulation/actuator_control.py`: backend-specific friction payload writers and native controller projections.
- `source/isaaclab_{physx,newton,ovphysx}/isaaclab_*/assets/articulation/articulation_data.py`: canonical joint-state buffers and rebind lifecycle, unchanged except where a binding hook is required.

Tests and public material:

- Extend existing actuator collection, Ideal PD, implicit, backend articulation, and native randomization tests.
- Migrate active robot/task configs and examples to canonical names.
- Update actuator concepts, articulation configuration, Newton migration, 3.0 migration, API docstrings, and the existing changelog fragment.

---

### Task 1: Canonical joint-limit configuration and deprecated aliases

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base_cfg.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Test: `source/isaaclab/test/actuators/test_actuator_collection.py`

**Interfaces:**
- Produces: `ActuatorBaseCfg.joint_effort_limit` and `ActuatorBaseCfg.joint_velocity_limit`.
- Produces: copied, normalized actuator configs passed to both native preparation and Lab group construction.
- Preserves: deprecated `effort_limit_sim` and `velocity_limit_sim` fields on config objects.

- [ ] **Step 1: Add focused alias tests to the existing collection suite**

Add three tests using the existing fake control and a minimal `IdealPDActuatorCfg`:

```python
def test_deprecated_joint_limit_aliases_warn_and_forward():
    cfg = _ideal_pd_cfg(effort_limit_sim=12.0, velocity_limit_sim=34.0)
    with pytest.warns(DeprecationWarning, match="joint_effort_limit"):
        collection = ActuatorCollection({"motor": cfg}, control)
    assert collection["motor"].cfg.joint_effort_limit == 12.0
    assert collection["motor"].cfg.joint_velocity_limit == 34.0


def test_equivalent_joint_limit_aliases_prefer_canonical_value():
    cfg = _ideal_pd_cfg(
        joint_effort_limit=12.0,
        effort_limit_sim=12.0,
        joint_velocity_limit={"joint_.*": 34.0},
        velocity_limit_sim={"joint_.*": 34.0},
    )
    with pytest.warns(DeprecationWarning):
        collection = ActuatorCollection({"motor": cfg}, control)
    assert collection["motor"].cfg.joint_effort_limit == 12.0


def test_conflicting_joint_limit_aliases_raise():
    cfg = _ideal_pd_cfg(joint_effort_limit=12.0, effort_limit_sim=13.0)
    with pytest.raises(ValueError, match="motor.*joint_effort_limit.*effort_limit_sim"):
        ActuatorCollection({"motor": cfg}, control)
```

Also assert that collection construction does not mutate the caller's config and that the fake native preparation hook receives the normalized copy.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  -k "deprecated_joint_limit_aliases or equivalent_joint_limit_aliases or conflicting_joint_limit_aliases" -q
```

Expected: failures because canonical fields and normalization do not exist.

- [ ] **Step 3: Add canonical config fields and concise deprecation docstrings**

In `ActuatorBaseCfg`, add:

```python
joint_effort_limit: dict[str, float] | float | None = None
"""Joint solver effort limit [N or N·m, depending on joint type]."""

joint_velocity_limit: dict[str, float] | float | None = None
"""Requested joint solver velocity limit [m/s or rad/s, depending on joint type]."""
```

Keep the old fields with `.. deprecated:: 3.0` and explicit replacements/removal in 4.0. Do not add a public `__post_init__` mutation: configs may be changed after creation.

- [ ] **Step 4: Normalize copied configs before any construction consumer**

In `ActuatorCollection.__init__`, first create shallow config copies, resolve aliases on those copies, and pass the resolved mapping to `_resolve_group_joints`, `prepare_native_actuators`, and `_build_groups`:

```python
resolved_cfgs = {name: cfg.copy() for name, cfg in actuator_cfgs.items()}
for name, cfg in resolved_cfgs.items():
    self._resolve_deprecated_limit_aliases(name, cfg)
```

The private resolver warns whenever an alias is set, adopts it only when the canonical value is `None`, accepts equal scalar/dict values, and raises on conflicts. Warning text names the replacement and removal in 4.0. Downstream code reads only canonical fields.

- [ ] **Step 5: Run the full core collection test file**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection.py -q
```

Expected: all tests pass; existing tests may require canonical field updates but no behavioral expectation changes.

- [ ] **Step 6: Run pre-commit and commit**

Run `./isaaclab.sh -f`, restage any formatter edits, rerun it, then commit:

```bash
git commit -m "Deprecate simulation actuator limit names" \
  -m "Introduce joint-qualified solver limit fields and normalize legacy aliases on copied actuator configurations before backend or Lab construction."
```

---

### Task 2: Construction-only joint properties and ordinary actuator storage

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_base.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_control.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_pd.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/actuator_control.py`
- Test: `source/isaaclab/test/actuators/test_actuator_collection.py`
- Test: `source/isaaclab/test/actuators/test_ideal_pd_actuator.py`

**Interfaces:**
- Consumes: normalized `ActuatorBaseCfg.joint_effort_limit` and `.joint_velocity_limit` from Task 1.
- Produces: `write_resolved_joint_properties(properties, joint_ids, *, implicit, native_managed)`.
- Produces: deprecated, non-owning group accessors resolved through `ActuatorControl`.
- Produces: ordinary executors containing only declared model parameters.

- [ ] **Step 1: Rewrite existing storage tests to express ownership**

Extend/consolidate `test_same_stateless_class_builds_one_execution_batch_with_group_views` and the existing init tests. Assert:

```python
assert type(group)._EXECUTION_PARAMETER_NAMES == (
    "effort_limit",
    "velocity_limit",
    "stiffness",
    "damping",
)
for name in ("joint_effort_limit", "joint_velocity_limit", "armature", "friction"):
    assert name not in type(group)._EXECUTION_PARAMETER_NAMES
```

Add a construction test whose fake control records an `ActuatorJointProperties` payload and proves configured armature/friction/joint limits reach the writer although the ordinary group's `__dict__` contains no tensors for them.

Add one compatibility-access test: change the fake articulation joint buffer after construction, access `group.armature` and `group.effort_limit_sim` under `pytest.warns(DeprecationWarning)`, and assert the current selected values are returned rather than construction snapshots.

- [ ] **Step 2: Verify the ownership tests fail before the refactor**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  -k "execution_batch_with_group_views or construction_payload or joint_property_access" -q
```

Expected: failures because joint values are stored on actuators and passed to writers through them.

- [ ] **Step 3: Make joint-property resolution a collection construction phase**

Keep `ActuatorJointProperties` as the backend extension payload. Add a private collection resolver that returns fresh `(num_instances, group_joints)` tensors for joint stiffness, damping, armature, friction fields, and canonical joint limits. Preserve precedence:

```text
explicit joint effort unset -> 1.0e9
implicit joint effort unset -> authored default
joint velocity unset -> authored default
armature/friction unset -> authored default
explicit model effort unset -> authored effort default
explicit model velocity unset -> resolved joint velocity snapshot
```

Resolve every group from authored defaults before writing any group to the backend. Store construction records only until the final write loop.

- [ ] **Step 4: Change the backend-neutral writer contract**

Change the abstract and concrete signatures to:

```python
def write_resolved_joint_properties(
    self,
    properties: ActuatorJointProperties,
    joint_ids: torch.Tensor | _WarpIndex | slice,
    *,
    implicit: bool,
    native_managed: bool,
) -> None:
    ...
```

Change `_write_joint_friction_properties` similarly. PhysX writes all three friction components, Newton writes static plus viscous, and OVPhysX writes the packed triple. No backend writer may read joint properties from an actuator object.

- [ ] **Step 5: Reduce ordinary actuator storage and executor declarations**

Remove solver-only fields from `ActuatorBase`'s owned constructor state and remove them from aggregation. Use concrete declarations:

```python
IdealPDActuator._EXECUTION_PARAMETER_NAMES = (
    "effort_limit",
    "velocity_limit",
    "stiffness",
    "damping",
)
DCMotor._EXECUTION_PARAMETER_NAMES = IdealPDActuator._EXECUTION_PARAMETER_NAMES
```

Keep deprecated group access through a private control binding, not owned tensors. The binding resolves `effort_limit_sim`, `velocity_limit_sim`, armature, and supported friction from current `ArticulationData`. Sliceable selections return views; indexed selections may materialize only on explicit public access. Direct construction outside a collection may retain a compatibility snapshot solely for the deprecated accessor cycle, but collection-bound actuators must discard it when bound.

- [ ] **Step 6: Update debug resolution without retaining runtime joint tensors**

Move joint-resolution table entries onto the construction record or finalize their formatted rows before discarding it. `_print_value_resolution_table()` may consume immutable rows, but it must not require actuator-owned solver tensors.

- [ ] **Step 7: Run core and backend-neutral interface tests**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py -q
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/assets/test_articulation_iface.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py -q
```

Expected: all available backend-neutral cases pass.

- [ ] **Step 8: Run pre-commit and commit**

Run `./isaaclab.sh -f` twice if it edits files, then commit:

```bash
git commit -m "Separate joint properties from actuators" \
  -m "Resolve solver properties as construction payloads, write them through articulation controls, and keep ordinary actuator executors limited to model state."
```

---

### Task 3: Live implicit articulation binding and allocation-free telemetry

**Files:**
- Modify: `source/isaaclab/isaaclab/actuators/actuator_control.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_collection.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_kernels.py`
- Modify: `source/isaaclab/isaaclab/actuators/actuator_pd.py`
- Modify: `source/isaaclab/isaaclab/envs/mdp/events.py`
- Test: `source/isaaclab/test/actuators/test_actuator_collection.py`
- Test: `source/isaaclab/test/actuators/test_implicit_actuator.py`

**Interfaces:**
- Consumes: construction payload and control binding from Task 2.
- Produces: `ActuatorControl.joint_stiffness`, `.joint_damping`, and `.joint_effort_limits` as `ProxyArray` properties.
- Produces: implicit batch kernel inputs indexed in articulation order.
- Preserves: actuator-owned implicit soft `velocity_limit` snapshot.

- [ ] **Step 1: Extend existing implicit and rebind tests**

Modify `test_disjoint_implicit_groups_share_one_execution_batch` to assert that the only local implicit execution parameter is `velocity_limit`.

Extend `test_actuator_batch_rebinds_cuda_state_provider_on_request` so its implicit case replaces position, velocity, stiffness, damping, and effort-limit ProxyArrays, calls `_rebind_state_inputs()`, and proves the next cached launch uses every replacement.

Add one indexed-group assertion to an existing implicit aggregation test:

```python
control.joint_stiffness.torch[:, [0, 2]] = torch.tensor([[11.0, 13.0]])
control.joint_damping.torch[:, [0, 2]] = torch.tensor([[2.0, 3.0]])
control.joint_effort_limits.torch[:, [0, 2]] = torch.tensor([[7.0, 9.0]])
collection.compute()
assert torch.equal(group.computed_effort, expected)
assert torch.equal(group.applied_effort, expected.clamp(-limit, limit))
```

Also verify that changing canonical joint velocity limits after construction does not change the implicit soft `velocity_limit` snapshot.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_implicit_actuator.py \
  -k "implicit or rebinds_cuda_state_provider" -q
```

Expected: current batches still consume actuator-local gain and effort tensors.

- [ ] **Step 3: Expose canonical drive arrays through the control interface**

Add abstract `ProxyArray` properties to `ActuatorControl` and forwarding implementations to `ArticulationActuatorControl`:

```python
@property
def joint_stiffness(self) -> ProxyArray:
    return self._articulation.data.joint_stiffness

@property
def joint_damping(self) -> ProxyArray:
    return self._articulation.data.joint_damping

@property
def joint_effort_limits(self) -> ProxyArray:
    return self._articulation.data.joint_effort_limits
```

Update the fake control with pointer-replaceable ProxyArrays.

- [ ] **Step 4: Read articulation-wide gains in the Warp kernel**

Change inputs 5-7 of `compute_implicit_actuator_batch` to full articulation arrays and index them with `joint_id`:

```python
effort = (
    joint_stiffness[env_id, joint_id] * (position_target - joint_pos[env_id, joint_id])
    + joint_damping[env_id, joint_id] * (velocity_target - joint_vel[env_id, joint_id])
    + feedforward
)
limit = joint_effort_limits[env_id, joint_id]
```

Keep soft `velocity_limit[env_id, batch_joint_id]` batch-shaped. This avoids a gain gather and preserves soft-limit behavior.

- [ ] **Step 5: Bind, rebind, and cache the full implicit inputs**

In `_make_execution_batch`, inputs 5-7 come from `self._control`'s full Warp arrays. Set:

```python
ImplicitActuator._EXECUTION_PARAMETER_NAMES = ("velocity_limit",)
```

Update `_rebind_state_inputs()` to refresh implicit inputs 3-7 and clear `("implicit", id(batch))`; explicit gather batches still refresh only position/velocity. Do not allocate on compute.

- [ ] **Step 6: Add the implicit public facade and update randomization**

Install a private nested joint-drive binding after construction properties are written. Its `stiffness`, `damping`, and `effort_limit` reads query the current control arrays and group selector. It may materialize indexed public reads, but the kernel never uses those reads.

In `randomize_actuator_gains`, stop mutating an implicit actuator tensor before the articulation writer. Write sampled values directly through `write_joint_stiffness_to_sim_index` / `write_joint_damping_to_sim_index`; subsequent actuator reads resolve the live data. Leave native explicit gain updates on `_write_native_actuator_gain`.

- [ ] **Step 7: Run the complete core actuator suite**

Run:

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators/test_actuator_collection.py \
  source/isaaclab/test/actuators/test_implicit_actuator.py \
  source/isaaclab/test/actuators/test_ideal_pd_actuator.py \
  source/isaaclab/test/actuators/test_dc_motor.py -q
```

Expected: all pass on available devices; CUDA rebind test passes when CUDA is available.

- [ ] **Step 8: Run pre-commit and commit**

Run `./isaaclab.sh -f`, restage any edits, rerun, then commit:

```bash
git commit -m "Bind implicit actuators to joint state" \
  -m "Read implicit drive gains and effort limits from live articulation buffers while retaining an independent soft velocity-limit snapshot."
```

---

### Task 4: Preserve backend propagation and native controller ownership

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/actuator_control.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/actuators/adapter.py`
- Test: `source/isaaclab_newton/test/assets/test_articulation.py`
- Test: `source/isaaclab_newton/test/assets/test_newton_actuators_newton.py`
- Test: `source/isaaclab_physx/test/assets/test_newton_actuators_physx.py`
- Test: `source/isaaclab_physx/test/assets/test_articulation.py`
- Test: `source/isaaclab_ovphysx/test/assets/test_articulation.py`

**Interfaces:**
- Consumes: construction-payload writer from Task 2 and live implicit interface from Task 3.
- Preserves: native controller `kp`/`kd` ownership and `_write_native_actuator_gain` routing.
- Produces: named native group parameter projections with no second authoritative copy.

- [ ] **Step 1: Update existing backend assertions before implementation**

In each backend's existing effort/velocity/armature/friction construction tests, assert canonical values through `articulation.data.joint_*`. Retain at most one deprecated group-access assertion per property family under `pytest.warns`, rather than cloning the core alias matrix across backends.

In existing Newton randomization tests, assert native `kp`/`kd` values through `ArticulationView.get_actuator_parameter` after the event writes; do not assert solver `joint_stiffness` for explicit native joints.

- [ ] **Step 2: Run focused backend tests and verify necessary failures**

Run the existing selector subsets:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_articulation.py \
  -k "effort_limit or velocity_limit or armature or friction or native_actuator_gain" -q
env OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/assets/test_articulation.py \
  -k "effort_limit or velocity_limit or armature or friction" -q
```

Expected: assertions tied to actuator-owned joint storage fail until adapters use the new payload/projection interfaces.

- [ ] **Step 3: Adapt backend friction and limit propagation**

Update the three backend controls to consume `ActuatorJointProperties` plus `joint_ids`. Preserve each backend's current supported fields and ordering behavior. Do not add Newton dynamic-friction support as part of this change.

- [ ] **Step 4: Make native named-group gains controller projections**

Keep Newton controller arrays canonical. Named native group gain reads must resolve current `kp`/`kd` through the adapter/control mapping; supported writes and domain randomization must call `_write_native_actuator_gain` directly. Indexed public reads may materialize. Do not create an articulation-wide collection gain mirror and do not route native explicit gains to joint solver writers.

Add one narrow control method and implement it through the adapter's existing
public-to-controller joint mapping:

```python
def get_native_actuator_gain(self, attr: Literal["kp", "kd"], joint_ids: torch.Tensor | slice) -> torch.Tensor | None:
    ...
```

Return `None` on non-native controls. Limit this task to public stiffness/damping because they are the supported runtime randomized native parameters; native effort/velocity configuration remains controller-owned but immutable through the current public runtime API.

- [ ] **Step 5: Verify native and Lab paths**

Run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_newton_actuators_newton.py \
  -k "RandomizeActuatorGainsViaEventsNewton or NewtonActuatorGainSnapshotEnvStride" -q
env OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p -m pytest \
  source/isaaclab_physx/test/assets/test_newton_actuators_physx.py \
  -k "randomize or gain" -q
```

Also run the focused articulation selectors from Step 2 for all locally available backends.

- [ ] **Step 6: Run pre-commit and commit**

Run `./isaaclab.sh -f` twice if needed, then commit:

```bash
git commit -m "Preserve backend actuator ownership" \
  -m "Propagate construction-only joint properties through each backend while keeping Newton controller gains canonical on native execution paths."
```

---

### Task 5: Migrate active configurations and public documentation

**Files:**
- Modify: active files reported by `rg -l 'effort_limit_sim|velocity_limit_sim' source/isaaclab_assets source/isaaclab_tasks scripts docs/source source/isaaclab/isaaclab/test/integration_scene_cfgs.py`
- Modify: `source/isaaclab/changelog.d/actuator-collection.minor.rst`
- Modify: `docs/source/migration/migrating_to_isaaclab_3-0.rst`
- Do not modify: `source/isaaclab/docs/CHANGELOG.rst`
- Do not modify: `docs/source/refs/release_notes.rst`
- Test: `source/isaaclab_tasks/test/core/test_dr_legs_physics_presets.py`

**Interfaces:**
- Consumes: canonical config names and compatibility semantics from Tasks 1-4.
- Produces: active examples/configs using only canonical names.
- Preserves: historical changelog and release-note text.

- [ ] **Step 1: Migrate executable configurations mechanically and review semantics**

In active Python configs and tutorials, replace:

```text
effort_limit_sim   -> joint_effort_limit
velocity_limit_sim -> joint_velocity_limit
```

Do not replace actuator-model `effort_limit` or `velocity_limit`. Review every implicit config to ensure soft `velocity_limit` remains present only when deliberately used by rewards/terminations.

- [ ] **Step 2: Update the public docs around ownership, not suffix history**

Update:

- `docs/source/overview/core-concepts/actuators.rst`
- `docs/source/how-to/write_articulation_cfg.rst`
- `docs/source/how-to/robots.rst`
- `docs/source/how-to/transfer_policies_between_physx_and_newton.rst`
- `docs/source/overview/core-concepts/physical-backends/newton/migrating-assets-from-physx-to-newton.rst`
- relevant policy-deployment and legacy migration examples
- `docs/source/migration/migrating_to_isaaclab_3-0.rst`

State that joint fields in actuator configs are construction-time overrides, runtime joint state lives on `ArticulationData`, ordinary actuators retain only model state, implicit drive values are live articulation projections, and `ActuatorCollection` exposes no joint property API.

- [ ] **Step 3: Update API docstrings and changelog migration guidance**

Use `.. deprecated:: 3.0` for legacy configuration/accessor symbols and name 4.0 removal. Add changelog entries in past tense under `Added`, `Deprecated`, and `Changed` as appropriate. Keep wording functional and concise.

- [ ] **Step 4: Prove active material no longer uses old names**

Run:

```bash
rg -n "effort_limit_sim|velocity_limit_sim" \
  source/isaaclab_assets source/isaaclab_tasks scripts/tutorials docs/source \
  -g "*.py" -g "*.rst"
```

Expected remaining matches only in:

- deprecation/compatibility sections;
- migration tables explicitly showing old-to-new names;
- `docs/source/refs/release_notes.rst` historical content.

- [ ] **Step 5: Run config and documentation checks**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/core/test_dr_legs_physics_presets.py -q
./isaaclab.sh -d
./isaaclab.sh -f
```

If documentation generation reports pre-existing warnings, record them separately and prove no new warnings reference the changed symbols.

- [ ] **Step 6: Commit the migration**

```bash
git commit -m "Document joint actuator ownership" \
  -m "Migrate active configurations to joint-qualified solver limits and document construction, runtime ownership, and compatibility access."
```

---

### Task 6: Final compatibility, convergence, and performance validation

**Files:**
- Modify only for defects found: files from Tasks 1-5
- No new test files unless an uncovered defect cannot fit an existing focused suite

**Interfaces:**
- Consumes: completed ownership migration.
- Produces: evidence that implicit behavior, explicit behavior, native execution, documentation, and performance remain acceptable.

- [ ] **Step 1: Run consolidated core tests**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab/test/actuators \
  source/isaaclab/test/assets/test_articulation_iface.py \
  source/isaaclab/test/assets/test_articulation_ordering_iface.py -q
```

- [ ] **Step 2: Run available backend integration tests**

Run the complete actuator-related PhysX, Newton, and OVPhysX test files on available backends. Record skips separately from passes and investigate every new failure.

- [ ] **Step 3: Run one full-size implicit training smoke**

Use the existing training benchmark with 4096 environments on a core implicit task, keeping the established backend/preset and seed. Confirm convergence remains in the historical range; this is behavioral validation, not a new committed test.

- [ ] **Step 4: Compare implicit runtime against pre-change PR head**

Run the established runtime benchmark with 250 measured steps and 50 warmup steps for Franka Reach or Cartpole on the same idle GPU, configuration, and environment. Compare the ownership-migration head against commit `372cc804c0`'s parent. The new articulation-wide implicit inputs must not regress actuator-step performance beyond measurement noise.

- [ ] **Step 5: Run final repository checks**

```bash
./isaaclab.sh -d
./isaaclab.sh -f
git diff --check
git status --short --branch
```

- [ ] **Step 6: Commit only necessary validation fixes**

If validation required code changes, make one focused fix commit after rerunning its failing test and all final checks. Do not create an empty validation commit.

- [ ] **Step 7: Update PR description and push only after approval**

Summarize the ownership boundary, canonical names, compatibility cycle, implicit live binding, backend behavior, tests, training result, and benchmark. Push only to the PR branch on the fork remote, never `origin`.
