# Articulation Ordering Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make configured articulation joint/body ordering correct at every public/backend boundary while preserving the zero-overhead `None` path and compatibility for third-party articulation backends.

**Architecture:** Keep `BaseArticulation` as the owner of ordering resolution, fixed-base root validation, and cached maps. Perform each runtime permutation exactly once at the backend boundary: reads become public order, writes become backend order, and existing transform/write kernels absorb index mapping where they already launch. Use deterministic mocked value-parity tests as the primary guardrail, then retain one manual fixed-base smoke and one resolver-driven ANYmal-D smoke.

**Tech Stack:** Python 3.12, Warp kernels/arrays, PyTorch tensors, pytest, Isaac Sim PhysX/OVPhysX, Newton/MJWarp, USD.

---

## Task 1: Checkpoint Existing Review Fixes And Rebase

**Files:**
- Verify and commit the existing tracked modifications under `source/isaaclab`, `source/isaaclab_physx`, `source/isaaclab_ovphysx`, and `source/isaaclab_newton`.
- Exclude: `tmp_render_debug/`

- [ ] **Step 1: Confirm the isolated worktree and inspect only tracked changes**

Run:

```bash
git status --short --branch
git diff --check
git diff --stat
```

Expected: branch `antoiner/articulation-ordering`; the 16 known source/test files are modified; `tmp_render_debug/` is untracked; `git diff --check` prints nothing.

- [ ] **Step 2: Verify the existing review-fix batch before committing it**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q
./isaaclab.sh -f
```

Expected: all three commands pass. If formatting changes files, inspect those changes and rerun `./isaaclab.sh -f`.

- [ ] **Step 3: Commit only the known review-fix files**

Run:

```bash
git add source/isaaclab/isaaclab/assets/__init__.pyi source/isaaclab/isaaclab/assets/articulation/__init__.pyi source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py source/isaaclab/isaaclab/assets/articulation/ordering.py source/isaaclab/isaaclab/assets/articulation/ordering_kernels.py source/isaaclab/isaaclab/assets/articulation/ordering_resolvers.py source/isaaclab/test/assets/test_articulation_iface.py source/isaaclab/test/assets/test_articulation_ordering.py source/isaaclab_newton/isaaclab_newton/actuators/kernels.py source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py
git commit -m "Fix articulation ordering data paths"
```

Expected: one focused commit; `tmp_render_debug/` remains untracked.

- [ ] **Step 4: Rebase onto current upstream develop**

Run:

```bash
git fetch origin develop
git rebase origin/develop
```

Resolve conflicts by retaining the new OVPhysX view API and `develop`'s `_reset_body_com_pose_b_dependents()` invalidation behavior, then reapply only ordering-specific additions. Do not stage or delete `tmp_render_debug/`.

- [ ] **Step 5: Verify the rebased baseline**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q
git status --short --branch
```

Expected: both suites pass; no tracked changes remain after the rebase.

## Task 2: Preserve Base Compatibility And Validate Fixed Roots

**Files:**
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation.py:170-250`
- Modify: `source/isaaclab/test/assets/test_articulation_ordering.py:790-840`
- Modify: `source/isaaclab/changelog.d/articulation-ordering.major.rst`
- Modify: `docs/superpowers/specs/2026-06-29-articulation-ordering-correctness-design.md`

- [ ] **Step 1: Write failing compatibility and root-placement tests**

Replace the abstract-contract assertion with a generated legacy subclass that implements every pre-existing abstract member but omits the four ordering members. Add explicit, symbolic, and floating-base root-placement cases:

```python
def _make_legacy_articulation_type() -> type[BaseArticulation]:
    ordering_members = {"backend_joint_names", "backend_body_names", "joint_ordering", "body_ordering"}
    namespace = {}
    for name in BaseArticulation.__abstractmethods__ - ordering_members:
        member = inspect.getattr_static(BaseArticulation, name)
        if isinstance(member, property):
            namespace[name] = property(lambda self, member_name=name: getattr(self, f"_{member_name}", None))
        else:
            namespace[name] = lambda self, *args, **kwargs: None
    return type("LegacyArticulation", (BaseArticulation,), namespace)


def _make_ordering_resolution_articulation(
    *,
    body_ordering,
    backend_body_names: tuple[str, ...],
    is_fixed_base: bool,
):
    data = types.SimpleNamespace(
        joint_ordering=None,
        body_ordering=None,
        joint_names=None,
        body_names=None,
        _apply_ordering_maps_after_resolve=lambda: None,
    )
    return types.SimpleNamespace(
        __backend_name__="mock",
        cfg=types.SimpleNamespace(
            prim_path="/World/Robot",
            joint_ordering=None,
            body_ordering=body_ordering,
        ),
        data=data,
        backend_joint_names=["joint"],
        backend_body_names=list(backend_body_names),
        is_fixed_base=is_fixed_base,
        device="cpu",
    )


def test_base_articulation_ordering_contract_preserves_legacy_subclasses() -> None:
    legacy_type = _make_legacy_articulation_type()
    articulation = object.__new__(legacy_type)
    articulation._joint_names = ["hip", "knee"]
    articulation._body_names = ["base", "foot"]
    articulation._data = types.SimpleNamespace(joint_ordering=None, body_ordering=None)

    with pytest.warns(DeprecationWarning, match="override backend_joint_names"):
        assert articulation.backend_joint_names == ["hip", "knee"]
    with pytest.warns(DeprecationWarning, match="override backend_body_names"):
        assert articulation.backend_body_names == ["base", "foot"]
    assert articulation.joint_ordering is None
    assert articulation.body_ordering is None


@pytest.mark.parametrize("ordering", [("foot", "base"), ArticulationOrderingConvention.MJWARP])
def test_fixed_base_body_ordering_rejects_root_relocation(ordering) -> None:
    data = types.SimpleNamespace(joint_ordering=None, body_ordering=None, joint_names=None, body_names=None)
    articulation = types.SimpleNamespace(
        __backend_name__="mock",
        cfg=types.SimpleNamespace(prim_path="/World/Robot", joint_ordering=None, body_ordering=ordering),
        data=data,
        backend_joint_names=["joint"],
        backend_body_names=["base", "foot"],
        _mjwarp_body_names=("foot", "base"),
        is_fixed_base=True,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid body_ordering for fixed-base articulation '/World/Robot': root body 'base' must remain "
            "at public index 0, but was requested at index 1"
        ),
    ):
        BaseArticulation._resolve_and_install_ordering_maps(articulation)
    assert data.body_ordering is None


def test_floating_base_body_ordering_accepts_root_relocation() -> None:
    articulation = _make_ordering_resolution_articulation(
        body_ordering=("foot", "base"),
        backend_body_names=("base", "foot"),
        is_fixed_base=False,
    )
    BaseArticulation._resolve_and_install_ordering_maps(articulation)
    assert articulation.data.body_names == ["foot", "base"]
```

Add `import inspect` for the generated legacy class.

- [ ] **Step 2: Run the new tests and verify the expected failures**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q -k "legacy_subclasses or root_relocation"
```

Expected: the legacy class remains abstract and cannot instantiate; fixed-base root relocation does not raise the actionable error.

- [ ] **Step 3: Replace the four immediate abstract requirements with compatibility fallbacks**

Implement these concrete properties in `BaseArticulation`:

```python
@property
def backend_joint_names(self) -> list[str]:
    """Ordered joint names exposed by the active backend."""
    warnings.warn(
        f"{type(self).__name__} must override backend_joint_names before it becomes abstract in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.joint_names

@property
def backend_body_names(self) -> list[str]:
    """Ordered body names exposed by the active backend."""
    warnings.warn(
        f"{type(self).__name__} must override backend_body_names before it becomes abstract in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.body_names

@property
def joint_ordering(self) -> ArticulationNameMap | None:
    """Mapping between backend and public joint order."""
    return getattr(self.data, "joint_ordering", None)

@property
def body_ordering(self) -> ArticulationNameMap | None:
    """Mapping between backend and public body order."""
    return getattr(self.data, "body_ordering", None)
```

Import `warnings` at module scope. Keep the built-in backend overrides unchanged so they do not warn.

- [ ] **Step 4: Validate the fixed root after name resolution and before map construction**

Insert this check after `body_user_names` is resolved and before `build_articulation_name_map()`:

```python
if self.is_fixed_base and body_user_names and body_user_names[0] != self.backend_body_names[0]:
    root_body_name = self.backend_body_names[0]
    requested_index = body_user_names.index(root_body_name)
    raise ValueError(
        f"Invalid body_ordering for fixed-base articulation '{self.cfg.prim_path}': root body "
        f"'{root_body_name}' must remain at public index 0, but was requested at index {requested_index}. "
        f"Put '{root_body_name}' first; all remaining bodies may be reordered freely."
    )
```

Do not normalize or silently replace the user's sequence.

- [ ] **Step 5: Update status and migration documentation**

Change the design status to `Approved`. Extend the major changelog fragment with:

```rst
Changed
^^^^^^^

* Changed custom :class:`~isaaclab.assets.BaseArticulation` backends to expose
  backend joint/body names and ordering maps. Existing backends continue to
  work through deprecated fallbacks; override ``backend_joint_names`` and
  ``backend_body_names`` before these properties become abstract in a future
  release.
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q
./isaaclab.sh -f
git add source/isaaclab/isaaclab/assets/articulation/base_articulation.py source/isaaclab/test/assets/test_articulation_ordering.py source/isaaclab/changelog.d/articulation-ordering.major.rst
git add -f docs/superpowers/specs/2026-06-29-articulation-ordering-correctness-design.md
git commit -m "Preserve legacy articulation ordering API"
```

Expected: tests and hooks pass; the warning and fixed-root contract land in one commit.

## Task 3: Cover Floating-Base And Root-Preserving Fixed-Base Dynamics

**Files:**
- Modify: `source/isaaclab/test/assets/test_articulation_iface.py:100-700,1180-1320`
- Modify: `source/isaaclab_physx/test/assets/test_articulation.py:320-430`
- Modify: `source/isaaclab_newton/test/assets/test_articulation.py:540-575`

- [ ] **Step 1: Make shared mock factories model fixed and floating bases explicitly**

Add `is_fixed_base: bool = False`, `joint_ordering: tuple[str, ...] | None = None`, and
`body_ordering: tuple[str, ...] | None = None` to `create_physx_articulation()`,
`create_ovphysx_articulation()`, `create_newton_articulation()`, and `get_articulation()`. Feed them into the configs and real mock metadata:

```python
articulation.cfg = ArticulationCfg(
    prim_path="/World/Robot",
    soft_joint_pos_limit_factor=1.0,
    actuators={},
    joint_ordering=joint_ordering,
    body_ordering=body_ordering,
)
mock_metatype.fixed_base = is_fixed_base
mock_bindings = MockOvPhysxBindingSet(
    num_instances=num_instances,
    num_joints=num_joints,
    num_bodies=num_bodies,
    is_fixed_base=is_fixed_base,
    joint_names=joint_names,
    body_names=body_names,
    num_fixed_tendons=num_fixed_tendons,
    num_spatial_tendons=num_spatial_tendons,
)
mock_view = NewtonMockArticulationView(
    num_instances=num_instances,
    num_bodies=num_bodies,
    num_joints=num_joints,
    device=device,
    is_fixed_base=is_fixed_base,
    joint_names=joint_names,
    body_names=body_names,
)
total_dofs = num_joints + (0 if is_fixed_base else 6)
```

Set OVPhysX articulation/data `_is_fixed_base` from the argument and forward all three arguments through
`get_articulation()`. This lets tests exercise real initialization-time map and staging-buffer setup.

- [ ] **Step 2: Add a root-preserving body permutation helper**

```python
def _root_preserving_reversed_body_names(art) -> tuple[str, ...]:
    """Keep a fixed root first and reverse every remaining body."""
    return (art.backend_body_names[0], *reversed(art.backend_body_names[1:]))


def _install_test_body_ordering(art) -> np.ndarray:
    """Install a valid maximally different public body ordering."""
    if art.is_fixed_base:
        body_names = _root_preserving_reversed_body_names(art)
    else:
        body_names = tuple(reversed(art.backend_body_names))
    art.cfg = art.cfg.replace(body_ordering=body_names)
    art._resolve_and_install_ordering_maps()
    art._cache_ordering_maps()
    return np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
```

Use this helper in all shared tests instead of always moving body zero.

- [ ] **Step 3: Extend the dynamics parity test to fixed-base articulations**

Parametrize the existing dynamics test with `is_fixed_base` and construct both identity/ordered instances with the same base type:

```python
@_dynamics_ordering_backends
@pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
@pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 3, 4)])
@pytest.mark.parametrize("device", ["cpu"])
def test_ordering_reorders_public_dynamics_quantities(
    self, backend, is_fixed_base, num_instances, num_joints, num_bodies, device
):
    identity_art, identity_raw = get_articulation(
        backend, num_instances, num_joints, num_bodies, device=device, is_fixed_base=is_fixed_base
    )
    ordered_art, ordered_raw = get_articulation(
        backend, num_instances, num_joints, num_bodies, device=device, is_fixed_base=is_fixed_base
    )
```

Retain the existing expected-value construction using `_jacobian_body_user_to_backend()` and `_generalized_dof_user_to_backend()`. For fixed base, assert the map excludes backend root zero and preserves the requested order of all remaining bodies.

- [ ] **Step 4: Verify the fixed-base case catches the old root-axis assumption**

Temporarily replace the fixed-base Jacobian map in each tested backend with the old unpermuted `range(num_bodies - 1)` behavior, run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q -k "dynamics_quantities and fixed"
```

Expected: FAIL with body Jacobian values in backend rather than root-preserving public order. Restore the implementation immediately and rerun; expected PASS.

- [ ] **Step 5: Make live Panda and Newton rebind tests root-preserving**

Define the explicit Panda order once in each live test module:

```python
_PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES = (
    _PANDA_BODY_NAMES[0],
    *reversed(_PANDA_BODY_NAMES[1:]),
)
```

Rename the PhysX test to `test_live_manual_root_preserving_ordering_reorders_backend_reads_and_writes` and use this body sequence. Use the same sequence in `test_newton_ordered_state_caches_invalidate_on_rebind`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q -k "dynamics_quantities"
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py::test_live_manual_root_preserving_ordering_reorders_backend_reads_and_writes -q
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_articulation.py::test_newton_ordered_state_caches_invalidate_on_rebind -q
./isaaclab.sh -f
git add source/isaaclab/test/assets/test_articulation_iface.py source/isaaclab_physx/test/assets/test_articulation.py source/isaaclab_newton/test/assets/test_articulation.py
git commit -m "Test fixed-base articulation ordering"
```

## Task 4: Convert External Wrenches At Each Backend Boundary

**Files:**
- Modify: `source/isaaclab/isaaclab/assets/articulation/ordering_kernels.py`
- Modify: `source/isaaclab/test/assets/test_articulation_iface.py`
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:347-375,4069-4100`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/kernels.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py:330-370`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/kernels.py:938-975`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py:300-335`

- [ ] **Step 1: Write a failing backend-value parity test for body wrenches**

Add helpers that set distinct public forces/torques and capture the actual backend buffer. Parametrize all three real mock backends and both base types:

```python
def _set_identity_body_poses(backend: str, art, raw_backend) -> None:
    """Give the OVPhysX wrench transform deterministic identity rotations."""
    if backend != "ovphysx":
        return
    from isaaclab_ovphysx import tensor_types as TT
    poses = np.zeros((art.num_instances, art.num_bodies, 7), dtype=np.float32)
    poses[..., 6] = 1.0
    raw_backend.bindings[TT.LINK_POSE]._data = poses
    art.data._reset_pose()


def _read_backend_wrench(backend: str, art, raw_backend, captured: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return force and torque from the concrete backend write target."""
    if backend == "physx":
        return captured["force"], captured["torque"]
    if backend == "newton":
        wrench = art.data._sim_bind_body_external_wrench.numpy()
        return wrench[..., :3], wrench[..., 3:6]
    from isaaclab_ovphysx import tensor_types as TT
    wrench = raw_backend.bindings[TT.LINK_WRENCH]._data
    return wrench[..., :3], wrench[..., 3:6]


@_non_mock_backends
@pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
@pytest.mark.parametrize("device", ["cpu"])
def test_external_wrenches_are_written_in_backend_body_order(self, backend, is_fixed_base, device):
    num_instances, num_joints, num_bodies = 2, 1, 4
    backend_body_names = tuple(f"body_{index}" for index in range(num_bodies))
    if is_fixed_base:
        body_ordering = (backend_body_names[0], *reversed(backend_body_names[1:]))
    else:
        body_ordering = tuple(reversed(backend_body_names))
    art, raw_backend = get_articulation(
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device=device,
        is_fixed_base=is_fixed_base,
        body_ordering=body_ordering,
    )
    _set_identity_body_poses(backend, art, raw_backend)
    captured = {}
    if backend == "physx":
        def capture_wrench(*, force_data, torque_data, position_data, indices, is_global):
            captured["force"] = force_data.numpy().reshape(num_instances, num_bodies, 3)
            captured["torque"] = torque_data.numpy().reshape(num_instances, num_bodies, 3)

        raw_backend.apply_forces_and_torques_at_position = capture_wrench

    forces = np.arange(num_instances * num_bodies * 3, dtype=np.float32).reshape(num_instances, num_bodies, 3)
    torques = forces + 100.0
    art.instantaneous_wrench_composer.set_forces_and_torques_index(
        forces=wp.array(forces, dtype=wp.vec3f, device=device),
        torques=wp.array(torques, dtype=wp.vec3f, device=device),
    )

    art.write_data_to_sim()

    backend_to_user = np.asarray(art.body_ordering.backend_to_user_indices, dtype=np.int64)
    backend_force, backend_torque = _read_backend_wrench(backend, art, raw_backend, captured)
    np.testing.assert_allclose(backend_force, forces[:, backend_to_user])
    np.testing.assert_allclose(backend_torque, torques[:, backend_to_user])
```

`_read_backend_wrench()` must inspect the real setter input for PhysX, `data._sim_bind_body_external_wrench` for Newton, and `TT.LINK_WRENCH` for OVPhysX. It must unpack the complete backend structures rather than asserting on the composer mock.

- [ ] **Step 2: Run the test and verify all three ordered cases fail by permutation**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q -k "external_wrenches_are_written"
```

Expected: non-identity cases fail because backend body `j` receives public body `j`; identity/no-order control behavior remains correct.

- [ ] **Step 3: Add one fused PhysX force/torque reorder kernel**

Add this kernel to shared `ordering_kernels.py`:

```python
@wp.kernel
def reorder_body_wrench_user_to_backend(
    user_force: wp.array2d(dtype=wp.vec3f),
    user_torque: wp.array2d(dtype=wp.vec3f),
    backend_to_user: wp.array(dtype=wp.int32),
    backend_force: wp.array2d(dtype=wp.vec3f),
    backend_torque: wp.array2d(dtype=wp.vec3f),
):
    """Reorder public body-frame force and torque into backend body order."""
    env_index, backend_body_index = wp.tid()
    user_body_index = backend_to_user[backend_body_index]
    backend_force[env_index, backend_body_index] = user_force[env_index, user_body_index]
    backend_torque[env_index, backend_body_index] = user_torque[env_index, user_body_index]
```

After `_cache_ordering_maps()`, allocate two `wp.vec3f` buffers only when `_has_body_ordering` is true; otherwise store `None`. In `write_data_to_sim()`, keep direct composer buffers on the `None`/identity path and launch this one kernel on the non-identity path before flattening for the Tensor API.

- [ ] **Step 4: Map the Newton output index inside its existing update launch**

Add an articulation-specific kernel without changing rigid-object callers:

```python
@wp.kernel
def update_wrench_array_with_force_and_torque_ordered(
    forces: wp.array2d(dtype=wp.vec3f),
    torques: wp.array2d(dtype=wp.vec3f),
    user_to_backend: wp.array(dtype=wp.int32),
    has_ordering: wp.bool,
    wrench: wp.array2d(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
    body_mask: wp.array(dtype=wp.bool),
):
    env_index, user_body_index = wp.tid()
    if env_mask[env_index] and body_mask[user_body_index]:
        backend_body_index = user_body_index
        if has_ordering:
            backend_body_index = user_to_backend[user_body_index]
        wrench[env_index, backend_body_index] = wp.spatial_vector(
            forces[env_index, user_body_index], torques[env_index, user_body_index], wp.float32
        )
```

Launch it from Newton `write_data_to_sim()` with `_body_user_to_backend` and `_has_body_ordering`. This remains one launch for both ordered and default paths.

- [ ] **Step 5: Map the OVPhysX output index inside the existing world-frame conversion**

Create an articulation variant of `_body_wrench_to_world` that takes `user_to_backend` and `has_ordering`. Read force, torque, and pose using `user_body_index`; write all nine output floats using `backend_body_index`. Launch it once from the articulation path. Leave rigid-object and collection callers on the existing unmapped kernel.

- [ ] **Step 6: Run parity tests and commit**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q -k "external_wrenches_are_written"
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q
./isaaclab.sh -f
git add source/isaaclab/isaaclab/assets/articulation/ordering_kernels.py source/isaaclab/test/assets/test_articulation_iface.py source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py source/isaaclab_newton/isaaclab_newton/assets/articulation/kernels.py source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py source/isaaclab_ovphysx/isaaclab_ovphysx/assets/kernels.py source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py
git commit -m "Fix external wrench body ordering"
```

Expected: all six ordered wrench cases pass and no extra PhysX reorder buffer exists on the `None` path.

## Task 5: Fix OVPhysX Public-Order Differencing And Cache Invalidation

**Files:**
- Modify: `source/isaaclab/test/assets/test_articulation_iface.py:1200-1260,1540-1580`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py:189-250,1109-1135`
- Modify: `source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py:set_coms_index,set_coms_mask`

- [ ] **Step 1: Add a failing ordered joint-acceleration test**

```python
def test_ovphysx_joint_acceleration_differences_public_order_velocities(self):
    if "ovphysx" not in BACKENDS:
        pytest.skip("OVPhysX backend is not available")
    art, raw_backend = get_articulation("ovphysx", 2, 3, 2, device="cpu")
    user_to_backend = _install_reversed_joint_ordering(art)
    from isaaclab_ovphysx import tensor_types as TT

    first = np.asarray([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]], dtype=np.float32)
    second = first + np.asarray([[3.0, 5.0, 7.0], [11.0, 13.0, 17.0]], dtype=np.float32)
    raw_backend.bindings[TT.DOF_VELOCITY]._data = first
    art.data.update(0.1)
    art.data.joint_acc.torch.clone()
    raw_backend.bindings[TT.DOF_VELOCITY]._data = second
    art.data.update(0.1)

    torch.testing.assert_close(art.data.joint_vel.torch, torch.from_numpy(second[:, user_to_backend]))
    torch.testing.assert_close(
        art.data.joint_acc.torch,
        torch.from_numpy((second - first)[:, user_to_backend] / 0.1),
    )
```

- [ ] **Step 2: Expand the same-step velocity cache regression**

Add these two explicit tests:

```python
def test_ovphysx_reversed_body_ordering_rereads_all_velocity_shadows_after_reset(self):
    if "ovphysx" not in BACKENDS:
        pytest.skip("OVPhysX backend is not available")
    art, raw_backend = get_articulation(
        "ovphysx",
        2,
        1,
        3,
        device="cpu",
        body_ordering=("body_2", "body_1", "body_0"),
    )
    from isaaclab_ovphysx import tensor_types as TT

    raw_data = list(_make_body_ordering_backend_data(2, 3))
    raw_data[3][..., :3] = 0.0
    _set_body_ordering_backend_data("ovphysx", art, raw_backend, *raw_data)
    user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
    art.data.update(0.01)
    art.data.body_com_vel_w.torch.clone()
    art.data.body_link_vel_w.torch.clone()
    art.data.root_link_vel_w.torch.clone()

    next_velocity = raw_data[4].copy()
    next_velocity[..., 0] += 500.0
    raw_backend.bindings[TT.LINK_VELOCITY]._data = next_velocity
    art.data._reset_velocity()

    expected = torch.from_numpy(next_velocity[:, user_to_backend])
    _assert_proxy_close(art.data.body_com_vel_w, expected)
    _assert_proxy_close(art.data.body_link_vel_w, expected)
    _assert_proxy_close(art.data.root_link_vel_w, torch.from_numpy(next_velocity[:, 0]))


def test_ovphysx_com_write_invalidates_all_dependent_caches_under_ordering(self):
    if "ovphysx" not in BACKENDS:
        pytest.skip("OVPhysX backend is not available")
    art, _ = get_articulation(
        "ovphysx",
        2,
        1,
        3,
        device="cpu",
        body_ordering=("body_2", "body_1", "body_0"),
    )
    cache_names = (
        "_root_com_pose_w",
        "_root_com_vel_w",
        "_root_link_vel_w",
        "_body_com_pose_w",
        "_body_com_vel_w",
        "_body_com_vel_w_backend",
        "_body_link_vel_w",
        "_body_link_vel_w_backend",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_root_state_w_buf",
        "_root_link_state_w_buf",
        "_root_com_state_w_buf",
        "_body_state_w_buf",
        "_body_link_state_w_buf",
        "_body_com_state_w_buf",
    )
    for cache_name in cache_names:
        getattr(art.data, cache_name).timestamp = art.data._sim_timestamp

    coms = np.zeros((art.num_instances, art.num_bodies, 7), dtype=np.float32)
    coms[..., 6] = 1.0
    art.set_coms_index(coms=wp.array(coms, dtype=wp.transformf, device=art.device))

    for cache_name in cache_names:
        assert getattr(art.data, cache_name).timestamp == -1.0, cache_name
```

- [ ] **Step 3: Run both tests and verify the expected failures**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q -k "joint_acceleration_differences or rereads_backend_shadows or com_dependents"
```

Expected: joint acceleration is permuted incorrectly; `root_link_vel_w` can reuse `_body_link_vel_w_backend`; any missing COM dependent remains current instead of invalidated.

- [ ] **Step 4: Difference the public velocity buffer**

In `joint_acc`, obtain the current velocity through the existing public-order getter before launching `_fd_joint_acc`:

```python
if self._joint_acc.timestamp < self._sim_timestamp:
    time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
    joint_vel = self.joint_vel.warp
    wp.launch(
        _fd_joint_acc,
        dim=(self._num_instances, self._num_joints),
        inputs=[joint_vel, self._previous_joint_vel, 1.0 / time_elapsed],
        outputs=[self._joint_acc.data],
        device=self.device,
    )
    self._joint_acc.timestamp = self._sim_timestamp
```

Do not read `DOF_VELOCITY` directly into `_joint_vel_buf` from this property.

- [ ] **Step 5: Invalidate every backend shadow used by velocity reads**

Preserve every entry from rebased `develop` and add both ordering shadows to the corresponding reset lists:

```python
# In _reset_velocity()
reset_timestamps(
    [
        self._root_link_vel_w if from_com else None,
        self._body_com_vel_w,
        self._body_com_vel_w_backend,
        self._body_link_vel_w,
        self._body_link_vel_w_backend,
        self._root_state_w_buf,
        self._root_link_state_w_buf,
        self._root_com_state_w_buf,
        self._body_state_w_buf,
        self._body_link_state_w_buf,
        self._body_com_state_w_buf,
    ]
)

# In _reset_body_com_pose_b_dependents(), append these to develop's list.
self._body_com_vel_w_backend,
self._body_link_vel_w_backend,
```

Both `set_coms_index()` and `set_coms_mask()` must call `self.data._reset_body_com_pose_b_dependents()` instead of maintaining narrower local timestamp assignments.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q -k "ovphysx"
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q
./isaaclab.sh -f
git add source/isaaclab/test/assets/test_articulation_iface.py source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation_data.py source/isaaclab_ovphysx/isaaclab_ovphysx/assets/articulation/articulation.py
git commit -m "Fix OVPhysX ordered state caching"
```

## Task 6: Normalize Newton Actuator Defaults Once

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/actuators/adapter.py:235-305`
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py:3680-3705`
- Modify: `source/isaaclab/test/assets/test_articulation_ordering.py`

- [ ] **Step 1: Add a failing heterogeneous-gain snapshot test**

```python
def test_newton_actuator_defaults_follow_requested_public_joint_order() -> None:
    from isaaclab_newton.actuators.adapter import build_newton_actuator_defaults

    controller = types.SimpleNamespace(
        kp=wp.array((10.0, 30.0, 11.0, 31.0), dtype=wp.float32, device="cpu"),
        kd=wp.array((1.0, 3.0, 1.1, 3.1), dtype=wp.float32, device="cpu"),
    )
    actuator = types.SimpleNamespace(
        controller=controller,
        indices=wp.array((0, 2, 3, 5), dtype=wp.int32, device="cpu"),
    )

    stiffness, damping, managed = build_newton_actuator_defaults(
        actuators=[actuator],
        num_envs=2,
        num_joints=3,
        dof_offset=0,
        device="cpu",
        joint_user_to_backend_indices=(2, 1, 0),
    )

    torch.testing.assert_close(stiffness, torch.tensor([[30.0, 0.0, 10.0], [31.0, 0.0, 11.0]]))
    torch.testing.assert_close(damping, torch.tensor([[3.0, 0.0, 1.0], [3.1, 0.0, 1.1]]))
    torch.testing.assert_close(managed, torch.tensor([0, 2], dtype=torch.int32))
```

- [ ] **Step 2: Run the test and verify the missing API fails**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q -k "actuator_defaults_follow"
```

Expected: FAIL because `build_newton_actuator_defaults()` does not accept `joint_user_to_backend_indices` and returns backend-order columns.

- [ ] **Step 3: Add optional one-time output normalization to the builder**

Extend the signature with `joint_user_to_backend_indices: Sequence[int] | None = None` and import `Sequence` from `collections.abc`. After backend-order scatter completes:

```python
if joint_user_to_backend_indices is not None:
    user_to_backend = tuple(int(index) for index in joint_user_to_backend_indices)
    if sorted(user_to_backend) != list(range(num_joints)):
        raise ValueError("joint_user_to_backend_indices must be a complete joint permutation.")
    user_to_backend_tensor = torch.tensor(user_to_backend, dtype=torch.long, device=device)
    stiffness = stiffness.index_select(1, user_to_backend_tensor)
    damping = damping.index_select(1, user_to_backend_tensor)
    if not isinstance(joint_indices, slice):
        backend_to_user = [0] * num_joints
        for user_index, backend_index in enumerate(user_to_backend):
            backend_to_user[backend_index] = user_index
        joint_indices = torch.tensor(
            sorted(backend_to_user[index] for index in managed_local),
            dtype=torch.int32,
            device=device,
        )
```

Document that the optional permutation converts backend-local outputs into public order. Keep the PhysX call unchanged.

- [ ] **Step 4: Pass the Newton articulation's static CPU map**

At the Newton call site add:

```python
joint_user_to_backend_indices=(
    self.joint_ordering.user_to_backend_indices if self._has_joint_ordering else None
),
```

This occurs once during initialization and adds nothing to domain-randomization or simulation-step hot paths.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q -k "actuator_defaults_follow"
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q
./isaaclab.sh -f
git add source/isaaclab_newton/isaaclab_newton/actuators/adapter.py source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py source/isaaclab/test/assets/test_articulation_ordering.py
git commit -m "Order Newton actuator defaults by joint"
```

## Task 7: Add Deterministic Resolver And Live Sim Coverage

**Files:**
- Create: `source/isaaclab_physx/test/assets/data/articulation_ordering_branching.usda`
- Modify: `source/isaaclab_physx/test/assets/test_articulation.py`
- Modify: `source/isaaclab_newton/test/assets/test_articulation.py:510-540`

- [ ] **Step 1: Create a local branching articulation fixture**

Author a five-body tree with two two-link branches. The topology and names are:

```text
base
|- left_shoulder -> left_upper
|  `- left_elbow -> left_tip
`- right_shoulder -> right_upper
   `- right_elbow -> right_tip
```

Write the fixture with this complete content:

```usda
#usda 1.0
(
    defaultPrim = "Robot"
    upAxis = "Z"
)

def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }

    def Xform "left_upper" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }

    def Xform "left_tip" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }

    def Xform "right_upper" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }

    def Xform "right_tip" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }

    def PhysicsRevoluteJoint "left_shoulder"
    {
        uniform token physics:axis = "Z"
        rel physics:body0 = </Robot/base>
        rel physics:body1 = </Robot/left_upper>
    }

    def PhysicsRevoluteJoint "left_elbow"
    {
        uniform token physics:axis = "Z"
        rel physics:body0 = </Robot/left_upper>
        rel physics:body1 = </Robot/left_tip>
    }

    def PhysicsRevoluteJoint "right_shoulder"
    {
        uniform token physics:axis = "Z"
        rel physics:body0 = </Robot/base>
        rel physics:body1 = </Robot/right_upper>
    }

    def PhysicsRevoluteJoint "right_elbow"
    {
        uniform token physics:axis = "Z"
        rel physics:body0 = </Robot/right_upper>
        rel physics:body1 = </Robot/right_tip>
    }
}
```

This topology yields explicit convention expectations:

```python
expected_physx_joint_names = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
expected_mjwarp_joint_names = ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow")
expected_physx_body_names = ("base", "left_upper", "right_upper", "left_tip", "right_tip")
expected_mjwarp_body_names = ("base", "left_upper", "left_tip", "right_upper", "right_tip")
```

- [ ] **Step 2: Add a resolver test that asserts both concrete sequences**

Add `from pathlib import Path` and import `get_mjwarp_articulation_name_ordering`. Then add:

```python
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_branching_fixture_resolves_distinct_conventions(sim, device, gravity_enabled):
    fixture_path = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(fixture_path)),
            actuators={},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    sim.reset()
    assert articulation.is_initialized

    expected_physx_joint_names = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
    expected_mjwarp_joint_names = ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow")
    expected_physx_body_names = ("base", "left_upper", "right_upper", "left_tip", "right_tip")
    expected_mjwarp_body_names = ("base", "left_upper", "left_tip", "right_upper", "right_tip")

    assert tuple(articulation.backend_joint_names) == expected_physx_joint_names
    assert tuple(articulation.backend_body_names) == expected_physx_body_names
    assert get_mjwarp_articulation_name_ordering(articulation, "joint") == expected_mjwarp_joint_names
    assert get_mjwarp_articulation_name_ordering(articulation, "body") == expected_mjwarp_body_names
    assert tuple(articulation.joint_names) == expected_mjwarp_joint_names
    assert tuple(articulation.body_names) == expected_mjwarp_body_names
    assert articulation.joint_ordering is not None and not articulation.joint_ordering.is_identity
    assert articulation.body_ordering is not None and not articulation.body_ordering.is_identity
```

- [ ] **Step 3: Verify the resolver test cannot pass through identity fallback**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -q -k "branching_fixture_resolves_distinct_conventions"
```

Expected before the test fixture/resolver wiring is complete: FAIL on one of the explicit convention tuples, never merely on shape.

- [ ] **Step 4: Replace the artificial Newton resolver case**

Remove `_PhysxLikeArticulation` with manually reversed names from `test_mjwarp_ordering_resolver_matches_newton_backend_names`. Keep the same-backend Newton identity assertion only; the branching fixture now owns cross-backend traversal correctness.

- [ ] **Step 5: Add a resolver-driven ANYmal-D PhysX smoke**

Import `ANYMAL_D_CFG` and `apply_articulation_ordering_preset`, then add the full boundary check:

```python
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_live_anymal_d_mjwarp_ordering_reorders_named_state(sim, device, gravity_enabled):
    cfg = apply_articulation_ordering_preset(
        ANYMAL_D_CFG.replace(prim_path="/World/Robot"),
        "mjwarp",
    )
    articulation = Articulation(cfg)
    sim.reset()
    assert articulation.is_initialized

    joint_non_identity = articulation.joint_ordering is not None and not articulation.joint_ordering.is_identity
    body_non_identity = articulation.body_ordering is not None and not articulation.body_ordering.is_identity
    assert joint_non_identity or body_non_identity, (
        "ANYmal-D no longer produces a non-identity PhysX/MJWarp ordering map; choose a representative "
        "branching asset before treating this as cross-backend coverage."
    )

    joint_user_to_backend = list(articulation.joint_ordering.user_to_backend_indices)
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    body_user_to_backend = list(articulation.body_ordering.user_to_backend_indices)
    joint_pos = torch.linspace(-0.1, 0.1, articulation.num_joints, device=device).unsqueeze(0)
    joint_vel = torch.linspace(0.01, 0.02, articulation.num_joints, device=device).unsqueeze(0)

    articulation.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, full_data=True)
    articulation.write_data_to_sim()

    _assert_user_write_reaches_backend(
        joint_pos,
        _to_device_tensor(articulation.root_view.get_dof_positions(), device),
        joint_backend_to_user,
    )
    _assert_user_write_reaches_backend(
        joint_vel,
        _to_device_tensor(articulation.root_view.get_dof_velocities(), device),
        joint_backend_to_user,
    )

    sim.step()
    articulation.update(sim.cfg.dt)
    _assert_backend_to_user(
        articulation.data.joint_pos.torch,
        _to_device_tensor(articulation.root_view.get_dof_positions(), device),
        joint_user_to_backend,
    )
    _assert_backend_to_user(
        articulation.data.body_link_pose_w.torch,
        _to_device_tensor(articulation.root_view.get_link_transforms(), device),
        body_user_to_backend,
    )
```

- [ ] **Step 6: Run live tests and commit**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -q -k "manual_root_preserving_ordering or branching_fixture_resolves_distinct_conventions or anymal_d_mjwarp_ordering"
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_articulation.py -q -k "mjwarp_ordering_resolver or ordered_state_caches"
./isaaclab.sh -f
git add source/isaaclab_physx/test/assets/data/articulation_ordering_branching.usda source/isaaclab_physx/test/assets/test_articulation.py source/isaaclab_newton/test/assets/test_articulation.py
git commit -m "Add live articulation ordering coverage"
```

## Task 8: Final Verification And Branch Review

**Files:**
- Review all files changed since `origin/develop`.
- Do not add: `tmp_render_debug/`

- [ ] **Step 1: Run focused ordering and backend suites**

Run:

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_ordering.py -q
./isaaclab.sh -p -m pytest source/isaaclab/test/assets/test_articulation_iface.py -q
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -q -k "ordering"
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_articulation.py -q -k "ordering or ordered_state_caches"
```

Expected: all selected tests pass with no tracebacks or new warnings.

- [ ] **Step 2: Build public documentation and run all hooks**

Run:

```bash
./isaaclab.sh -d
./isaaclab.sh -f
```

Expected: public API documentation builds; every pre-commit hook passes. If a hook modifies files, inspect/stage them and rerun `./isaaclab.sh -f`.

- [ ] **Step 3: Audit performance-sensitive paths and the final diff**

Run:

```bash
git diff --check origin/develop...HEAD
git diff --stat origin/develop...HEAD
git status --short --branch
```

Inspect the full diff and confirm:

- `joint_ordering=None` and `body_ordering=None` construct no name maps.
- PhysX allocates no body-wrench staging arrays and launches no reorder kernel on the `None` path.
- Newton and OVPhysX fuse body-index mapping into launches they already perform.
- Fixed-base validation runs only during ordering resolution.
- Newton gain normalization runs only during actuator initialization.
- Built-in backends override all four base compatibility properties.
- `tmp_render_debug/` remains untracked and absent from every commit.

- [ ] **Step 4: Run the PR-review toolkit and address only verified findings**

Use `pr-review-toolkit:review-pr` on `origin/develop...HEAD`. For each finding, verify the concrete data flow and add a failing regression before changing behavior. Repeat the focused suite and `./isaaclab.sh -f` after any correction.

- [ ] **Step 5: Report the verified commit list and test evidence**

Run:

```bash
git log --oneline origin/develop..HEAD
git status --short --branch
```

Expected: focused commits are visible, the branch has no tracked modifications, and only the pre-existing untracked debug directory may remain.
