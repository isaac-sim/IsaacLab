# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Mocked articulation backends for the contract tests.

Each factory bypasses ``__init__`` and sets the attributes that ``_initialize_impl`` would create, so the real
articulation and data classes run against mocked views. They return the articulation and the raw backend view or
binding set.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

from .backends import (
    AVAILABLE,
    finish_shell,
    indices,
    install_physx_recording_setters,
    newton_manager,
    patch_ovphysx_manager,
    patch_physx_manager,
)

if "physx" in AVAILABLE:
    import isaaclab_physx.assets as physx_assets
    from isaaclab_physx.test.fixtures.views import MockArticulationViewWarp

if "newton" in AVAILABLE:
    import isaaclab_newton.assets as newton_assets
    import isaaclab_newton.assets.articulation.articulation_data as newton_articulation_data
    from isaaclab_newton.test.fixtures.views import MockNewtonArticulationView

if "ovphysx" in AVAILABLE:
    import isaaclab_ov.assets as ovphysx_assets
    from isaaclab_ov import tensor_types as TT
    from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet

_PHYSX_STORAGE = {
    "set_root_transforms": "_root_transforms",
    "set_root_velocities": "_root_velocities",
    "set_dof_positions": "_dof_positions",
    "set_dof_velocities": "_dof_velocities",
    "set_dof_position_targets": "_dof_position_targets",
    "set_dof_velocity_targets": "_dof_velocity_targets",
    "set_dof_actuation_forces": "_dof_actuation_forces",
    "set_dof_limits": "_dof_limits",
    "set_dof_stiffnesses": "_dof_stiffnesses",
    "set_dof_dampings": "_dof_dampings",
    "set_dof_max_forces": "_dof_max_forces",
    "set_dof_max_velocities": "_dof_max_velocities",
    "set_dof_armatures": "_dof_armatures",
    "set_dof_friction_coefficients": "_dof_friction_coefficients",
    "set_dof_friction_properties": "_dof_friction_properties",
    "set_masses": "_masses",
    "set_coms": "_coms",
    "set_inertias": "_inertias",
}


def _names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(count)]


def _shell(cls, joint_ordering, body_ordering):
    """Create an uninitialized articulation of ``cls`` with a minimal configuration."""
    articulation = object.__new__(cls)
    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
    )
    articulation._sim_cfg = None
    articulation._fixed_tendon_target_dirty = False
    return articulation


def _set_tendon_names(articulation, fixed: list[str], spatial: list[str]) -> None:
    articulation._fixed_tendon_names = articulation.data.fixed_tendon_names = fixed
    articulation._spatial_tendon_names = articulation.data.spatial_tendon_names = spatial


def _physx_articulation(N, J, B, FT, ST, device, is_fixed_base, joint_ordering, body_ordering, monkeypatch):
    patch_physx_manager(monkeypatch)
    view = MockArticulationViewWarp(
        count=N, num_links=B, num_dofs=J, device=device, max_fixed_tendons=FT, max_spatial_tendons=ST
    )
    view.set_random_mock_data()
    install_physx_recording_setters(view, _PHYSX_STORAGE)
    view._shared_metatype = MagicMock(
        fixed_base=is_fixed_base, dof_count=J, link_count=B, dof_names=_names("joint", J), link_names=_names("body", B)
    )

    articulation = _shell(physx_assets.Articulation, joint_ordering, body_ordering)
    articulation._root_view = view
    articulation._device = device
    articulation._clamped_default_count = wp.zeros(1, dtype=wp.int32, device=device)
    articulation._data = physx_assets.ArticulationData(view, device)
    _set_tendon_names(articulation, _names("fixed_tendon", FT), _names("spatial_tendon", ST))
    finish_shell(articulation)
    articulation._ALL_INDICES = articulation._ALL_INDICES_WP = indices(N, device)
    articulation._ALL_BODY_INDICES = articulation._ALL_BODY_INDICES_WP = indices(B, device)
    articulation._ALL_JOINT_INDICES = indices(J, device)
    articulation._ALL_FIXED_TENDON_INDICES = indices(FT, device)
    articulation._ALL_SPATIAL_TENDON_INDICES = indices(ST, device)
    for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target", "body_wrench_force"):
        setattr(articulation, f"_{name}_backend", None)
    articulation._body_wrench_torque_backend = None
    articulation._resolve_and_install_ordering_maps()
    articulation._ordering_configure_backend_staging()
    # Cached ``.view(wp.float32)`` wrappers.
    for name in ("root_link_pose_w", "root_com_vel_w", "root_link_vel_w", "inst_wrench_force", "inst_wrench_torque"):
        setattr(articulation, f"_{name}_f32", None)
    articulation._perm_wrench_force_f32 = articulation._perm_wrench_torque_f32 = None
    # Pinned CPU staging buffers for PhysX TensorAPI writes.
    articulation._sim_env_ids = wp.empty(N, dtype=wp.int32, device=device)
    articulation._sim_env_ids_views = {}
    articulation._cpu_env_ids_all = indices(N, "cpu")
    articulation._cpu_env_ids = wp.empty(N, dtype=wp.int32, device="cpu", pinned=wp.is_cuda_available())
    articulation._cpu_env_ids_views = {}
    for name, shape in {
        "joint_stiffness": (N, J),
        "joint_damping": (N, J),
        "joint_pos_limits": (N, J, 2),
        "joint_vel_limits": (N, J),
        "joint_effort_limits": (N, J),
        "joint_armature": (N, J),
        "joint_friction_props": (N, J, 3),
        "body_mass": (N, B),
        "body_coms": (N, B, 7),
        "body_inertia": (N, B, 9),
    }.items():
        setattr(articulation, f"_cpu_{name}", wp.zeros(shape, dtype=wp.float32, device="cpu"))
    articulation._process_actuators_cfg()
    return articulation, view


def _ovphysx_articulation(N, J, B, FT, ST, device, is_fixed_base, joint_ordering, body_ordering, monkeypatch):
    patch_ovphysx_manager(monkeypatch)
    joint_names, body_names = _names("joint", J), _names("body", B)
    bindings = MockOvPhysxBindingSet(
        num_instances=N,
        num_joints=J,
        num_bodies=B,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
        num_fixed_tendons=FT,
        num_spatial_tendons=ST,
    )
    bindings.set_random_data()

    articulation = _shell(ovphysx_assets.Articulation, joint_ordering, body_ordering)
    articulation._device = device
    articulation._ovphysx = MagicMock()
    articulation._root_view = bindings.view
    articulation._bindings = bindings.bindings
    articulation._num_instances = N
    articulation._num_joints = J
    articulation._num_bodies = B
    articulation._is_fixed_base = is_fixed_base
    articulation._joint_names = joint_names
    articulation._body_names = body_names
    articulation._num_fixed_tendons = FT
    articulation._num_spatial_tendons = ST
    # Counts come from the view; names are set on the data afterwards.
    articulation._data = ovphysx_assets.ArticulationData(bindings.view, device)
    articulation._data.body_names = body_names
    articulation._data.joint_names = joint_names
    articulation._data._is_fixed_base = is_fixed_base
    _set_tendon_names(articulation, _names("fixed_tendon", FT), _names("spatial_tendon", ST))
    articulation._resolve_and_install_ordering_maps()
    articulation._create_buffers()
    finish_shell(articulation)
    articulation._process_actuators_cfg()
    articulation._can_write_effort = articulation._get_binding(TT.DOF_ACTUATION_FORCE) is not None
    articulation._can_write_pos_target = articulation._get_binding(TT.DOF_POSITION_TARGET) is not None
    articulation._can_write_vel_target = articulation._get_binding(TT.DOF_VELOCITY_TARGET) is not None
    return articulation, bindings


def _newton_articulation(N, J, B, FT, ST, device, is_fixed_base, joint_ordering, body_ordering, monkeypatch):
    # Newton supports fixed tendons but not spatial tendons.
    fixed_tendon_names = _names("fixed_tendon", FT)
    view = MockNewtonArticulationView(
        num_instances=N,
        num_bodies=B,
        num_joints=J,
        num_tendons=FT,
        device=device,
        is_fixed_base=is_fixed_base,
        joint_names=_names("joint", J),
        body_names=_names("body", B),
        tendon_names=fixed_tendon_names,
    )
    view.set_random_mock_data()
    # MuJoCo tendon properties the data binds when the articulation has fixed tendons, with distinct values.
    tendon_values = np.arange(1, N * FT + 1, dtype=np.float32).reshape(N, 1, FT)
    view._attributes["mujoco.tendon_stiffness"] = wp.array(tendon_values, dtype=wp.float32, device=device)
    view._attributes["mujoco.tendon_damping"] = wp.array(tendon_values + 0.5, dtype=wp.float32, device=device)
    tendon_range = np.stack((-tendon_values, tendon_values), axis=-1)
    view._attributes["mujoco.tendon_range"] = wp.array(tendon_range, dtype=wp.vec2f, device=device)

    manager = newton_manager(N, device)
    # Sizes of the lazy task-space buffers; the mock model holds only this homogeneous articulation batch.
    model = view.model = manager.get_model.return_value
    total_dofs = J + (0 if is_fixed_base else 6)
    model.articulation_count = N
    model.max_joints_per_articulation = B
    model.max_dofs_per_articulation = total_dofs
    model.joint_dof_count = N * total_dofs
    model.body_count = N * B
    monkeypatch.setattr(newton_articulation_data, "SimulationManager", manager, raising=False)

    articulation = _shell(newton_assets.Articulation, joint_ordering, body_ordering)
    articulation._root_view = view
    articulation._device = device
    articulation._clamped_default_count = wp.zeros(1, dtype=wp.int32, device=device)
    articulation._data = newton_assets.ArticulationData(view, device)
    articulation._test_simulation_manager = manager
    # The solver builds this adapter; the shell has no model, so it stays absent.
    articulation._fixed_tendon_control = None
    _set_tendon_names(articulation, fixed_tendon_names, [])
    finish_shell(articulation, supports_world_at_com=False)
    articulation._ALL_INDICES = indices(N, device)
    articulation._ALL_BODY_INDICES = indices(B, device)
    articulation._ALL_JOINT_INDICES = indices(J, device)
    articulation._ALL_ENV_MASK = wp.ones((N,), dtype=wp.bool, device=device)
    articulation._ALL_BODY_MASK = wp.ones((B,), dtype=wp.bool, device=device)
    articulation._ALL_JOINT_MASK = wp.ones((J,), dtype=wp.bool, device=device)
    articulation._resolve_and_install_ordering_maps()
    articulation._ordering_configure_backend_staging()
    articulation._ALL_FIXED_TENDON_INDICES = indices(FT, device)
    articulation._ALL_FIXED_TENDON_MASK = wp.ones((FT,), dtype=wp.bool, device=device)
    articulation._ALL_SPATIAL_TENDON_INDICES = indices(0, device)
    articulation._ALL_SPATIAL_TENDON_MASK = wp.ones((0,), dtype=wp.bool, device=device)
    articulation._process_actuators_cfg()
    return articulation, view


def get_articulation(
    backend: str,
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
    *,
    monkeypatch: pytest.MonkeyPatch,
):
    """Create a mocked articulation of one backend and return it with its backend view or bindings."""
    factory = {"physx": _physx_articulation, "ovphysx": _ovphysx_articulation, "newton": _newton_articulation}[backend]
    counts = (num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons)
    return factory(*counts, device, is_fixed_base, joint_ordering, body_ordering, monkeypatch)
