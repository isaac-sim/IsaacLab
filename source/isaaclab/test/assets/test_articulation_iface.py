# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""
Checks that the articulation interfaces are consistent across backends, and are providing the exact same data as what
the base articulation class advertises. All articulation interfaces need to comply with the same interface contract.

The setup is a bit convoluted so that we can run these tests without requiring Isaac Sim or GPU simulation.
"""

"""Launch Isaac Sim Simulator first (when available)."""

import os
import sys
from unittest.mock import MagicMock

# When running kitless (e.g., ovphysx backend via run_ovphysx.sh), AppLauncher
# will try to boot Kit and hang. Skip it entirely: run_ovphysx.sh sets
# LD_PRELOAD to the ovphysx libcarb.so, which is the signature of a kitless
# ovphysx run. Also guard the case where neither LD_PRELOAD nor EXP_PATH is
# set (bare Python, no Kit at all).
_kitless = "ovphysx" in os.environ.get("LD_PRELOAD", "") or (
    os.environ.get("LD_PRELOAD", "") == "" and "EXP_PATH" not in os.environ
)

if not _kitless:
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
else:
    simulation_app = None
    # Stub out the Kit/Omniverse modules that are not present under
    # run_ovphysx.sh (pxr, carb, omni, omni.kit[.app] are real on PYTHONPATH).
    # ``omni`` is a real namespace package, so missing submodules also need
    # to be installed as attributes on it -- ``sys.modules`` alone is not
    # enough because attribute access on the real ``omni`` won't fall
    # through to ``sys.modules``.
    import omni as _omni

    for _mod in ("physics", "physics.tensors", "physx", "timeline", "usd"):
        _stub = MagicMock()
        sys.modules[f"omni.{_mod}"] = _stub
        # Bind the leaf attribute so that ``omni.<leaf>`` resolves.
        setattr(_omni, _mod.split(".", 1)[0], _stub)
    for _mod in ("isaacsim.core", "isaacsim.core.simulation_manager"):
        sys.modules.setdefault(_mod, MagicMock())

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer
from isaaclab.utils.buffers import TimestampedBufferWarp
from isaaclab.utils.wrench_composer import WrenchComposer

# Mock SimulationManager.get_physics_sim_view() to return a mock object with gravity.
# This is needed because the PhysX Data classes call
# SimulationManager.get_physics_sim_view().get_gravity() but there's no actual
# physics scene when running unit tests.
_mock_physics_sim_view = MagicMock()
_mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)

from isaaclab_physx.physics import PhysxManager as SimulationManager

SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)

"""
Check which backends are available.
"""

BACKENDS = ["mock"]  # Mock backend is always available.

try:
    from isaaclab_physx.assets.articulation.articulation import Articulation as PhysXArticulation
    from isaaclab_physx.assets.articulation.articulation_data import ArticulationData as PhysXArticulationData
    from isaaclab_physx.test.mock_interfaces.views import MockArticulationViewWarp as PhysXMockArticulationViewWarp

    BACKENDS.append("physx")
except ImportError:
    pass

try:
    from isaaclab_newton.assets.articulation.articulation import Articulation as NewtonArticulation
    from isaaclab_newton.assets.articulation.articulation_data import ArticulationData as NewtonArticulationData
    from isaaclab_newton.test.mock_interfaces.views import MockNewtonArticulationView as NewtonMockArticulationView

    BACKENDS.append("newton")
except ImportError:
    pass

try:
    from isaaclab_ovphysx.assets.articulation.articulation import Articulation as OvPhysxArticulation
    from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData as OvPhysxArticulationData
    from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet

    BACKENDS.append("ovphysx")
except ImportError:
    pass


def create_physx_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create a test Articulation instance with mocked dependencies."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    fixed_tendon_names = [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]
    spatial_tendon_names = [f"spatial_tendon_{i}" for i in range(num_spatial_tendons)]

    articulation = object.__new__(PhysXArticulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
    )

    # Create PhysX mock view
    mock_view = PhysXMockArticulationViewWarp(
        count=num_instances,
        num_links=num_bodies,
        num_dofs=num_joints,
        device=device,
        max_fixed_tendons=num_fixed_tendons,
        max_spatial_tendons=num_spatial_tendons,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Set up the mock view's metatype for accessing names/counts
    mock_metatype = MagicMock()
    mock_metatype.fixed_base = is_fixed_base
    mock_metatype.dof_count = num_joints
    mock_metatype.link_count = num_bodies
    mock_metatype.dof_names = joint_names
    mock_metatype.link_names = body_names
    object.__setattr__(mock_view, "_shared_metatype", mock_metatype)

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)

    # We can't call the initialize method here, because we don't have a good mock for the actuators yet.
    # We need to set the _data attribute manually.

    # Create ArticulationData instance (SimulationManager already mocked at module level)
    data = PhysXArticulationData(mock_view, device)
    object.__setattr__(articulation, "_data", data)

    # Set tendon names on articulation and data
    object.__setattr__(articulation, "_fixed_tendon_names", fixed_tendon_names)
    object.__setattr__(articulation, "_spatial_tendon_names", spatial_tendon_names)
    data.fixed_tendon_names = fixed_tendon_names
    data.spatial_tendon_names = spatial_tendon_names

    # Create mock wrench composers (pass articulation which has num_instances, num_bodies, device properties)
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)

    # Set up other required attributes
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)
    object.__setattr__(articulation, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(
        articulation, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )
    object.__setattr__(
        articulation, "_ALL_JOINT_INDICES", wp.array(np.arange(num_joints, dtype=np.int32), device=device)
    )

    # Tendon index arrays
    object.__setattr__(
        articulation,
        "_ALL_FIXED_TENDON_INDICES",
        wp.array(np.arange(num_fixed_tendons, dtype=np.int32), device=device),
    )
    object.__setattr__(
        articulation,
        "_ALL_SPATIAL_TENDON_INDICES",
        wp.array(np.arange(num_spatial_tendons, dtype=np.int32), device=device),
    )

    # Warp arrays for set_external_force_and_torque
    object.__setattr__(
        articulation, "_ALL_INDICES_WP", wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    )
    object.__setattr__(
        articulation, "_ALL_BODY_INDICES_WP", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )

    object.__setattr__(articulation, "_joint_pos_target_backend", None)
    object.__setattr__(articulation, "_joint_vel_target_backend", None)
    object.__setattr__(articulation, "_joint_effort_target_backend", None)
    object.__setattr__(articulation, "_body_wrench_force_backend", None)
    object.__setattr__(articulation, "_body_wrench_torque_backend", None)

    articulation._resolve_and_install_ordering_maps()
    articulation._cache_ordering_maps()

    # Initialize joint targets
    joint_target_shape = (num_instances, num_joints)
    object.__setattr__(
        articulation, "_joint_pos_target_sim", wp.zeros(joint_target_shape, dtype=wp.float32, device=device)
    )
    object.__setattr__(
        articulation, "_joint_vel_target_sim", wp.zeros(joint_target_shape, dtype=wp.float32, device=device)
    )
    object.__setattr__(
        articulation, "_joint_effort_target_sim", wp.zeros(joint_target_shape, dtype=wp.float32, device=device)
    )

    # Cached .view(wp.float32) wrappers
    object.__setattr__(articulation, "_root_link_pose_w_f32", None)
    object.__setattr__(articulation, "_root_com_vel_w_f32", None)
    object.__setattr__(articulation, "_root_link_vel_w_f32", None)
    object.__setattr__(articulation, "_inst_wrench_force_f32", None)
    object.__setattr__(articulation, "_inst_wrench_torque_f32", None)
    object.__setattr__(articulation, "_perm_wrench_force_f32", None)
    object.__setattr__(articulation, "_perm_wrench_torque_f32", None)

    # Pre-allocated pinned CPU buffers for PhysX TensorAPI writes
    N, J, B = num_instances, num_joints, num_bodies
    cpu_env_ids = wp.array(np.arange(N, dtype=np.int32), device="cpu")
    object.__setattr__(articulation, "_cpu_env_ids_all", cpu_env_ids)
    object.__setattr__(articulation, "_cpu_joint_stiffness", wp.zeros((N, J), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_joint_damping", wp.zeros((N, J), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_joint_pos_limits", wp.zeros((N, J, 2), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_joint_vel_limits", wp.zeros((N, J), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_joint_effort_limits", wp.zeros((N, J), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_joint_armature", wp.zeros((N, J), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_joint_friction_props", wp.zeros((N, J, 3), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_body_mass", wp.zeros((N, B), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_body_coms", wp.zeros((N, B, 7), dtype=wp.float32, device="cpu"))
    object.__setattr__(articulation, "_cpu_body_inertia", wp.zeros((N, B, 9), dtype=wp.float32, device="cpu"))

    return articulation, mock_view


def create_ovphysx_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create a test OvPhysX Articulation instance with mocked tensor bindings."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(OvPhysxArticulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
    )

    # Create mock binding set
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
    mock_bindings.set_random_data()

    fixed_tendon_names = [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]
    spatial_tendon_names = [f"spatial_tendon_{i}" for i in range(num_spatial_tendons)]

    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_ovphysx", MagicMock())
    object.__setattr__(articulation, "_root_view", mock_bindings.view)
    object.__setattr__(articulation, "_bindings", mock_bindings.bindings)
    object.__setattr__(articulation, "_num_instances", num_instances)
    object.__setattr__(articulation, "_num_joints", num_joints)
    object.__setattr__(articulation, "_num_bodies", num_bodies)
    object.__setattr__(articulation, "_is_fixed_base", is_fixed_base)
    object.__setattr__(articulation, "_joint_names", joint_names)
    object.__setattr__(articulation, "_body_names", body_names)
    object.__setattr__(articulation, "_fixed_tendon_names", fixed_tendon_names)
    object.__setattr__(articulation, "_spatial_tendon_names", spatial_tendon_names)
    object.__setattr__(articulation, "_num_fixed_tendons", num_fixed_tendons)
    object.__setattr__(articulation, "_num_spatial_tendons", num_spatial_tendons)

    # Create ArticulationData; counts come from the view, names are set after.
    data = OvPhysxArticulationData(mock_bindings.view, device)
    data.body_names = body_names
    data.joint_names = joint_names
    data.fixed_tendon_names = fixed_tendon_names
    data.spatial_tendon_names = spatial_tendon_names
    data._is_fixed_base = is_fixed_base
    object.__setattr__(articulation, "_data", data)

    # Allocate the articulation-side index/mask caches and wrench buffer that
    # _initialize_impl would normally populate.  Wrench composers created here
    # are immediately overwritten by the mocks below.
    articulation._resolve_and_install_ordering_maps()
    articulation._create_buffers()

    # Wrench composers
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)
    object.__setattr__(articulation, "_effort_write_view", None)
    object.__setattr__(articulation, "_pos_target_write_view", None)
    object.__setattr__(articulation, "_vel_target_write_view", None)

    # Prevent __del__ / _clear_callbacks from raising
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)

    return articulation, mock_bindings


def create_newton_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create a test Newton Articulation instance with mocked dependencies."""
    import isaaclab_newton.assets.articulation.articulation_data as newton_data_module

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    # Create Newton mock view
    mock_view = NewtonMockArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Mock NewtonManager (aliased as SimulationManager in Newton modules)
    mock_model = MagicMock()
    mock_model.gravity = wp.array(np.array([[0.0, 0.0, -9.81]], dtype=np.float32), dtype=wp.vec3f, device=device)
    # Sizes consumed by the task-space scratch buffers in NewtonArticulationData.__init__.
    # Model-wide counts equal the per-articulation counts here because the mock contains a
    # single homogeneous world.
    mock_model.articulation_count = num_instances
    mock_model.max_joints_per_articulation = num_bodies
    total_dofs = num_joints + (0 if is_fixed_base else 6)
    mock_model.max_dofs_per_articulation = total_dofs
    mock_model.joint_dof_count = num_instances * total_dofs
    mock_model.body_count = num_instances * num_bodies
    mock_state = MagicMock()
    mock_control = MagicMock()

    mock_manager = MagicMock()
    mock_manager.get_model.return_value = mock_model
    mock_manager.get_state_0.return_value = mock_state
    mock_manager.get_state_1.return_value = mock_state
    mock_manager.get_control.return_value = mock_control

    # Patch SimulationManager in the Newton data module
    original_sim_manager = newton_data_module.SimulationManager
    newton_data_module.SimulationManager = mock_manager

    try:
        data = NewtonArticulationData(mock_view, device)
    finally:
        newton_data_module.SimulationManager = original_sim_manager

    # Create Articulation shell (bypass __init__)
    articulation = object.__new__(NewtonArticulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
    )

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_data", data)
    object.__setattr__(articulation, "_test_simulation_manager", mock_manager)

    # Tendon names (Newton doesn't support tendons)
    object.__setattr__(articulation, "_fixed_tendon_names", [])
    object.__setattr__(articulation, "_spatial_tendon_names", [])
    data.fixed_tendon_names = []
    data.spatial_tendon_names = []

    # Mock wrench composers
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)

    # Other required attributes
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)

    # Newton uses wp.array for indices (not torch)
    object.__setattr__(articulation, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(
        articulation, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )
    object.__setattr__(
        articulation, "_ALL_JOINT_INDICES", wp.array(np.arange(num_joints, dtype=np.int32), device=device)
    )

    # Newton uses wp.bool masks
    object.__setattr__(articulation, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_JOINT_MASK", wp.ones((num_joints,), dtype=wp.bool, device=device))

    articulation._resolve_and_install_ordering_maps()
    articulation._cache_ordering_maps()

    # Tendon arrays (empty)
    object.__setattr__(articulation, "_ALL_FIXED_TENDON_INDICES", wp.array(np.array([], dtype=np.int32), device=device))
    object.__setattr__(articulation, "_ALL_FIXED_TENDON_MASK", wp.ones((0,), dtype=wp.bool, device=device))
    object.__setattr__(
        articulation, "_ALL_SPATIAL_TENDON_INDICES", wp.array(np.array([], dtype=np.int32), device=device)
    )
    object.__setattr__(articulation, "_ALL_SPATIAL_TENDON_MASK", wp.ones((0,), dtype=wp.bool, device=device))

    # Joint targets (Newton uses warp, not torch)
    object.__setattr__(
        articulation,
        "_joint_pos_target_sim",
        wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device),
    )
    object.__setattr__(
        articulation,
        "_joint_vel_target_sim",
        wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device),
    )
    object.__setattr__(
        articulation,
        "_joint_effort_target_sim",
        wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device),
    )

    return articulation, mock_view


def create_mock_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
):
    from isaaclab.test.mock_interfaces.assets.mock_articulation import MockArticulation

    art = MockArticulation(
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=num_bodies,
        num_fixed_tendons=num_fixed_tendons,
        num_spatial_tendons=num_spatial_tendons,
        device=device,
    )
    return art, None  # No view for mock backend


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
):
    if backend == "physx":
        return create_physx_articulation(
            num_instances,
            num_joints,
            num_bodies,
            num_fixed_tendons,
            num_spatial_tendons,
            device,
            is_fixed_base=is_fixed_base,
            joint_ordering=joint_ordering,
            body_ordering=body_ordering,
        )
    elif backend == "ovphysx":
        return create_ovphysx_articulation(
            num_instances,
            num_joints,
            num_bodies,
            num_fixed_tendons,
            num_spatial_tendons,
            device,
            is_fixed_base=is_fixed_base,
            joint_ordering=joint_ordering,
            body_ordering=body_ordering,
        )
    elif backend == "newton":
        return create_newton_articulation(
            num_instances,
            num_joints,
            num_bodies,
            device,
            is_fixed_base=is_fixed_base,
            joint_ordering=joint_ordering,
            body_ordering=body_ordering,
        )
    elif backend.lower() == "mock":
        return create_mock_articulation(
            num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


@pytest.fixture
def articulation_iface(request):
    backend = request.getfixturevalue("backend")
    num_instances = request.getfixturevalue("num_instances")
    num_joints = request.getfixturevalue("num_joints")
    num_bodies = request.getfixturevalue("num_bodies")
    device = request.getfixturevalue("device")
    try:
        num_fixed_tendons = request.getfixturevalue("num_fixed_tendons")
    except pytest.FixtureLookupError:
        num_fixed_tendons = 0
    try:
        num_spatial_tendons = request.getfixturevalue("num_spatial_tendons")
    except pytest.FixtureLookupError:
        num_spatial_tendons = 0
    return get_articulation(
        backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_proxy_array(arr, *, expected_shape: tuple, expected_dtype: type, name: str):
    """Assert that `arr` is a ProxyArray with the expected shape and dtype."""
    from isaaclab.utils.warp import ProxyArray

    assert isinstance(arr, ProxyArray), f"{name}: expected ProxyArray, got {type(arr)}"
    assert arr.shape == expected_shape, f"{name}: expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == expected_dtype, f"{name}: expected dtype {expected_dtype}, got {arr.dtype}"


def _make_body_ordering_backend_data(num_instances: int, num_bodies: int) -> tuple[np.ndarray, ...]:
    """Create deterministic backend-order body data with identity rotations."""
    root_pose = np.zeros((num_instances, 7), dtype=np.float32)
    root_pose[:, 6] = 1.0
    root_vel = np.zeros((num_instances, 6), dtype=np.float32)
    link_pose = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    com_pose_b = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    body_com_vel = np.zeros((num_instances, num_bodies, 6), dtype=np.float32)
    body_acc = np.zeros((num_instances, num_bodies, 6), dtype=np.float32)

    for env_index in range(num_instances):
        root_pose[env_index, :3] = (10.0 + env_index, 0.0, 0.0)
        root_vel[env_index, 0] = 3.0 + env_index
        root_vel[env_index, 5] = 7.0 + env_index
        for body_index in range(num_bodies):
            link_pose[env_index, body_index, :3] = (10.0 * body_index, float(env_index), 0.0)
            link_pose[env_index, body_index, 6] = 1.0
            com_pose_b[env_index, body_index, :3] = (float(body_index + 1), 0.0, 0.0)
            com_pose_b[env_index, body_index, 6] = 1.0
            body_com_vel[env_index, body_index, 0] = 20.0 + body_index
            body_com_vel[env_index, body_index, 5] = 30.0 + body_index
            body_acc[env_index, body_index, 0] = 100.0 + body_index
            body_acc[env_index, body_index, 3] = 200.0 + body_index

    return root_pose, root_vel, link_pose, com_pose_b, body_com_vel, body_acc


def _install_test_body_ordering(art) -> np.ndarray:
    """Install a non-identity body ordering that preserves a fixed root body."""
    if art.is_fixed_base:
        body_ordering = (art.backend_body_names[0], *reversed(art.backend_body_names[1:]))
    else:
        body_ordering = tuple(reversed(art.backend_body_names))
    art.cfg = art.cfg.replace(body_ordering=body_ordering)
    art._resolve_and_install_ordering_maps()
    art._cache_ordering_maps()
    return np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)


def _install_reversed_joint_ordering(art) -> np.ndarray:
    """Install a reversed public joint ordering on an already constructed articulation."""
    art.cfg = art.cfg.replace(joint_ordering=tuple(reversed(art.backend_joint_names)))
    art._resolve_and_install_ordering_maps()
    art._cache_ordering_maps()
    return np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)


def _joint_ordering_for_mode(mode: str, num_joints: int) -> tuple[str, ...] | None:
    """Return the configured public joint ordering for a parity-test mode."""
    backend_names = tuple(f"joint_{index}" for index in range(num_joints))
    if mode == "none":
        return None
    if mode == "identity":
        return backend_names
    if mode == "reversed":
        return tuple(reversed(backend_names))
    raise ValueError(f"Unsupported joint ordering mode: {mode}")


def _body_ordering_for_mode(mode: str, num_bodies: int) -> tuple[str, ...] | None:
    """Return the configured public body ordering for a parity-test mode."""
    backend_names = tuple(f"body_{index}" for index in range(num_bodies))
    if mode == "none":
        return None
    if mode == "identity":
        return backend_names
    if mode == "reversed":
        return tuple(reversed(backend_names))
    raise ValueError(f"Unsupported body ordering mode: {mode}")


def _ordering_shadow_shape(buffer: wp.array | TimestampedBufferWarp) -> tuple[int, ...]:
    """Return the allocation shape for a raw or timestamped ordering shadow."""
    if isinstance(buffer, TimestampedBufferWarp):
        buffer = buffer.data
    return tuple(buffer.shape)


_ORDERING_UNCHANGED = object()


def _transition_articulation_ordering(
    art,
    *,
    joint_ordering=_ORDERING_UNCHANGED,
    body_ordering=_ORDERING_UNCHANGED,
) -> None:
    """Install selected ordering maps through the production rebind path."""
    cfg_updates = {}
    if joint_ordering is not _ORDERING_UNCHANGED:
        cfg_updates["joint_ordering"] = joint_ordering
    if body_ordering is not _ORDERING_UNCHANGED:
        cfg_updates["body_ordering"] = body_ordering
    art.cfg = art.cfg.replace(**cfg_updates)
    art._resolve_and_install_ordering_maps()
    art._cache_ordering_maps()


def _ordering_user_to_backend(ordering, item_count: int) -> np.ndarray:
    """Return public-to-backend indices for an optional ordering map."""
    if ordering is None:
        return np.arange(item_count, dtype=np.int64)
    return np.asarray(ordering.user_to_backend_indices, dtype=np.int64)


def _seed_public_joint_commands(backend: str, art, backend_targets: tuple[np.ndarray, ...]) -> None:
    """Seed public and backend-writer joint command buffers in backend order."""
    public_buffers = (
        art.data._joint_pos_target,
        art.data._joint_vel_target,
        art.data._joint_effort_target,
    )
    for buffer, values in zip(public_buffers, backend_targets, strict=True):
        buffer.assign(wp.array(values, dtype=wp.float32, device=art.device))

    art._has_implicit_actuators = True
    if backend in ("physx", "newton"):
        for buffer, values in zip(
            (
                art._joint_pos_target_sim,
                art._joint_vel_target_sim,
                art._joint_effort_target_sim,
            ),
            backend_targets,
            strict=True,
        ):
            buffer.assign(wp.array(values, dtype=wp.float32, device=art.device))
    elif backend == "ovphysx":
        art.data._applied_torque.assign(wp.array(backend_targets[2], dtype=wp.float32, device=art.device))
        art._pos_target_write_view = object()
        art._vel_target_write_view = object()
        art._effort_write_view = object()
    else:
        raise AssertionError(f"Unsupported backend for command lifecycle test: {backend}")


def _write_joint_commands_to_backend(backend: str, art, raw_backend) -> tuple[np.ndarray, ...]:
    """Write commands and return backend-order position, velocity, and effort targets."""
    if backend == "physx":
        captured = {}

        def capture_position(values, indices):
            captured["position"] = values.numpy().copy()

        def capture_velocity(values, indices):
            captured["velocity"] = values.numpy().copy()

        def capture_effort(values, indices):
            captured["effort"] = values.numpy().copy()

        raw_backend.set_dof_position_targets = capture_position
        raw_backend.set_dof_velocity_targets = capture_velocity
        raw_backend.set_dof_actuation_forces = capture_effort
        art.write_data_to_sim()
        return captured["position"], captured["velocity"], captured["effort"]

    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        art.write_data_to_sim()
        return (
            raw_backend.bindings[TT.DOF_POSITION_TARGET]._data.copy(),
            raw_backend.bindings[TT.DOF_VELOCITY_TARGET]._data.copy(),
            raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data.copy(),
        )

    if backend == "newton":
        art.write_data_to_sim()
        return (
            art.data._sim_bind_joint_position_target.numpy().copy(),
            art.data._sim_bind_joint_velocity_target.numpy().copy(),
            art.data._sim_bind_joint_effort.numpy().copy(),
        )

    raise AssertionError(f"Unsupported backend for command lifecycle test: {backend}")


def _assert_public_joint_commands(backend: str, art, backend_targets: tuple[np.ndarray, ...]) -> None:
    """Assert public targets and active backend shadows represent the same physical commands."""
    user_to_backend = _ordering_user_to_backend(art.joint_ordering, art.num_joints)
    public_buffers = (
        art.data._joint_pos_target,
        art.data._joint_vel_target,
        art.data._joint_effort_target,
    )
    backend_fields = (
        "_joint_pos_target_backend",
        "_joint_vel_target_backend",
        "_joint_effort_target_backend",
    )
    for public_buffer, backend_field, backend_values in zip(
        public_buffers, backend_fields, backend_targets, strict=True
    ):
        np.testing.assert_array_equal(public_buffer.numpy(), backend_values[:, user_to_backend])
        if backend == "newton":
            continue
        backend_buffer = getattr(art, backend_field)
        if art._has_joint_ordering:
            assert backend_buffer is not None
            np.testing.assert_array_equal(backend_buffer.numpy(), backend_values)
        else:
            assert backend_buffer is None


def _assert_public_ordered_properties(
    art,
    backend_joint_stiffness: np.ndarray,
    backend_body_mass: np.ndarray,
) -> None:
    """Assert representative joint/body properties retain their backend physical identity."""
    joint_user_to_backend = _ordering_user_to_backend(art.joint_ordering, art.num_joints)
    body_user_to_backend = _ordering_user_to_backend(art.body_ordering, art.num_bodies)
    np.testing.assert_array_equal(
        art.data.joint_stiffness.warp.numpy(),
        backend_joint_stiffness[:, joint_user_to_backend],
    )
    np.testing.assert_array_equal(
        art.data.body_mass.warp.numpy(),
        backend_body_mass[:, body_user_to_backend],
    )


def _make_backend_com_poses(num_instances: int, num_bodies: int) -> np.ndarray:
    """Create distinct deterministic body COM transforms in backend order."""
    coms = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    for env_index in range(num_instances):
        for body_index in range(num_bodies):
            value = float(100 * env_index + 10 * body_index)
            coms[env_index, body_index, :3] = (value + 1.0, value + 2.0, value + 3.0)
            coms[env_index, body_index, 6] = 1.0
    return coms


def _seed_backend_com_poses(backend: str, art, raw_backend, coms: np.ndarray) -> None:
    """Seed backend-order COM transforms and invalidate both public and ordered caches."""
    if backend == "physx":
        raw_backend._coms = wp.array(coms, dtype=wp.float32, device="cpu")
    elif backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.BODY_COM_POSE]._data = coms.copy()
    else:
        raise AssertionError(f"Unsupported backend for COM parity test: {backend}")

    art.data._body_com_pose_b.timestamp = -1.0
    backend_staging = art.data._body_com_pose_b_backend
    if backend_staging is not None:
        backend_staging.timestamp = -1.0


def _read_backend_com_poses(backend: str, art, raw_backend) -> np.ndarray:
    """Read complete backend-order body COM transforms through the backend API."""
    if backend == "physx":
        return raw_backend.get_coms().numpy().reshape(art.num_instances, art.num_bodies, 7).copy()
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        return (
            art.root_view.get_attribute(TT.BODY_COM_POSE).numpy().reshape(art.num_instances, art.num_bodies, 7).copy()
        )
    raise AssertionError(f"Unsupported backend for COM parity test: {backend}")


def _make_selected_com_poses(env_ids: list[int], body_ids: list[int], base_value: float) -> np.ndarray:
    """Create distinct deterministic public-order COM transforms for a selected rectangle."""
    coms = np.zeros((len(env_ids), len(body_ids), 7), dtype=np.float32)
    for env_offset, env_id in enumerate(env_ids):
        for body_offset, body_id in enumerate(body_ids):
            value = base_value + float(100 * env_id + 10 * body_id)
            coms[env_offset, body_offset, :3] = (value + 1.0, value + 2.0, value + 3.0)
            coms[env_offset, body_offset, 6] = 1.0
    return coms


def _seed_backend_joint_state(
    backend: str,
    art,
    raw_backend,
    position: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """Seed deterministic joint state directly in backend-order storage."""
    if backend == "physx":
        raw_backend._noop_setters = False
        raw_backend._dof_positions = wp.array(position, dtype=wp.float32, device=art.device)
        raw_backend._dof_velocities = wp.array(velocity, dtype=wp.float32, device=art.device)

        def set_backend_rows(storage: wp.array, values: wp.array, indices: wp.array | None) -> None:
            if indices is None:
                storage.assign(values)
                return
            env_ids = indices.numpy()
            # PhysX writers pass global staging rows; the generic mock setter expects compact rows.
            source_ids = env_ids if values.shape[0] == art.num_instances else np.arange(indices.shape[0])
            storage.numpy()[env_ids] = values.numpy()[source_ids]

        def set_dof_positions(values: wp.array, indices: wp.array | None = None) -> None:
            set_backend_rows(raw_backend._dof_positions, values, indices)

        def set_dof_velocities(values: wp.array, indices: wp.array | None = None) -> None:
            set_backend_rows(raw_backend._dof_velocities, values, indices)

        raw_backend.set_dof_positions = set_dof_positions
        raw_backend.set_dof_velocities = set_dof_velocities
        art.data._joint_pos.timestamp = -1.0
        art.data._joint_vel.timestamp = -1.0
        return
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.DOF_POSITION]._data = position.copy()
        raw_backend.bindings[TT.DOF_VELOCITY]._data = velocity.copy()
        for buffer_name in ("_joint_pos_buf", "_joint_vel_buf", "_joint_pos_backend", "_joint_vel_backend"):
            buffer = getattr(art.data, buffer_name, None)
            if buffer is not None:
                buffer.timestamp = -1.0
        return
    if backend == "newton":
        art.data._sim_bind_joint_pos.assign(wp.array(position, dtype=wp.float32, device=art.device))
        art.data._sim_bind_joint_vel.assign(wp.array(velocity, dtype=wp.float32, device=art.device))
        return
    raise AssertionError(f"Unsupported backend for joint-state parity test: {backend}")


def _mutate_backend_joint_state(
    backend: str,
    art,
    raw_backend,
    position: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """Mutate backend-order joint state without invalidating staging timestamps."""
    if backend == "physx":
        raw_backend._dof_positions.assign(wp.array(position, dtype=wp.float32, device=art.device))
        raw_backend._dof_velocities.assign(wp.array(velocity, dtype=wp.float32, device=art.device))
        return
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.DOF_POSITION]._data = position.copy()
        raw_backend.bindings[TT.DOF_VELOCITY]._data = velocity.copy()
        return
    raise AssertionError(f"Unsupported backend for stale joint-state parity test: {backend}")


def _read_backend_joint_state(backend: str, art, raw_backend) -> tuple[np.ndarray, np.ndarray]:
    """Return joint position and velocity from backend-order storage."""
    if backend == "physx":
        return raw_backend._dof_positions.numpy().copy(), raw_backend._dof_velocities.numpy().copy()
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        position = np.asarray(raw_backend.bindings[TT.DOF_POSITION]._data).copy()
        velocity = np.asarray(raw_backend.bindings[TT.DOF_VELOCITY]._data).copy()
        return position, velocity
    if backend == "newton":
        return art.data._sim_bind_joint_pos.numpy().copy(), art.data._sim_bind_joint_vel.numpy().copy()
    raise AssertionError(f"Unsupported backend for joint-state parity test: {backend}")


def _coverage_ids(coverage: str, num_instances: int, num_items: int) -> tuple[list[int], list[int]]:
    """Return environment and item IDs for a write-coverage mode."""
    if coverage == "one_env_one_item":
        return [1], [0]
    if coverage == "all_envs_one_item":
        return list(range(num_instances)), [0]
    if coverage == "one_env_all_items":
        return [1], list(range(num_items))
    raise ValueError(f"Unsupported write coverage: {coverage}")


def _write_selected_joint_state(
    art,
    backend: str,
    selection: str,
    operation: str,
    coverage: str,
    position_value: float,
    velocity_value: float,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Write selected public joint state through an index or mask API."""
    if operation not in ("position", "velocity", "state"):
        raise ValueError(f"Unsupported joint-state operation: {operation}")

    env_ids, joint_ids = _coverage_ids(coverage, art.num_instances, art.num_joints)
    payload_offsets = 10.0 * np.asarray(env_ids, dtype=np.float32)[:, None]
    payload_offsets = payload_offsets + np.asarray(joint_ids, dtype=np.float32)[None, :]
    position_payload = np.asarray(position_value + payload_offsets, dtype=np.float32)
    velocity_payload = np.asarray(velocity_value + payload_offsets, dtype=np.float32)
    if selection == "index":
        env_ids_wp = wp.array(env_ids, dtype=wp.int32, device=art.device)
        joint_ids_wp = wp.array(joint_ids, dtype=wp.int32, device=art.device)
        position = torch.tensor(position_payload, dtype=torch.float32, device=art.device)
        velocity = torch.tensor(velocity_payload, dtype=torch.float32, device=art.device)
        if operation == "position":
            art.write_joint_position_to_sim_index(
                position=position,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
        elif operation == "velocity":
            art.write_joint_velocity_to_sim_index(
                velocity=velocity,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
        elif backend == "ovphysx":
            with pytest.deprecated_call():
                art.write_joint_state_to_sim(
                    position=position,
                    velocity=velocity,
                    env_ids=env_ids_wp,
                    joint_ids=joint_ids_wp,
                )
        else:
            art.write_joint_state_to_sim_index(
                position=position,
                velocity=velocity,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
        return env_ids, joint_ids, position_payload, velocity_payload

    if selection == "mask":
        env_mask = wp.array(
            [env_index in env_ids for env_index in range(art.num_instances)],
            dtype=wp.bool,
            device=art.device,
        )
        joint_mask = wp.array(
            [joint_index in joint_ids for joint_index in range(art.num_joints)],
            dtype=wp.bool,
            device=art.device,
        )
        position = torch.zeros((art.num_instances, art.num_joints), dtype=torch.float32, device=art.device)
        velocity = torch.zeros((art.num_instances, art.num_joints), dtype=torch.float32, device=art.device)
        env_index = torch.tensor(env_ids, dtype=torch.long, device=art.device)
        joint_index = torch.tensor(joint_ids, dtype=torch.long, device=art.device)
        position[env_index[:, None], joint_index] = torch.tensor(
            position_payload, dtype=torch.float32, device=art.device
        )
        velocity[env_index[:, None], joint_index] = torch.tensor(
            velocity_payload, dtype=torch.float32, device=art.device
        )
        if operation == "position":
            art.write_joint_position_to_sim_mask(
                position=position,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )
        elif operation == "velocity":
            art.write_joint_velocity_to_sim_mask(
                velocity=velocity,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )
        else:
            art.write_joint_state_to_sim_mask(
                position=position,
                velocity=velocity,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )
        return env_ids, joint_ids, position_payload, velocity_payload

    raise ValueError(f"Unsupported joint selection mode: {selection}")


def _exercise_partial_joint_write(
    backend: str,
    ordering_mode: str,
    selection: str,
    operation: str,
    coverage: str,
) -> None:
    """Assert partial public writes preserve complete backend joint rows."""
    num_instances = 2
    num_joints = 3
    joint_ordering = _joint_ordering_for_mode(ordering_mode, num_joints)
    art, raw_backend = get_articulation(
        backend,
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=2,
        device="cpu",
        joint_ordering=joint_ordering,
    )
    initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
    _seed_backend_joint_state(backend, art, raw_backend, initial_position, initial_velocity)

    if art.joint_ordering is None:
        user_to_backend = np.arange(num_joints, dtype=np.int64)
    else:
        user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
    expected_position = initial_position.copy()
    expected_velocity = initial_velocity.copy()

    for position_value, velocity_value in ((901.0, 902.0), (903.0, 904.0)):
        env_ids, joint_ids, position_payload, velocity_payload = _write_selected_joint_state(
            art,
            backend,
            selection,
            operation,
            coverage,
            position_value,
            velocity_value,
        )
        backend_joint_ids = user_to_backend[joint_ids]
        selected_cells = np.ix_(env_ids, backend_joint_ids)
        if operation in ("position", "state"):
            expected_position[selected_cells] = position_payload
        if operation in ("velocity", "state"):
            expected_velocity[selected_cells] = velocity_payload

        backend_position, backend_velocity = _read_backend_joint_state(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_position, expected_position)
        np.testing.assert_array_equal(backend_velocity, expected_velocity)

    public_position = art.data.joint_pos.torch.detach().cpu().numpy()
    public_velocity = art.data.joint_vel.torch.detach().cpu().numpy()
    np.testing.assert_array_equal(public_position, expected_position[:, user_to_backend])
    np.testing.assert_array_equal(public_velocity, expected_velocity[:, user_to_backend])


def _exercise_stale_partial_joint_write(backend: str, operation: str) -> None:
    """Assert partial writes refresh complete rows after the simulation timestamp advances."""
    num_instances = 2
    num_joints = 3
    art, raw_backend = get_articulation(
        backend,
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=2,
        device="cpu",
        joint_ordering=_joint_ordering_for_mode("reversed", num_joints),
    )
    initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
    _seed_backend_joint_state(backend, art, raw_backend, initial_position, initial_velocity)
    user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)

    expected_position = initial_position.copy()
    expected_velocity = initial_velocity.copy()
    public_position_staging = art.data._joint_pos if backend == "physx" else art.data._joint_pos_buf
    public_velocity_staging = art.data._joint_vel if backend == "physx" else art.data._joint_vel_buf
    staging_buffers = []
    if operation in ("position", "state"):
        _ = art.data.joint_pos
        staging_buffers.extend(
            (("public position", public_position_staging), ("backend position", art.data._joint_pos_backend))
        )
    if operation in ("velocity", "state"):
        _ = art.data.joint_vel
        staging_buffers.extend(
            (("public velocity", public_velocity_staging), ("backend velocity", art.data._joint_vel_backend))
        )
    for staging_name, staging_buffer in staging_buffers:
        assert staging_buffer is not None
        assert staging_buffer.timestamp == art.data._sim_timestamp, f"{backend} {staging_name} staging is not current"

    for write_index, (position_value, velocity_value) in enumerate(((901.0, 902.0), (903.0, 904.0))):
        if write_index == 1:
            for staging_name, staging_buffer in staging_buffers:
                assert staging_buffer is not None
                assert staging_buffer.timestamp >= 0.0, f"{backend} {staging_name} staging was invalidated"
                assert staging_buffer.timestamp < art.data._sim_timestamp
        env_ids, joint_ids, position_payload, velocity_payload = _write_selected_joint_state(
            art,
            backend,
            "index",
            operation,
            "one_env_one_item",
            position_value,
            velocity_value,
        )
        backend_joint_ids = user_to_backend[joint_ids]
        selected_cells = np.ix_(env_ids, backend_joint_ids)
        if operation in ("position", "state"):
            expected_position[selected_cells] = position_payload
        if operation in ("velocity", "state"):
            expected_velocity[selected_cells] = velocity_payload

        backend_position, backend_velocity = _read_backend_joint_state(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_position, expected_position)
        np.testing.assert_array_equal(backend_velocity, expected_velocity)

        if write_index == 0:
            selected_mask = np.zeros((num_instances, num_joints), dtype=bool)
            selected_mask[selected_cells] = True
            art.data._sim_timestamp += 0.01
            if operation in ("position", "state"):
                newer_position = 1000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(
                    num_instances, num_joints
                )
                expected_position[~selected_mask] = newer_position[~selected_mask]
            if operation in ("velocity", "state"):
                newer_velocity = 2000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(
                    num_instances, num_joints
                )
                expected_velocity[~selected_mask] = newer_velocity[~selected_mask]
            _mutate_backend_joint_state(backend, art, raw_backend, expected_position, expected_velocity)


def _exercise_duplicate_full_length_joint_write(backend: str, operation: str) -> None:
    """Assert duplicate full-length selectors preserve omitted backend joints."""
    num_instances = 2
    num_joints = 3
    art, raw_backend = get_articulation(
        backend,
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=2,
        device="cpu",
        joint_ordering=_joint_ordering_for_mode("reversed", num_joints),
    )
    initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
    _seed_backend_joint_state(backend, art, raw_backend, initial_position, initial_velocity)

    staging_buffers = []
    if operation in ("position", "state"):
        _ = art.data.joint_pos
        staging_buffers.append(art.data._joint_pos if backend == "physx" else art.data._joint_pos_buf)
        staging_buffers.append(art.data._joint_pos_backend)
    if operation in ("velocity", "state"):
        _ = art.data.joint_vel
        staging_buffers.append(art.data._joint_vel if backend == "physx" else art.data._joint_vel_buf)
        staging_buffers.append(art.data._joint_vel_backend)
    for staging_buffer in staging_buffers:
        assert staging_buffer is not None
        assert staging_buffer.timestamp == art.data._sim_timestamp

    art.data._sim_timestamp += 0.01
    newer_position = 1000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
    newer_velocity = 2000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
    _mutate_backend_joint_state(backend, art, raw_backend, newer_position, newer_velocity)

    env_ids = [1]
    joint_ids = [0, 1, 1]
    position_payload = np.asarray([[901.0, 902.0, 902.0]], dtype=np.float32)
    velocity_payload = np.asarray([[1901.0, 1902.0, 1902.0]], dtype=np.float32)
    env_ids_wp = wp.array(env_ids, dtype=wp.int32, device=art.device)
    joint_ids_wp = wp.array(joint_ids, dtype=wp.int32, device=art.device)
    position = torch.tensor(position_payload, dtype=torch.float32, device=art.device)
    velocity = torch.tensor(velocity_payload, dtype=torch.float32, device=art.device)
    if operation == "position":
        art.write_joint_position_to_sim_index(position=position, env_ids=env_ids_wp, joint_ids=joint_ids_wp)
    elif operation == "velocity":
        art.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids_wp, joint_ids=joint_ids_wp)
    elif backend == "ovphysx":
        with pytest.deprecated_call():
            art.write_joint_state_to_sim(
                position=position,
                velocity=velocity,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
    else:
        art.write_joint_state_to_sim_index(
            position=position,
            velocity=velocity,
            env_ids=env_ids_wp,
            joint_ids=joint_ids_wp,
        )

    user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
    expected_position = newer_position.copy()
    expected_velocity = newer_velocity.copy()
    for payload_index, joint_id in enumerate(joint_ids):
        backend_joint_id = user_to_backend[joint_id]
        if operation in ("position", "state"):
            expected_position[env_ids[0], backend_joint_id] = position_payload[0, payload_index]
        if operation in ("velocity", "state"):
            expected_velocity[env_ids[0], backend_joint_id] = velocity_payload[0, payload_index]

    backend_position, backend_velocity = _read_backend_joint_state(backend, art, raw_backend)
    np.testing.assert_array_equal(backend_position, expected_position)
    np.testing.assert_array_equal(backend_velocity, expected_velocity)
    omitted_backend_joint = user_to_backend[2]
    duplicate_backend_joint = user_to_backend[1]
    if operation in ("position", "state"):
        assert backend_position[env_ids[0], omitted_backend_joint] == newer_position[env_ids[0], omitted_backend_joint]
        assert backend_position[env_ids[0], duplicate_backend_joint] == position_payload[0, 1]
    if operation in ("velocity", "state"):
        assert backend_velocity[env_ids[0], omitted_backend_joint] == newer_velocity[env_ids[0], omitted_backend_joint]
        assert backend_velocity[env_ids[0], duplicate_backend_joint] == velocity_payload[0, 1]


def _make_dynamics_ordering_backend_data(
    num_instances: int, num_joints: int, num_jacobi_bodies: int, num_base_dofs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic backend-order dynamics data."""
    num_dofs = num_joints + num_base_dofs
    jacobian = np.arange(num_instances * num_jacobi_bodies * 6 * num_dofs, dtype=np.float32).reshape(
        num_instances, num_jacobi_bodies, 6, num_dofs
    )
    mass_matrix = np.arange(num_instances * num_dofs * num_dofs, dtype=np.float32).reshape(
        num_instances, num_dofs, num_dofs
    )
    gravity = np.arange(num_instances * num_dofs, dtype=np.float32).reshape(num_instances, num_dofs)
    return jacobian, mass_matrix, gravity


def _generalized_dof_user_to_backend(num_base_dofs: int, joint_user_to_backend: np.ndarray) -> np.ndarray:
    """Return generalized DoF user-to-backend indices including the floating base prefix."""
    return np.concatenate(
        (
            np.arange(num_base_dofs, dtype=np.int64),
            num_base_dofs + joint_user_to_backend,
        )
    )


def _jacobian_body_user_to_backend(art, body_user_to_backend: np.ndarray) -> np.ndarray:
    """Return Jacobian body-axis user-to-backend indices with the fixed root excluded."""
    if art.num_base_dofs == 0:
        return np.asarray([backend_id - 1 for backend_id in body_user_to_backend if backend_id != 0], dtype=np.int64)
    return body_user_to_backend


def _set_dynamics_ordering_backend_data(
    backend: str,
    art,
    raw_backend,
    jacobian: np.ndarray,
    mass_matrix: np.ndarray,
    gravity: np.ndarray,
) -> None:
    """Write deterministic backend-order dynamics data into the backend mock."""
    if backend == "physx":
        raw_backend.set_mock_jacobians(wp.array(jacobian, dtype=wp.float32, device=art.device))
        raw_backend.set_mock_generalized_mass_matrices(wp.array(mass_matrix, dtype=wp.float32, device=art.device))
        raw_backend.set_mock_gravity_compensation_forces(wp.array(gravity, dtype=wp.float32, device=art.device))
    elif backend == "newton":
        if art.is_fixed_base:
            model_jacobian = np.zeros((art.num_instances, art.num_bodies, 6, art.num_joints), dtype=np.float32)
            model_jacobian[:, 1:] = jacobian
        else:
            model_jacobian = jacobian
        raw_backend.set_mock_jacobians(wp.array(model_jacobian, dtype=wp.float32, device=art.device))
        raw_backend.set_mock_mass_matrices(wp.array(mass_matrix, dtype=wp.float32, device=art.device))
    else:
        raise AssertionError(f"Unsupported backend for dynamics-ordering test: {backend}")


def _set_body_ordering_backend_data(
    backend: str,
    art,
    raw_backend,
    root_pose: np.ndarray,
    root_vel: np.ndarray,
    link_pose: np.ndarray,
    com_pose_b: np.ndarray,
    body_com_vel: np.ndarray,
    body_acc: np.ndarray,
) -> None:
    """Write deterministic backend-order body state into the backend mock."""
    if backend == "physx":
        raw_backend._root_transforms = wp.array(root_pose, dtype=wp.float32, device=art.device)
        raw_backend._root_velocities = wp.array(root_vel, dtype=wp.float32, device=art.device)
        raw_backend._link_transforms = wp.array(link_pose, dtype=wp.float32, device=art.device)
        raw_backend._link_velocities = wp.array(body_com_vel, dtype=wp.float32, device=art.device)
        raw_backend._link_accelerations = wp.array(body_acc, dtype=wp.float32, device=art.device)
        raw_backend._coms = wp.array(com_pose_b, dtype=wp.float32, device="cpu")
    elif backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.ROOT_POSE]._data = root_pose.copy()
        raw_backend.bindings[TT.ROOT_VELOCITY]._data = root_vel.copy()
        raw_backend.bindings[TT.LINK_POSE]._data = link_pose.copy()
        raw_backend.bindings[TT.LINK_VELOCITY]._data = body_com_vel.copy()
        raw_backend.bindings[TT.LINK_ACCELERATION]._data = body_acc.copy()
        raw_backend.bindings[TT.BODY_COM_POSE]._data = com_pose_b.copy()
        art.data._body_com_pose_b.timestamp = -1.0
        if art.data._body_com_pose_b_backend is not None:
            art.data._body_com_pose_b_backend.timestamp = -1.0
    elif backend == "newton":
        root_pose_wp = wp.array(root_pose[:, None, :], dtype=wp.transformf, device=art.device)
        root_vel_wp = wp.array(root_vel[:, None, :], dtype=wp.spatial_vectorf, device=art.device)
        link_pose_wp = wp.array(link_pose[:, None, :, :], dtype=wp.transformf, device=art.device)
        body_com_vel_wp = wp.array(body_com_vel[:, None, :, :], dtype=wp.spatial_vectorf, device=art.device)
        body_com_pos_wp = wp.array(com_pose_b[:, None, :, :3], dtype=wp.vec3f, device=art.device)
        raw_backend.set_mock_root_transforms(root_pose_wp)
        raw_backend.set_mock_root_velocities(root_vel_wp)
        raw_backend.set_mock_link_transforms(link_pose_wp)
        raw_backend.set_mock_link_velocities(body_com_vel_wp)
        raw_backend.set_mock_coms(body_com_pos_wp)
        art.data._sim_bind_root_link_pose_w.assign(root_pose_wp[:, 0])
        art.data._sim_bind_root_com_vel_w.assign(root_vel_wp[:, 0])
        art.data._sim_bind_body_link_pose_w.assign(link_pose_wp[:, 0])
        art.data._sim_bind_body_com_vel_w.assign(body_com_vel_wp[:, 0])
        art.data._sim_bind_body_com_pos_b.assign(body_com_pos_wp[:, 0])
        art.data._previous_body_com_vel.assign(
            wp.zeros((art.num_instances, art.num_bodies), dtype=wp.spatial_vectorf, device=art.device)
        )
    else:
        raise AssertionError(f"Unsupported backend for body-ordering test: {backend}")


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


def _seed_root_writer_backend_com(backend: str, art, raw_backend, backend_coms: np.ndarray) -> None:
    """Seed root-writer COM inputs in backend order and verify the public view."""
    if backend in ("physx", "ovphysx"):
        _seed_backend_com_poses(backend, art, raw_backend, backend_coms)
        backend_actual = art.data._backend_body_com_pose_b.numpy()
        public_actual = art.data.body_com_pose_b.warp.numpy()
        expected_public = backend_coms[:, _ordering_user_to_backend(art.body_ordering, art.num_bodies)]
    elif backend == "newton":
        backend_positions = backend_coms[..., :3]
        art.data._sim_bind_body_com_pos_b.assign(wp.array(backend_positions, dtype=wp.vec3f, device=art.device))
        if art._has_body_ordering:
            art.data._sync_user_ordered_body_property_buffers()
        backend_actual = art.data._sim_bind_body_com_pos_b.numpy()
        public_actual = art.data.body_com_pos_b.warp.numpy()
        expected_public = backend_positions[:, _ordering_user_to_backend(art.body_ordering, art.num_bodies)]
        backend_coms = backend_positions
    else:
        raise AssertionError(f"Unsupported backend for root-writer parity test: {backend}")

    np.testing.assert_array_equal(backend_actual, backend_coms)
    np.testing.assert_array_equal(public_actual, expected_public)


def _write_root_writer_command(
    art,
    selection: str,
    operation: str,
    root_com_pose: np.ndarray,
    root_link_velocity: np.ndarray,
) -> None:
    """Write one root command through an index or mask API."""
    if selection == "index":
        env_ids = wp.array([0], dtype=wp.int32, device=art.device)
        if operation == "com_pose":
            art.write_root_com_pose_to_sim_index(
                root_pose=wp.array(root_com_pose[:1], dtype=wp.transformf, device=art.device),
                env_ids=env_ids,
            )
        else:
            art.write_root_link_pose_to_sim_index(
                root_pose=wp.array(root_com_pose[:1], dtype=wp.transformf, device=art.device),
                env_ids=env_ids,
            )
            art.write_root_link_velocity_to_sim_index(
                root_velocity=wp.array(root_link_velocity[:1], dtype=wp.spatial_vectorf, device=art.device),
                env_ids=env_ids,
            )
    elif selection == "mask":
        env_mask = wp.array([True, False], dtype=wp.bool, device=art.device)
        if operation == "com_pose":
            art.write_root_com_pose_to_sim_mask(
                root_pose=wp.array(root_com_pose, dtype=wp.transformf, device=art.device),
                env_mask=env_mask,
            )
        else:
            art.write_root_link_pose_to_sim_mask(
                root_pose=wp.array(root_com_pose, dtype=wp.transformf, device=art.device),
                env_mask=env_mask,
            )
            art.write_root_link_velocity_to_sim_mask(
                root_velocity=wp.array(root_link_velocity, dtype=wp.spatial_vectorf, device=art.device),
                env_mask=env_mask,
            )
    else:
        raise AssertionError(f"Unsupported root-writer selection: {selection}")


def _read_backend_root_state(backend: str, art, raw_backend) -> tuple[np.ndarray, np.ndarray]:
    """Read backend root link pose and COM velocity after a writer call."""
    if backend == "physx":
        return (
            raw_backend._root_transforms.numpy().copy(),
            art.data.root_com_vel_w.warp.numpy().copy(),
        )
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        return (
            raw_backend.bindings[TT.ROOT_POSE]._data.copy(),
            raw_backend.bindings[TT.ROOT_VELOCITY]._data.copy(),
        )
    if backend == "newton":
        return (
            art.data._sim_bind_root_link_pose_w.numpy().copy(),
            art.data._sim_bind_root_com_vel_w.numpy().copy(),
        )
    raise AssertionError(f"Unsupported backend for root-writer parity test: {backend}")


def _clone_proxy_tensor(arr) -> torch.Tensor:
    """Return a CPU clone of a proxy array's torch view."""
    return arr.torch.detach().cpu().clone()


def _assert_proxy_close(actual, expected: torch.Tensor) -> None:
    """Assert a proxy array is close to a CPU tensor."""
    torch.testing.assert_close(actual.torch.cpu(), expected, rtol=1.0e-5, atol=1.0e-5)


def _clone_backend_tensor(array) -> torch.Tensor:
    """Return a CPU tensor clone from a backend Warp or NumPy array."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.copy())
    return wp.to_torch(array).detach().cpu().clone()


def _get_backend_joint_property_tensors(backend: str, art, raw_backend) -> dict[str, torch.Tensor]:
    """Return backend-order joint property values for ordering parity checks."""
    if backend == "physx":
        return {
            "stiffness": _clone_backend_tensor(raw_backend.get_dof_stiffnesses()),
            "damping": _clone_backend_tensor(raw_backend.get_dof_dampings()),
            "armature": _clone_backend_tensor(raw_backend.get_dof_armatures()),
            "position_limits": _clone_backend_tensor(raw_backend.get_dof_limits()),
            "velocity_limits": _clone_backend_tensor(raw_backend.get_dof_max_velocities()),
            "effort_limits": _clone_backend_tensor(raw_backend.get_dof_max_forces()),
            "friction": _clone_backend_tensor(raw_backend.get_dof_friction_properties())[..., 0],
        }
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        return {
            "stiffness": _clone_backend_tensor(raw_backend.bindings[TT.DOF_STIFFNESS]._data),
            "damping": _clone_backend_tensor(raw_backend.bindings[TT.DOF_DAMPING]._data),
            "armature": _clone_backend_tensor(raw_backend.bindings[TT.DOF_ARMATURE]._data),
            "position_limits": _clone_backend_tensor(raw_backend.bindings[TT.DOF_LIMIT]._data),
            "velocity_limits": _clone_backend_tensor(raw_backend.bindings[TT.DOF_MAX_VELOCITY]._data),
            "effort_limits": _clone_backend_tensor(raw_backend.bindings[TT.DOF_MAX_FORCE]._data),
            "friction": _clone_backend_tensor(raw_backend.bindings[TT.DOF_FRICTION_PROPERTIES]._data)[..., 0],
        }
    if backend == "newton":
        return {
            "stiffness": _clone_backend_tensor(art.data._sim_bind_joint_stiffness_sim),
            "damping": _clone_backend_tensor(art.data._sim_bind_joint_damping_sim),
            "armature": _clone_backend_tensor(art.data._sim_bind_joint_armature),
            "position_limits": torch.stack(
                (
                    _clone_backend_tensor(art.data._sim_bind_joint_pos_limits_lower),
                    _clone_backend_tensor(art.data._sim_bind_joint_pos_limits_upper),
                ),
                dim=-1,
            ),
            "velocity_limits": _clone_backend_tensor(art.data._sim_bind_joint_vel_limits_sim),
            "effort_limits": _clone_backend_tensor(art.data._sim_bind_joint_effort_limits_sim),
            "friction": _clone_backend_tensor(art.data._sim_bind_joint_friction_coeff),
        }
    raise AssertionError(f"Unsupported backend for joint-property ordering test: {backend}")


# Common parametrize decorator for all interface tests
_backends = pytest.mark.parametrize("backend", BACKENDS, indirect=False)

# We also need to provide the fixture params that articulation_iface reads:
_default_dims = pytest.mark.parametrize(
    "num_instances, num_joints, num_bodies",
    [(1, 1, 1), (1, 2, 2), (2, 6, 7), (100, 8, 13)],
)

_default_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])
_index_resolution_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton") if backend in BACKENDS], indirect=False
)
_dynamics_ordering_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton") if backend in BACKENDS], indirect=False
)


# ---------------------------------------------------------------------------
# Tests: Index resolution helpers
# ---------------------------------------------------------------------------


class TestArticulationIndexResolution:
    """Test backend-specific index resolution helpers."""

    @_index_resolution_backends
    def test_resolve_env_ids_handles_tensor_view_shape(self, backend):
        art, _ = get_articulation(backend, num_instances=4, device="cpu")

        env_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = art._resolve_env_ids(env_ids)
        resolved_view = art._resolve_env_ids(env_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2

    @_index_resolution_backends
    def test_resolve_joint_ids_handles_tensor_view_shape(self, backend):
        art, _ = get_articulation(backend, num_joints=4, device="cpu")

        joint_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = art._resolve_joint_ids(joint_ids)
        resolved_view = art._resolve_joint_ids(joint_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2

    @_index_resolution_backends
    def test_resolve_body_ids_handles_tensor_view_shape(self, backend):
        art, _ = get_articulation(backend, num_bodies=4, device="cpu")

        body_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = art._resolve_body_ids(body_ids)
        resolved_view = art._resolve_body_ids(body_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: Articulation properties
# ---------------------------------------------------------------------------


class TestArticulationProperties:
    """Test that articulation properties return the correct types/values."""

    @_backends
    @_default_dims
    @_default_devices
    def test_num_instances(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert art.num_instances == num_instances

    @_backends
    @_default_dims
    @_default_devices
    def test_num_joints(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert art.num_joints == num_joints

    @_backends
    @_default_dims
    @_default_devices
    def test_num_bodies(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert art.num_bodies == num_bodies

    @_backends
    @_default_dims
    @_default_devices
    def test_is_fixed_base(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert isinstance(art.is_fixed_base, bool)

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_names(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        names = art.joint_names
        assert isinstance(names, list)
        assert len(names) == num_joints
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_body_names(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        names = art.body_names
        assert isinstance(names, list)
        assert len(names) == num_bodies
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_data_returns_articulation_data(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface
    ):
        from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

        art, _ = articulation_iface
        assert isinstance(art.data, BaseArticulationData)


# ---------------------------------------------------------------------------
# Tests: Articulation finder methods
# ---------------------------------------------------------------------------


class TestArticulationFinders:
    """Test that finder methods return (list[int], list[str]) tuples."""

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_all(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        indices, names = art.find_bodies(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_bodies
        assert len(names) == num_bodies
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_find_joints_all(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        indices, names = art.find_joints(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_joints
        assert len(names) == num_joints

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_single(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        first_body = art.body_names[0]
        indices, names = art.find_bodies(first_body)
        assert indices == [0]
        assert names == [first_body]

    @_backends
    @_default_dims
    @_default_devices
    def test_find_joints_single(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        first_joint = art.joint_names[0]
        indices, names = art.find_joints(first_joint)
        assert indices == [0]
        assert names == [first_joint]


# ---------------------------------------------------------------------------
# Tests: resolve_matching_names caching behavior
# ---------------------------------------------------------------------------


_non_mock_backends = pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "mock"], indirect=False)


class TestResolveMatchingNamesCache:
    """Test that resolve_matching_names caching returns correct, isolated results."""

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_unmatched_regex_raises(self, backend, num_instances, num_joints, num_bodies, device):
        """ValueError from resolve_matching_names propagates correctly."""
        art, _ = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        with pytest.raises(ValueError):
            art.find_bodies("nonexistent_body_xyz")
        with pytest.raises(ValueError):
            art.find_joints("nonexistent_joint_xyz")

    @_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_mutating_result_does_not_corrupt_cache(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface
    ):
        """Mutating returned lists must not affect future cached results."""
        art, _ = articulation_iface

        for finder, expected_len in [("find_bodies", num_bodies), ("find_joints", num_joints)]:
            idx1, names1 = getattr(art, finder)(".*")
            assert len(idx1) == expected_len

            idx1.clear()
            names1.append("corrupted")

            idx2, names2 = getattr(art, finder)(".*")
            assert len(idx2) == expected_len
            assert "corrupted" not in names2

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_find_with_multiple_patterns(self, backend, num_instances, num_joints, num_bodies, device):
        """Passing a list of regex patterns works correctly."""
        art, _ = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        idx, names = art.find_joints(["joint_0", "joint_1"])
        assert "joint_0" in names
        assert "joint_1" in names
        assert len(names) == 2

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_find_with_preserve_order(self, backend, num_instances, num_joints, num_bodies, device):
        """preserve_order=True returns names in the order of the input patterns."""
        art, _ = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        idx_fwd, names_fwd = art.find_joints(["joint_1", "joint_0"], preserve_order=True)
        assert names_fwd == ["joint_1", "joint_0"]

        idx_rev, names_rev = art.find_joints(["joint_0", "joint_1"], preserve_order=True)
        assert names_rev == ["joint_0", "joint_1"]


# ---------------------------------------------------------------------------
# Tests: ArticulationData root state properties
# ---------------------------------------------------------------------------


class TestArticulationDataRootState:
    """Test data properties for root rigid body state."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_link_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_link_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_com_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_com_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_w",
        )


# ---------------------------------------------------------------------------
# Tests: ArticulationData derived properties
# ---------------------------------------------------------------------------


class TestArticulationDataDerivedProperties:
    """Test derived/computed data properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_projected_gravity_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.projected_gravity_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="projected_gravity_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_heading_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.heading_w, expected_shape=(num_instances,), expected_dtype=wp.float32, name="heading_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_b",
        )


# ---------------------------------------------------------------------------
# Tests: ArticulationData body state properties
# ---------------------------------------------------------------------------


class TestArticulationDataBodyState:
    """Test data properties for all body states."""

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 1, 3)])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_reversed_body_ordering_reorders_public_body_quantities(
        self, backend, num_instances, num_joints, num_bodies, device
    ):
        identity_art, identity_raw = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        ordered_art, ordered_raw = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        raw_data = _make_body_ordering_backend_data(num_instances, num_bodies)
        _set_body_ordering_backend_data(backend, identity_art, identity_raw, *raw_data)
        _set_body_ordering_backend_data(backend, ordered_art, ordered_raw, *raw_data)
        user_to_backend = _install_test_body_ordering(ordered_art)

        identity_art.data.update(dt=0.01)
        ordered_art.data.update(dt=0.01)

        identity_body_com_pose_b = _clone_proxy_tensor(identity_art.data.body_com_pose_b)
        identity_body_com_acc_w = _clone_proxy_tensor(identity_art.data.body_com_acc_w)
        identity_body_link_vel_w = _clone_proxy_tensor(identity_art.data.body_link_vel_w)
        identity_body_com_pose_w = _clone_proxy_tensor(identity_art.data.body_com_pose_w)
        identity_root_com_pose_w = _clone_proxy_tensor(identity_art.data.root_com_pose_w)
        identity_root_link_vel_w = _clone_proxy_tensor(identity_art.data.root_link_vel_w)

        _assert_proxy_close(ordered_art.data.body_com_pose_b, identity_body_com_pose_b[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.body_com_acc_w, identity_body_com_acc_w[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.body_link_vel_w, identity_body_link_vel_w[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.body_com_pose_w, identity_body_com_pose_w[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.root_com_pose_w, identity_root_com_pose_w)
        _assert_proxy_close(ordered_art.data.root_link_vel_w, identity_root_link_vel_w)

    def test_ovphysx_reversed_body_ordering_rereads_backend_shadows_after_reset(self):
        """Refresh OVPhysX backend shadow buffers after same-step pose/velocity invalidation."""
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        num_instances = 2
        num_joints = 1
        num_bodies = 3
        art, raw_backend = get_articulation("ovphysx", num_instances, num_joints, num_bodies, device="cpu")
        raw_data = _make_body_ordering_backend_data(num_instances, num_bodies)
        _set_body_ordering_backend_data("ovphysx", art, raw_backend, *raw_data)
        user_to_backend = _install_test_body_ordering(art)
        art.data.update(dt=0.01)

        _ = _clone_proxy_tensor(art.data.body_link_pose_w)
        _ = _clone_proxy_tensor(art.data.body_com_vel_w)

        from isaaclab_ovphysx import tensor_types as TT

        next_link_pose = raw_data[2].copy()
        next_link_pose[..., 0] += 1000.0
        next_body_com_vel = raw_data[4].copy()
        next_body_com_vel[..., 0] += 500.0
        raw_backend.bindings[TT.LINK_POSE]._data = next_link_pose
        raw_backend.bindings[TT.LINK_VELOCITY]._data = next_body_com_vel

        art.data._reset_pose()
        art.data._reset_velocity()

        expected_link_pose = torch.from_numpy(next_link_pose[:, user_to_backend])
        expected_body_com_vel = torch.from_numpy(next_body_com_vel[:, user_to_backend])
        _assert_proxy_close(art.data.body_link_pose_w, expected_link_pose)
        _assert_proxy_close(art.data.body_com_vel_w, expected_body_com_vel)

    def test_ovphysx_reversed_body_ordering_rereads_all_velocity_shadows_after_reset(self):
        """Refresh every OVPhysX velocity shadow after a same-step reset."""
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        art, raw_backend = get_articulation(
            "ovphysx", 2, 1, 3, device="cpu", body_ordering=("body_2", "body_1", "body_0")
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
        """Invalidate every public and backend cache derived from OVPhysX COM poses."""
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        art, _ = get_articulation("ovphysx", 2, 1, 3, device="cpu", body_ordering=("body_2", "body_1", "body_0"))
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

    @_dynamics_ordering_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 3, 4)])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
    def test_ordering_reorders_public_dynamics_quantities(
        self, backend, num_instances, num_joints, num_bodies, device, is_fixed_base
    ):
        identity_art, identity_raw = get_articulation(
            backend, num_instances, num_joints, num_bodies, device=device, is_fixed_base=is_fixed_base
        )
        ordered_art, ordered_raw = get_articulation(
            backend, num_instances, num_joints, num_bodies, device=device, is_fixed_base=is_fixed_base
        )
        raw_body_data = _make_body_ordering_backend_data(num_instances, num_bodies)
        _set_body_ordering_backend_data(backend, identity_art, identity_raw, *raw_body_data)
        _set_body_ordering_backend_data(backend, ordered_art, ordered_raw, *raw_body_data)

        num_base_dofs = identity_art.num_base_dofs
        num_jacobi_bodies = num_bodies - (1 if num_base_dofs == 0 else 0)
        raw_dynamics_data = _make_dynamics_ordering_backend_data(
            num_instances, num_joints, num_jacobi_bodies, num_base_dofs
        )
        _set_dynamics_ordering_backend_data(backend, identity_art, identity_raw, *raw_dynamics_data)
        _set_dynamics_ordering_backend_data(backend, ordered_art, ordered_raw, *raw_dynamics_data)
        body_user_to_backend = _install_test_body_ordering(ordered_art)
        joint_user_to_backend = _install_reversed_joint_ordering(ordered_art)

        identity_art.data.update(dt=0.01)
        ordered_art.data.update(dt=0.01)

        if backend == "newton":
            import isaaclab_newton.assets.articulation.articulation_data as newton_data_module

            original_simulation_manager = newton_data_module.SimulationManager
            newton_data_module.SimulationManager = identity_art._test_simulation_manager
        else:
            newton_data_module = None
            original_simulation_manager = None

        try:
            jacobian_body_indices = _jacobian_body_user_to_backend(ordered_art, body_user_to_backend)
            dof_indices = _generalized_dof_user_to_backend(num_base_dofs, joint_user_to_backend)
            identity_body_com_jacobian_w = _clone_proxy_tensor(identity_art.data.body_com_jacobian_w)
            identity_body_link_jacobian_w = _clone_proxy_tensor(identity_art.data.body_link_jacobian_w)
            identity_mass_matrix = _clone_proxy_tensor(identity_art.data.mass_matrix)

            expected_body_com_jacobian_w = identity_body_com_jacobian_w[:, jacobian_body_indices][:, :, :, dof_indices]
            expected_body_link_jacobian_w = identity_body_link_jacobian_w[:, jacobian_body_indices][
                :, :, :, dof_indices
            ]
            expected_mass_matrix = identity_mass_matrix[:, dof_indices][:, :, dof_indices]

            _assert_proxy_close(ordered_art.data.body_com_jacobian_w, expected_body_com_jacobian_w)
            _assert_proxy_close(ordered_art.data.body_link_jacobian_w, expected_body_link_jacobian_w)
            _assert_proxy_close(ordered_art.data.mass_matrix, expected_mass_matrix)

            if backend == "physx":
                identity_gravity_compensation_forces = _clone_proxy_tensor(
                    identity_art.data.gravity_compensation_forces
                )
                expected_gravity_compensation_forces = identity_gravity_compensation_forces[:, dof_indices]
                _assert_proxy_close(
                    ordered_art.data.gravity_compensation_forces,
                    expected_gravity_compensation_forces,
                )
        finally:
            if newton_data_module is not None:
                newton_data_module.SimulationManager = original_simulation_manager

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_pose_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pose_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_acc_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_acc_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_acc_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        if backend == "newton":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pose_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_com_pose_b",
        )

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_body_com_pose_b_is_cached_across_sim_timestamps(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        art.data.update(dt=0.01)
        art.data.body_com_pose_b
        assert num_get_coms_calls == 1

        art.data.update(dt=0.01)
        art.data.body_com_pose_b
        assert num_get_coms_calls == 1

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_set_coms_index_updates_body_com_pose_b_cache(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        coms = wp.zeros((art.num_instances, art.num_bodies), dtype=wp.transformf, device="cpu")
        art.set_coms_index(coms=coms, full_data=True)
        art.data.body_com_pose_b

        assert num_get_coms_calls == 0

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_joint_position_write_preserves_body_com_pose_b_cache(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        art.data.update(dt=0.01)
        art.data.body_com_pose_b
        assert num_get_coms_calls == 1

        joint_pos = torch.zeros((art.num_instances, art.num_joints), device="cpu")
        art.write_joint_position_to_sim_index(position=joint_pos, full_data=True)
        art.data.body_com_pose_b

        assert num_get_coms_calls == 1

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_partial_set_coms_index_initializes_cold_body_com_pose_b_cache(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")
        initial_coms = view.get_coms().numpy().copy()

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        coms = wp.zeros((1, 1), dtype=wp.transformf, device="cpu")
        art.set_coms_index(
            coms=coms,
            env_ids=wp.array([0], dtype=wp.int32, device="cpu"),
            body_ids=wp.array([0], dtype=wp.int32, device="cpu"),
        )
        body_com_pose_b = art.data.body_com_pose_b.torch

        assert num_get_coms_calls == 1
        torch.testing.assert_close(body_com_pose_b[1, 1], torch.from_numpy(initial_coms[1, 1]))

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_set_coms_index_invalidates_body_com_pose_b_dependents(self):
        art, _ = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        art.data.update(dt=0.01)
        dependent_buffers = [
            ("root_com_pose_w", art.data._root_com_pose_w),
            ("root_com_vel_w", art.data._root_com_vel_w),
            ("root_link_vel_w", art.data._root_link_vel_w),
            ("body_com_pose_w", art.data._body_com_pose_w),
            ("body_com_vel_w", art.data._body_com_vel_w),
            ("body_link_vel_w", art.data._body_link_vel_w),
            ("root_link_lin_vel_b", art.data._root_link_lin_vel_b),
            ("root_link_ang_vel_b", art.data._root_link_ang_vel_b),
            ("root_com_lin_vel_b", art.data._root_com_lin_vel_b),
            ("root_com_ang_vel_b", art.data._root_com_ang_vel_b),
            ("root_state_w", art.data._root_state_w),
            ("root_link_state_w", art.data._root_link_state_w),
            ("root_com_state_w", art.data._root_com_state_w),
            ("body_state_w", art.data._body_state_w),
            ("body_link_state_w", art.data._body_link_state_w),
            ("body_com_state_w", art.data._body_com_state_w),
            ("body_com_jacobian_w", art.data._body_com_jacobian_w),
            ("mass_matrix", art.data._mass_matrix),
            ("gravity_compensation_forces", art.data._gravity_compensation_forces),
        ]
        for _, buffer in dependent_buffers:
            buffer.timestamp = art.data._sim_timestamp

        coms = wp.zeros((art.num_instances, art.num_bodies), dtype=wp.transformf, device="cpu")
        art.set_coms_index(coms=coms, full_data=True)

        for name, buffer in dependent_buffers:
            assert buffer.timestamp < art.data._sim_timestamp, name

    @_backends
    @_default_dims
    @_default_devices
    def test_body_mass(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_mass, expected_shape=(num_instances, num_bodies), expected_dtype=wp.float32, name="body_mass"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_inertia(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_inertia,
            expected_shape=(num_instances, num_bodies, 9),
            expected_dtype=wp.float32,
            name="body_inertia",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_pos_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_pos_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_quat_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_link_quat_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_lin_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_lin_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_ang_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_ang_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pos_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_pos_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_quat_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_com_quat_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pos_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_pos_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        if backend == "newton":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_quat_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_com_quat_b",
        )


class TestArticulationOrderingAllocation:
    """Test that ordering-only shadows exist only for nonidentity mappings."""

    _DATA_SHADOWS = {
        "physx": (
            "_joint_pos_backend",
            "_joint_vel_backend",
            "_joint_stiffness_backend",
            "_joint_damping_backend",
            "_joint_armature_backend",
            "_joint_pos_limits_backend",
            "_joint_vel_limits_backend",
            "_joint_effort_limits_backend",
            "_joint_friction_props_backend",
            "_joint_friction_props_user",
            "_body_com_pose_b_backend",
            "_body_mass_backend",
            "_body_inertia_backend",
        ),
        "ovphysx": (
            "_body_link_pose_w_backend",
            "_body_link_vel_w_backend",
            "_body_com_pose_b_backend",
            "_body_com_vel_w_backend",
            "_body_com_acc_w_backend",
            "_joint_pos_backend",
            "_joint_vel_backend",
            "_joint_stiffness_backend",
            "_joint_damping_backend",
            "_joint_armature_backend",
            "_joint_pos_limits_backend",
            "_joint_vel_limits_backend",
            "_joint_effort_limits_backend",
            "_joint_friction_props_backend",
            "_body_mass_backend",
            "_body_inertia_backend",
        ),
        "newton": (
            "_joint_pos_user",
            "_joint_vel_user",
            "_joint_stiffness_user",
            "_joint_damping_user",
            "_joint_armature_user",
            "_joint_friction_coeff_user",
            "_joint_pos_limits_lower_user",
            "_joint_pos_limits_upper_user",
            "_joint_vel_limits_user",
            "_joint_effort_limits_user",
            "_body_link_pose_w_user",
            "_body_com_vel_w_user",
            "_body_com_acc_w_backend",
            "_body_mass_user",
            "_body_inertia_user",
            "_body_com_pos_b_user",
        ),
    }
    _ARTICULATION_SHADOWS = {
        "physx": (
            "_joint_pos_target_backend",
            "_joint_vel_target_backend",
            "_joint_effort_target_backend",
            "_body_wrench_force_backend",
            "_body_wrench_torque_backend",
        ),
        "ovphysx": (
            "_joint_pos_target_backend",
            "_joint_vel_target_backend",
            "_joint_effort_target_backend",
        ),
        "newton": (),
    }

    @pytest.mark.parametrize("backend", ["physx", "ovphysx"])
    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    def test_physx_and_ovphysx_allocate_shadows_only_for_nonidentity_ordering(
        self, backend: str, ordering_mode: str
    ) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        num_instances = 2
        num_joints = 3
        num_bodies = 4
        art, _ = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode(ordering_mode, num_joints),
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )

        if ordering_mode == "none":
            assert art.joint_ordering is None
            assert art.body_ordering is None
        else:
            assert art.joint_ordering is not None
            assert art.body_ordering is not None
            expected_identity = ordering_mode == "identity"
            assert art.joint_ordering.is_identity is expected_identity
            assert art.body_ordering.is_identity is expected_identity

        expected_ordering = ordering_mode == "reversed"
        assert art._has_joint_ordering is expected_ordering
        assert art._has_body_ordering is expected_ordering
        assert art.data._has_joint_ordering is expected_ordering
        assert art.data._has_body_ordering is expected_ordering

        for owner, field_names in (
            (art.data, self._DATA_SHADOWS[backend]),
            (art, self._ARTICULATION_SHADOWS[backend]),
        ):
            for field_name in field_names:
                shadow = getattr(owner, field_name)
                if not expected_ordering:
                    assert shadow is None, f"{backend} {ordering_mode} unexpectedly allocated {field_name}"
                    continue
                assert shadow is not None, f"{backend} reversed ordering did not allocate {field_name}"
                item_count = num_bodies if "_body_" in field_name else num_joints
                assert _ordering_shadow_shape(shadow)[:2] == (num_instances, item_count)

    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    def test_newton_allocates_shadows_only_for_nonidentity_ordering(self, ordering_mode: str) -> None:
        if "newton" not in BACKENDS:
            pytest.skip("Newton backend is not available")
        num_instances = 2
        num_joints = 3
        num_bodies = 4
        art, _ = get_articulation(
            "newton",
            num_instances,
            num_joints,
            num_bodies,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode(ordering_mode, num_joints),
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )

        if ordering_mode == "none":
            assert art.joint_ordering is None
            assert art.body_ordering is None
        else:
            assert art.joint_ordering is not None
            assert art.body_ordering is not None
            expected_identity = ordering_mode == "identity"
            assert art.joint_ordering.is_identity is expected_identity
            assert art.body_ordering.is_identity is expected_identity

        expected_ordering = ordering_mode == "reversed"
        assert art._has_joint_ordering is expected_ordering
        assert art._has_body_ordering is expected_ordering
        assert art.data._has_joint_ordering is expected_ordering
        assert art.data._has_body_ordering is expected_ordering

        unexpected_fields = []
        for field_name in self._DATA_SHADOWS["newton"]:
            shadow = getattr(art.data, field_name)
            if not expected_ordering:
                if shadow is not None:
                    unexpected_fields.append(field_name)
                continue
            assert shadow is not None, f"newton reversed ordering did not allocate {field_name}"
            item_count = num_bodies if "_body_" in field_name else num_joints
            assert _ordering_shadow_shape(shadow)[:2] == (num_instances, item_count)
        assert not unexpected_fields, f"newton {ordering_mode} unexpectedly allocated {unexpected_fields}"


class TestArticulationOrderingLifecycle:
    """Test persistent commands and shadows across articulation map transitions."""

    @pytest.mark.parametrize("backend", ["physx", "ovphysx", "newton"])
    def test_joint_targets_preserve_backend_identity_across_map_transitions(self, backend: str) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        num_instances = 2
        num_joints = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=num_joints,
            num_bodies=4,
            device="cpu",
        )
        values = np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
        backend_targets = (values + 10.0, values + 110.0, values + 1010.0)
        backend_joint_stiffness = art.data.joint_stiffness.warp.numpy().copy()
        backend_body_mass = art.data.body_mass.warp.numpy().copy()
        _seed_public_joint_commands(backend, art, backend_targets)

        for actual, expected in zip(
            _write_joint_commands_to_backend(backend, art, raw_backend),
            backend_targets,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        _assert_public_joint_commands(backend, art, backend_targets)
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

        backend_names = tuple(art.backend_joint_names)
        ordering_a = tuple(backend_names[index] for index in (2, 0, 3, 1))
        ordering_b = tuple(backend_names[index] for index in (1, 3, 0, 2))
        for joint_ordering in (ordering_a, ordering_b, backend_names, None):
            _transition_articulation_ordering(art, joint_ordering=joint_ordering)
            for actual, expected in zip(
                _write_joint_commands_to_backend(backend, art, raw_backend),
                backend_targets,
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            _assert_public_joint_commands(backend, art, backend_targets)
            _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

    def test_physx_body_wrenches_preserve_backend_identity_across_map_transitions(self) -> None:
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            "physx",
            num_instances=num_instances,
            num_joints=3,
            num_bodies=num_bodies,
            device="cpu",
        )
        _set_identity_body_poses("physx", art, raw_backend)
        object.__setattr__(art, "_instantaneous_wrench_composer", WrenchComposer(art))
        object.__setattr__(art, "_permanent_wrench_composer", WrenchComposer(art))

        backend_force = (
            np.arange(num_instances * num_bodies * 3, dtype=np.float32).reshape(num_instances, num_bodies, 3) + 1.0
        )
        backend_torque = backend_force + 100.0
        art.permanent_wrench_composer.set_forces_and_torques_index(
            forces=wp.array(backend_force, dtype=wp.vec3f, device=art.device),
            torques=wp.array(backend_torque, dtype=wp.vec3f, device=art.device),
        )
        backend_joint_stiffness = art.data.joint_stiffness.warp.numpy().copy()
        backend_body_mass = art.data.body_mass.warp.numpy().copy()
        captured = {}

        def capture_wrench(*, force_data, torque_data, position_data, indices, is_global):
            captured["force"] = force_data.numpy().reshape(num_instances, num_bodies, 3).copy()
            captured["torque"] = torque_data.numpy().reshape(num_instances, num_bodies, 3).copy()

        raw_backend.apply_forces_and_torques_at_position = capture_wrench

        backend_names = tuple(art.backend_body_names)
        ordering_a = tuple(backend_names[index] for index in (2, 0, 3, 1))
        ordering_b = tuple(backend_names[index] for index in (1, 3, 0, 2))
        for body_ordering in (_ORDERING_UNCHANGED, ordering_a, ordering_b, backend_names, None):
            if body_ordering is not _ORDERING_UNCHANGED:
                _transition_articulation_ordering(art, body_ordering=body_ordering)
            art.write_data_to_sim()
            np.testing.assert_array_equal(captured["force"], backend_force)
            np.testing.assert_array_equal(captured["torque"], backend_torque)

            user_to_backend = _ordering_user_to_backend(art.body_ordering, num_bodies)
            np.testing.assert_array_equal(
                art.permanent_wrench_composer.local_force_b.numpy(),
                backend_force[:, user_to_backend],
            )
            np.testing.assert_array_equal(
                art.permanent_wrench_composer.local_torque_b.numpy(),
                backend_torque[:, user_to_backend],
            )
            if art._has_body_ordering:
                assert art._body_wrench_force_backend is not None
                assert art._body_wrench_torque_backend is not None
                np.testing.assert_array_equal(art._body_wrench_force_backend.numpy(), backend_force)
                np.testing.assert_array_equal(art._body_wrench_torque_backend.numpy(), backend_torque)
            else:
                assert art._body_wrench_force_backend is None
                assert art._body_wrench_torque_backend is None
            _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

    @pytest.mark.parametrize("backend", ["physx", "ovphysx", "newton"])
    def test_joint_and_body_transitions_manage_only_their_domain_shadows(self, backend: str) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        art, _ = get_articulation(
            backend,
            num_instances=2,
            num_joints=4,
            num_bodies=4,
            device="cpu",
        )
        backend_joint_stiffness = art.data.joint_stiffness.warp.numpy().copy()
        backend_body_mass = art.data.body_mass.warp.numpy().copy()
        backend_joint_names = tuple(art.backend_joint_names)
        backend_body_names = tuple(art.backend_body_names)
        joint_ordering_a = tuple(backend_joint_names[index] for index in (2, 0, 3, 1))
        joint_ordering_b = tuple(backend_joint_names[index] for index in (1, 3, 0, 2))
        body_ordering_a = tuple(backend_body_names[index] for index in (1, 3, 0, 2))
        body_ordering_b = tuple(backend_body_names[index] for index in (2, 0, 3, 1))
        data_joint_fields = tuple(
            name for name in TestArticulationOrderingAllocation._DATA_SHADOWS[backend] if "_joint_" in name
        )
        data_body_fields = tuple(
            name for name in TestArticulationOrderingAllocation._DATA_SHADOWS[backend] if "_body_" in name
        )
        articulation_joint_fields = tuple(
            name for name in TestArticulationOrderingAllocation._ARTICULATION_SHADOWS[backend] if "_joint_" in name
        )
        articulation_body_fields = tuple(
            name for name in TestArticulationOrderingAllocation._ARTICULATION_SHADOWS[backend] if "_body_" in name
        )

        _transition_articulation_ordering(art, joint_ordering=joint_ordering_a)
        for owner, fields in ((art.data, data_joint_fields), (art, articulation_joint_fields)):
            assert all(getattr(owner, name) is not None for name in fields)
        for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields)):
            assert all(getattr(owner, name) is None for name in fields)
        joint_shadow_objects = {
            (id(owner), name): getattr(owner, name)
            for owner, fields in ((art.data, data_joint_fields), (art, articulation_joint_fields))
            for name in fields
        }
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

        _transition_articulation_ordering(art, joint_ordering=joint_ordering_b)
        for owner, fields in ((art.data, data_joint_fields), (art, articulation_joint_fields)):
            for name in fields:
                assert getattr(owner, name) is joint_shadow_objects[(id(owner), name)]
        for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields)):
            assert all(getattr(owner, name) is None for name in fields)
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

        _transition_articulation_ordering(art, body_ordering=body_ordering_a)
        for owner, fields in ((art.data, data_joint_fields), (art, articulation_joint_fields)):
            for name in fields:
                assert getattr(owner, name) is joint_shadow_objects[(id(owner), name)]
        for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields)):
            assert all(getattr(owner, name) is not None for name in fields)
        body_shadow_objects = {
            (id(owner), name): getattr(owner, name)
            for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields))
            for name in fields
        }
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

        _transition_articulation_ordering(art, body_ordering=body_ordering_b)
        for owner, fields in ((art.data, data_joint_fields), (art, articulation_joint_fields)):
            for name in fields:
                assert getattr(owner, name) is joint_shadow_objects[(id(owner), name)]
        for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields)):
            for name in fields:
                assert getattr(owner, name) is body_shadow_objects[(id(owner), name)]
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

        _transition_articulation_ordering(art, joint_ordering=backend_joint_names)
        for owner, fields in ((art.data, data_joint_fields), (art, articulation_joint_fields)):
            assert all(getattr(owner, name) is None for name in fields)
        for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields)):
            for name in fields:
                assert getattr(owner, name) is body_shadow_objects[(id(owner), name)]
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)

        _transition_articulation_ordering(art, body_ordering=backend_body_names)
        for owner, fields in ((art.data, data_body_fields), (art, articulation_body_fields)):
            assert all(getattr(owner, name) is None for name in fields)
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)
        _transition_articulation_ordering(art, joint_ordering=None, body_ordering=None)
        assert art.joint_ordering is None
        assert art.body_ordering is None
        for owner, fields in (
            (art.data, data_joint_fields),
            (art, articulation_joint_fields),
            (art.data, data_body_fields),
            (art, articulation_body_fields),
        ):
            assert all(getattr(owner, name) is None for name in fields)
        _assert_public_ordered_properties(art, backend_joint_stiffness, backend_body_mass)


class TestArticulationOrderingComWrites:
    """Test coherent partial COM writes and step-independent static COM caches."""

    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize(
        "coverage",
        ["one_env_one_item", "all_envs_one_item", "one_env_all_items"],
    )
    def test_ovphysx_partial_com_write_preserves_backend_rows(
        self,
        ordering_mode: str,
        selection: str,
        coverage: str,
    ) -> None:
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            "ovphysx",
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        _seed_backend_com_poses("ovphysx", art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses("ovphysx", art, raw_backend), backend_seed)

        art.data._body_com_pose_b.timestamp = -1.0
        backend_staging = art.data._body_com_pose_b_backend
        if backend_staging is not None:
            backend_staging.timestamp = -1.0

        if art.body_ordering is None:
            user_to_backend = np.arange(num_bodies, dtype=np.int64)
        else:
            user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)

        first_env_ids, first_body_ids = _coverage_ids(coverage, num_instances, num_bodies)
        if len(first_body_ids) == num_bodies:
            second_env_ids = [0]
            second_body_ids = first_body_ids
        else:
            second_env_ids = first_env_ids
            second_body_ids = [1]
        write_selections = (
            (first_env_ids, first_body_ids, 1000.0),
            (second_env_ids, second_body_ids, 2000.0),
        )
        if ordering_mode == "reversed":
            for _, body_ids, _ in write_selections:
                if len(body_ids) < num_bodies:
                    assert all(user_to_backend[body_id] != body_id for body_id in body_ids)

        expected_backend = backend_seed.copy()
        first_backend_cells = None
        first_payload = None
        backend_after = None
        for write_index, (env_ids, body_ids, base_value) in enumerate(write_selections):
            payload = _make_selected_com_poses(env_ids, body_ids, base_value)
            expected_before = expected_backend.copy()
            if selection == "index":
                art.set_coms_index(
                    coms=wp.array(payload, dtype=wp.transformf, device=art.device),
                    env_ids=wp.array(env_ids, dtype=wp.int32, device=art.device),
                    body_ids=wp.array(body_ids, dtype=wp.int32, device=art.device),
                )
            else:
                full_coms = np.zeros_like(backend_seed)
                full_coms[np.ix_(env_ids, body_ids)] = payload
                env_mask = wp.array(
                    [env_index in env_ids for env_index in range(num_instances)],
                    dtype=wp.bool,
                    device=art.device,
                )
                body_mask = None
                if len(body_ids) < num_bodies:
                    body_mask = wp.array(
                        [body_index in body_ids for body_index in range(num_bodies)],
                        dtype=wp.bool,
                        device=art.device,
                    )
                art.set_coms_mask(
                    coms=wp.array(full_coms, dtype=wp.transformf, device=art.device),
                    env_mask=env_mask,
                    body_mask=body_mask,
                )

            backend_body_ids = user_to_backend[np.asarray(body_ids, dtype=np.int64)]
            selected_cells = np.ix_(env_ids, backend_body_ids)
            selected_mask = np.zeros((num_instances, num_bodies), dtype=bool)
            selected_mask[selected_cells] = True
            expected_backend[selected_cells] = payload

            backend_after = _read_backend_com_poses("ovphysx", art, raw_backend)
            np.testing.assert_array_equal(backend_after, expected_backend)
            np.testing.assert_array_equal(backend_after[~selected_mask], expected_before[~selected_mask])
            public_coms = art.data.body_com_pose_b.torch.detach().cpu().numpy()
            np.testing.assert_array_equal(public_coms, expected_backend[:, user_to_backend])

            if write_index == 0:
                first_backend_cells = selected_cells
                first_payload = payload.copy()

        assert backend_after is not None
        assert first_backend_cells is not None
        assert first_payload is not None
        np.testing.assert_array_equal(backend_after[first_backend_cells], first_payload)

    @pytest.mark.parametrize("backend", ["physx", "ovphysx"])
    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    def test_duplicate_body_ids_preserve_omitted_backend_com(self, backend: str, ordering_mode: str) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        if backend == "physx":
            raw_backend._noop_setters = False
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        _seed_backend_com_poses(backend, art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses(backend, art, raw_backend), backend_seed)

        if art.body_ordering is None:
            user_to_backend = np.arange(num_bodies, dtype=np.int64)
        else:
            user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
        duplicate_body_ids = [0, 1, 1, 2]
        omitted_public_body_id = 3
        env_ids = list(range(num_instances))
        payload = _make_selected_com_poses(env_ids, duplicate_body_ids, 3000.0)

        art.set_coms_index(
            coms=wp.array(payload, dtype=wp.transformf, device=art.device),
            body_ids=wp.array(duplicate_body_ids, dtype=wp.int32, device=art.device),
        )

        expected_backend = backend_seed.copy()
        for env_offset, env_id in enumerate(env_ids):
            for body_offset, public_body_id in enumerate(duplicate_body_ids):
                expected_backend[env_id, user_to_backend[public_body_id]] = payload[env_offset, body_offset]
        backend_after = _read_backend_com_poses(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_after, expected_backend)
        np.testing.assert_array_equal(
            backend_after[:, user_to_backend[omitted_public_body_id]],
            backend_seed[:, user_to_backend[omitted_public_body_id]],
        )
        public_after = art.data.body_com_pose_b.torch.detach().cpu().numpy()
        np.testing.assert_array_equal(public_after, expected_backend[:, user_to_backend])

    @pytest.mark.parametrize("backend", ["physx", "ovphysx"])
    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    def test_duplicate_env_ids_leave_global_com_cache_invalid(self, backend: str, ordering_mode: str) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        if backend == "physx":
            raw_backend._noop_setters = False
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        _seed_backend_com_poses(backend, art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses(backend, art, raw_backend), backend_seed)

        if art.body_ordering is None:
            user_to_backend = np.arange(num_bodies, dtype=np.int64)
        else:
            user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
        duplicate_env_ids = [1, 1]
        omitted_env_id = 0
        public_body_ids = list(range(num_bodies))
        payload = _make_selected_com_poses(duplicate_env_ids, public_body_ids, 4000.0)

        art.set_coms_index(
            coms=wp.array(payload, dtype=wp.transformf, device=art.device),
            env_ids=wp.array(duplicate_env_ids, dtype=wp.int32, device=art.device),
        )

        expected_backend = backend_seed.copy()
        for env_offset, env_id in enumerate(duplicate_env_ids):
            expected_backend[env_id, user_to_backend] = payload[env_offset]
        backend_after = _read_backend_com_poses(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_after, expected_backend)
        np.testing.assert_array_equal(backend_after[omitted_env_id], backend_seed[omitted_env_id])

        public_cache_was_invalid = art.data._body_com_pose_b.timestamp < 0.0
        backend_staging = art.data._body_com_pose_b_backend
        backend_cache_was_invalid = backend_staging is None or backend_staging.timestamp < 0.0
        public_after = art.data.body_com_pose_b.torch.detach().cpu().numpy()
        np.testing.assert_array_equal(public_after, expected_backend[:, user_to_backend])
        assert public_cache_was_invalid
        assert backend_cache_was_invalid

    @pytest.mark.parametrize("backend", ["physx", "ovphysx"])
    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    def test_static_com_cache_does_not_follow_sim_timestamp(self, backend: str, ordering_mode: str) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        root_pose = np.zeros((num_instances, 7), dtype=np.float32)
        root_pose[:, :3] = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        root_pose[:, 6] = 1.0
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        if backend == "physx":
            raw_backend._root_transforms = wp.array(root_pose, dtype=wp.float32, device=art.device)
        else:
            from isaaclab_ovphysx import tensor_types as TT

            raw_backend.bindings[TT.ROOT_POSE]._data = root_pose.copy()
        _seed_backend_com_poses(backend, art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses(backend, art, raw_backend), backend_seed)
        art.data._body_com_pose_b.timestamp = -1.0
        backend_staging = art.data._body_com_pose_b_backend
        if backend_staging is not None:
            backend_staging.timestamp = -1.0

        if art.body_ordering is None:
            user_to_backend = np.arange(num_bodies, dtype=np.int64)
        else:
            user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
        expected_public = backend_seed[:, user_to_backend]
        expected_root = root_pose.copy()
        expected_root[:, :3] += backend_seed[:, 0, :3]

        if backend == "physx":
            raw_backend.get_coms = MagicMock(wraps=raw_backend.get_coms)
            read_spy = raw_backend.get_coms
        else:
            com_binding = raw_backend.bindings[TT.BODY_COM_POSE]
            com_binding.read = MagicMock(wraps=com_binding.read)
            read_spy = com_binding.read

        np.testing.assert_array_equal(art.data.body_com_pose_b.torch.detach().cpu().numpy(), expected_public)
        np.testing.assert_array_equal(art.data.root_com_pose_w.torch.detach().cpu().numpy(), expected_root)
        assert read_spy.call_count == 1

        for _ in range(3):
            art.data.update(dt=0.01)
            np.testing.assert_array_equal(art.data.body_com_pose_b.torch.detach().cpu().numpy(), expected_public)
            np.testing.assert_array_equal(art.data.root_com_pose_w.torch.detach().cpu().numpy(), expected_root)
            assert read_spy.call_count == 1


# ---------------------------------------------------------------------------
# Tests: ArticulationData joint state and properties
# ---------------------------------------------------------------------------


class TestArticulationOrderingRootWriteParity:
    """Test root writers against backend-order COM offsets."""

    @pytest.mark.parametrize("operation", ["com_pose", "link_velocity"])
    @pytest.mark.parametrize(
        "backend, selection",
        [
            pytest.param("physx", "index", id="physx-index"),
            pytest.param("physx", "mask", id="physx-mask"),
            pytest.param("ovphysx", "index", id="ovphysx-index"),
            pytest.param("ovphysx", "mask", id="ovphysx-mask"),
            pytest.param("newton", "index", id="newton-index"),
            pytest.param("newton", "mask", id="newton-mask"),
        ],
    )
    def test_floating_root_writers_match_identity_after_body_reordering(
        self,
        backend: str,
        selection: str,
        operation: str,
    ) -> None:
        if backend not in BACKENDS:
            pytest.skip(f"{backend} backend is not available")
        num_instances = 2
        num_bodies = 4
        backend_body_names = tuple(f"body_{index}" for index in range(num_bodies))
        identity_art, identity_raw = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            is_fixed_base=False,
            body_ordering=backend_body_names,
        )
        ordered_art, ordered_raw = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            is_fixed_base=False,
            body_ordering=tuple(reversed(backend_body_names)),
        )
        assert identity_art.body_ordering is not None and identity_art.body_ordering.is_identity
        assert ordered_art.body_ordering is not None and not ordered_art.body_ordering.is_identity
        assert ordered_art.body_ordering.backend_to_user_indices[0] != 0
        if backend == "physx":
            identity_raw._noop_setters = False
            ordered_raw._noop_setters = False

        backend_coms = _make_backend_com_poses(num_instances, num_bodies)
        _seed_root_writer_backend_com(backend, identity_art, identity_raw, backend_coms)
        _seed_root_writer_backend_com(backend, ordered_art, ordered_raw, backend_coms)

        root_com_pose = np.asarray(
            [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0], [14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        root_link_velocity = np.asarray(
            [[1.0, 2.0, 3.0, 0.4, 0.5, 0.6], [11.0, 12.0, 13.0, 1.4, 1.5, 1.6]],
            dtype=np.float32,
        )
        _write_root_writer_command(identity_art, selection, operation, root_com_pose, root_link_velocity)
        _write_root_writer_command(ordered_art, selection, operation, root_com_pose, root_link_velocity)

        identity_root_link_pose, identity_root_com_velocity = _read_backend_root_state(
            backend, identity_art, identity_raw
        )
        ordered_root_link_pose, ordered_root_com_velocity = _read_backend_root_state(backend, ordered_art, ordered_raw)
        if operation == "com_pose":
            np.testing.assert_allclose(ordered_root_link_pose[0], identity_root_link_pose[0], atol=1e-6)
        else:
            np.testing.assert_allclose(ordered_root_com_velocity[0], identity_root_com_velocity[0], atol=1e-6)


class TestArticulationOrderingWriteParity:
    """Test that partial joint writes preserve unselected backend state."""

    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    @pytest.mark.parametrize(
        "coverage",
        ["one_env_one_item", "all_envs_one_item", "one_env_all_items"],
    )
    def test_physx_partial_joint_write_preserves_backend_rows(
        self,
        ordering_mode: str,
        selection: str,
        operation: str,
        coverage: str,
    ) -> None:
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        _exercise_partial_joint_write("physx", ordering_mode, selection, operation, coverage)

    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    @pytest.mark.parametrize(
        "coverage",
        ["one_env_one_item", "all_envs_one_item", "one_env_all_items"],
    )
    def test_ovphysx_partial_joint_write_preserves_backend_rows(
        self,
        ordering_mode: str,
        selection: str,
        operation: str,
        coverage: str,
    ) -> None:
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        _exercise_partial_joint_write("ovphysx", ordering_mode, selection, operation, coverage)

    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    @pytest.mark.parametrize(
        "coverage",
        ["one_env_one_item", "all_envs_one_item", "one_env_all_items"],
    )
    def test_newton_partial_joint_write_preserves_backend_columns(
        self,
        ordering_mode: str,
        selection: str,
        operation: str,
        coverage: str,
    ) -> None:
        if "newton" not in BACKENDS:
            pytest.skip("Newton backend is not available")
        _exercise_partial_joint_write("newton", ordering_mode, selection, operation, coverage)

    @pytest.mark.parametrize(
        "operation, expected_position_refreshes, expected_velocity_refreshes",
        [("position", 1, 0), ("velocity", 0, 1), ("state", 1, 1)],
    )
    def test_physx_mask_partial_joint_write_refreshes_each_staging_row_once(
        self,
        operation: str,
        expected_position_refreshes: int,
        expected_velocity_refreshes: int,
    ) -> None:
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        art, raw_backend = get_articulation(
            "physx",
            num_instances=2,
            num_joints=3,
            num_bodies=2,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode("reversed", 3),
        )
        initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
        _seed_backend_joint_state("physx", art, raw_backend, initial_position, initial_velocity)
        position_refresh = MagicMock(wraps=art.data._refresh_joint_pos)
        velocity_refresh = MagicMock(wraps=art.data._refresh_joint_vel)
        art.data._refresh_joint_pos = position_refresh
        art.data._refresh_joint_vel = velocity_refresh

        env_ids, joint_ids, position_payload, velocity_payload = _write_selected_joint_state(
            art,
            "physx",
            "mask",
            operation,
            "one_env_one_item",
            901.0,
            902.0,
        )

        assert position_refresh.call_count == expected_position_refreshes
        assert velocity_refresh.call_count == expected_velocity_refreshes
        user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
        selected_cells = np.ix_(env_ids, user_to_backend[joint_ids])
        if operation in ("position", "state"):
            initial_position[selected_cells] = position_payload
        if operation in ("velocity", "state"):
            initial_velocity[selected_cells] = velocity_payload
        backend_position, backend_velocity = _read_backend_joint_state("physx", art, raw_backend)
        np.testing.assert_array_equal(backend_position, initial_position)
        np.testing.assert_array_equal(backend_velocity, initial_velocity)

    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_physx_duplicate_full_length_joint_selector_preserves_newer_backend_rows(self, operation: str) -> None:
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        _exercise_duplicate_full_length_joint_write("physx", operation)

    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_ovphysx_duplicate_full_length_joint_selector_preserves_newer_backend_rows(self, operation: str) -> None:
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        _exercise_duplicate_full_length_joint_write("ovphysx", operation)

    @pytest.mark.parametrize("selector", ["default_mask", "explicit_full_index"])
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_physx_joint_selector_provenance_controls_staging_refresh(self, operation: str, selector: str) -> None:
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        art, raw_backend = get_articulation(
            "physx",
            num_instances=2,
            num_joints=3,
            num_bodies=2,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode("reversed", 3),
        )
        initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
        _seed_backend_joint_state("physx", art, raw_backend, initial_position, initial_velocity)
        position_refresh = MagicMock(wraps=art.data._refresh_joint_pos)
        velocity_refresh = MagicMock(wraps=art.data._refresh_joint_vel)
        art.data._refresh_joint_pos = position_refresh
        art.data._refresh_joint_vel = velocity_refresh

        position = torch.tensor(initial_position[:, ::-1].copy(), dtype=torch.float32, device=art.device)
        velocity = torch.tensor(initial_velocity[:, ::-1].copy(), dtype=torch.float32, device=art.device)
        if selector == "default_mask":
            if operation == "position":
                art.write_joint_position_to_sim_mask(position=position)
            elif operation == "velocity":
                art.write_joint_velocity_to_sim_mask(velocity=velocity)
            else:
                art.write_joint_state_to_sim_mask(position=position, velocity=velocity)
        else:
            joint_ids = wp.array([0, 1, 2], dtype=wp.int32, device=art.device)
            if operation == "position":
                art.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids)
            elif operation == "velocity":
                art.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids)
            else:
                art.write_joint_state_to_sim_index(position=position, velocity=velocity, joint_ids=joint_ids)

        should_refresh = selector == "explicit_full_index"
        assert position_refresh.call_count == int(should_refresh and operation in ("position", "state"))
        assert velocity_refresh.call_count == int(should_refresh and operation in ("velocity", "state"))

    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_physx_stale_partial_joint_write_preserves_newer_backend_rows(self, operation: str) -> None:
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        _exercise_stale_partial_joint_write("physx", operation)

    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_ovphysx_stale_partial_joint_write_preserves_newer_backend_rows(self, operation: str) -> None:
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        _exercise_stale_partial_joint_write("ovphysx", operation)


class TestArticulationDataJointState:
    """Test data properties for joint state and joint properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_pos(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_pos, expected_shape=(num_instances, num_joints), expected_dtype=wp.float32, name="joint_pos"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_vel(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_vel, expected_shape=(num_instances, num_joints), expected_dtype=wp.float32, name="joint_vel"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_acc(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_acc, expected_shape=(num_instances, num_joints), expected_dtype=wp.float32, name="joint_acc"
        )

    def test_ovphysx_joint_acceleration_differences_public_order_velocities(self):
        """Finite-difference OVPhysX joint velocity entirely in public joint order."""
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

    def test_ovphysx_ordered_joint_state_is_cached_per_sim_timestamp(self):
        """Gather ordered OVPhysX joint position and velocity at most once per timestamp."""
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        art, _ = get_articulation(
            "ovphysx",
            2,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        art.data.update(0.01)

        art.data.joint_pos.torch.clone()
        art.data.joint_vel.torch.clone()

        assert art.data._joint_pos_buf.timestamp == art.data._sim_timestamp
        assert art.data._joint_vel_buf.timestamp == art.data._sim_timestamp

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 3, 2)])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_reversed_joint_ordering_reorders_public_joint_properties(
        self, backend, num_instances, num_joints, num_bodies, device
    ):
        """Expose every backend joint property under the matching public joint name."""
        joint_ordering = tuple(f"joint_{index}" for index in reversed(range(num_joints)))
        art, raw_backend = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device=device,
            joint_ordering=joint_ordering,
        )
        user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
        backend_properties = _get_backend_joint_property_tensors(backend, art, raw_backend)
        public_properties = {
            "stiffness": art.data.joint_stiffness,
            "damping": art.data.joint_damping,
            "armature": art.data.joint_armature,
            "position_limits": art.data.joint_pos_limits,
            "velocity_limits": art.data.joint_vel_limits,
            "effort_limits": art.data.joint_effort_limits,
            "friction": art.data.joint_friction_coeff,
        }

        for property_name, public_property in public_properties.items():
            _assert_proxy_close(public_property, backend_properties[property_name][:, user_to_backend])

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_stiffness(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_stiffness,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_stiffness",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_damping(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_damping,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_damping",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_armature(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_armature,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_armature",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_friction_coeff(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_friction_coeff,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_friction_coeff",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_pos_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_pos_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.vec2f,
            name="joint_pos_limits",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_vel_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_vel_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_vel_limits",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_effort_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_effort_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_effort_limits",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_soft_joint_pos_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.soft_joint_pos_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.vec2f,
            name="soft_joint_pos_limits",
        )


# ---------------------------------------------------------------------------
# Tests: ArticulationData defaults and command targets
# ---------------------------------------------------------------------------


class TestArticulationDataDefaults:
    """Test default state and command target properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_pose(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_root_pose,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="default_root_pose",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_vel(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_root_vel,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="default_root_vel",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_joint_pos(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_joint_pos,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="default_joint_pos",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_joint_vel(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_joint_vel,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="default_joint_vel",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_pos_target(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_pos_target,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_pos_target",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_vel_target(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_vel_target,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_vel_target",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_effort_target(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_effort_target,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_effort_target",
        )


# ---------------------------------------------------------------------------
# Writer/setter test helpers
# ---------------------------------------------------------------------------

# Map warp structured dtypes to their torch trailing dimension size.
_WP_DTYPE_TO_TRAILING = {
    wp.transformf: 7,
    wp.spatial_vectorf: 6,
    wp.vec2f: 2,
    wp.float32: 0,  # no trailing dimension
}


def _make_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create valid torch test data for a given warp dtype.

    For transformf shapes, appends a trailing dim of 7 and sets quat w=1.
    For spatial_vectorf, appends trailing 6.
    For vec2f, appends trailing 2 with [-1, 1].
    For float32, no trailing dim.
    """
    trailing = _WP_DTYPE_TO_TRAILING[wp_dtype]
    if trailing:
        full_shape = (*shape, trailing)
    else:
        full_shape = shape
    data = torch.zeros(full_shape, device=device, dtype=torch.float32)
    if wp_dtype == wp.transformf:
        data[..., 6] = 1.0  # identity quat w
    elif wp_dtype == wp.vec2f:
        data[..., 0] = -1.0
        data[..., 1] = 1.0
    elif wp_dtype == wp.float32:
        data.fill_(1.0)
    return data


def _make_data_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create valid warp test data for a given warp dtype.

    Warp structured types collapse the trailing dim into the dtype,
    so a (N,) transformf array is equivalent to (N, 7) float32 in torch.
    """
    t = _make_data_torch(shape, device, wp_dtype)
    if wp_dtype == wp.float32:
        return wp.from_torch(t, dtype=wp.float32)
    # For structured types, the torch tensor has the trailing dim; convert to warp
    return wp.from_torch(t.contiguous(), dtype=wp_dtype)


def _make_bad_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create torch data with wrong leading shape for negative testing.

    Adds +1 to the first dimension so the shape doesn't match.
    """
    bad_shape = (shape[0] + 1,) + shape[1:]
    return _make_data_torch(bad_shape, device, wp_dtype)


def _make_bad_data_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create warp data with wrong leading shape for negative testing."""
    bad_shape = (shape[0] + 1,) + shape[1:]
    return _make_data_warp(bad_shape, device, wp_dtype)


def _make_env_mask(num_instances: int, device: str, partial: bool) -> wp.array | None:
    """Create an env_mask: None for all envs, or a partial bool mask."""
    if not partial:
        return None
    mask_np = np.zeros(num_instances, dtype=bool)
    mask_np[0] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


def _make_env_ids(device: str, subset: bool) -> torch.Tensor | None:
    """Create env_ids: None for all envs, or [0] for a subset."""
    if not subset:
        return None
    return torch.tensor([0], dtype=torch.int32, device=device)


def _make_item_mask(total: int, selected: list[int], device: str) -> wp.array:
    """Create a bool warp mask with True at `selected` indices, False elsewhere."""
    mask_np = np.zeros(total, dtype=bool)
    for i in selected:
        mask_np[i] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


# ---------------------------------------------------------------------------
# Tests: Articulation operations
# ---------------------------------------------------------------------------


class TestArticulationOperations:
    """Test cross-cutting articulation operations."""

    @_non_mock_backends
    @pytest.mark.parametrize("with_body_ordering", [False, True], ids=["none", "ordered"])
    @pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_external_wrenches_are_written_in_backend_body_order(
        self, backend, with_body_ordering, is_fixed_base, device
    ):
        """Write public-order body wrenches to each backend in backend body order."""
        num_instances, num_joints, num_bodies = 2, 1, 4
        backend_body_names = tuple(f"body_{index}" for index in range(num_bodies))
        body_ordering = None
        if with_body_ordering:
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
        object.__setattr__(art, "_instantaneous_wrench_composer", WrenchComposer(art))
        object.__setattr__(art, "_permanent_wrench_composer", WrenchComposer(art))
        captured = {}
        if backend == "physx":

            def capture_wrench(*, force_data, torque_data, position_data, indices, is_global):
                captured["force"] = force_data.numpy().reshape(num_instances, num_bodies, 3).copy()
                captured["torque"] = torque_data.numpy().reshape(num_instances, num_bodies, 3).copy()

            raw_backend.apply_forces_and_torques_at_position = capture_wrench

        forces = np.arange(num_instances * num_bodies * 3, dtype=np.float32).reshape(num_instances, num_bodies, 3)
        torques = forces + 100.0
        art.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=wp.array(forces, dtype=wp.vec3f, device=device),
            torques=wp.array(torques, dtype=wp.vec3f, device=device),
        )

        art.write_data_to_sim()

        backend_to_user = (
            np.arange(num_bodies, dtype=np.int64)
            if art.body_ordering is None
            else np.asarray(art.body_ordering.backend_to_user_indices, dtype=np.int64)
        )
        backend_force, backend_torque = _read_backend_wrench(backend, art, raw_backend, captured)
        np.testing.assert_allclose(backend_force, forces[:, backend_to_user])
        np.testing.assert_allclose(backend_torque, torques[:, backend_to_user])

    def test_physx_none_ordering_allocates_no_external_wrench_staging_buffers(self):
        """Keep the default PhysX wrench path free of reorder staging allocations."""
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        art, _ = get_articulation("physx", 2, 1, 4, device="cpu")

        assert art.body_ordering is None
        assert art._body_wrench_force_backend is None
        assert art._body_wrench_torque_backend is None

    def test_ovphysx_configured_defaults_use_public_joint_names(self):
        """Resolve configured OVPhysX defaults against public joint names."""
        if "ovphysx" not in BACKENDS:
            pytest.skip("OVPhysX backend is not available")
        art, _ = get_articulation(
            "ovphysx",
            1,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        zeros = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
        art.data._default_joint_pos.assign(zeros)
        art.data._default_joint_vel.assign(zeros)
        patterns = {"joint_0": 10.0, "joint_1": 20.0, "joint_2": 30.0}

        art._resolve_joint_values(patterns, art.data._default_joint_pos)

        expected = np.asarray([[patterns[name] for name in art.joint_names]], dtype=np.float32)
        np.testing.assert_array_equal(art.data.default_joint_pos.warp.numpy(), expected)

    def test_physx_newton_actuator_forces_are_written_in_backend_order(self):
        """Write Newton-actuator PhysX forces in backend joint order."""
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        num_instances = 2
        num_joints = 4
        num_bodies = 2
        art, raw_backend = get_articulation("physx", num_instances, num_joints, num_bodies, device="cpu")
        _install_reversed_joint_ordering(art)
        user_forces_np = np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
        user_forces = wp.array(user_forces_np, dtype=wp.float32, device=art.device)
        object.__setattr__(art, "_joint_effort_target_backend", wp.zeros_like(art.data.joint_effort_target.warp))
        wrapper = MagicMock()
        wrapper.joint_f_2d = user_forces
        object.__setattr__(art, "_physx_actuator_wrapper", wrapper)
        object.__setattr__(art, "_has_newton_actuators", True)
        object.__setattr__(art, "_has_implicit_actuators", False)
        art._apply_actuator_model_newton = MagicMock()
        captured = {}

        def _capture_forces(forces, indices):
            captured["forces"] = wp.clone(forces, device="cpu").numpy()
            captured["indices"] = wp.clone(indices, device="cpu").numpy()

        raw_backend.set_dof_actuation_forces = _capture_forces

        art.write_data_to_sim()

        backend_to_user = np.asarray(art.joint_ordering.backend_to_user_indices, dtype=np.int64)
        np.testing.assert_allclose(captured["forces"], user_forces_np[:, backend_to_user])

    @pytest.mark.parametrize(
        ("method_name", "value_name", "controller_attr"),
        [
            ("write_actuator_stiffness_to_sim", "stiffness", "kp"),
            ("write_actuator_damping_to_sim", "damping", "kd"),
        ],
    )
    def test_physx_newton_actuator_gain_updates_use_public_joint_ids(
        self, method_name: str, value_name: str, controller_attr: str
    ):
        """Route PhysX Newton-actuator gain updates by public joint ID."""
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        art, _ = get_articulation(
            "physx",
            1,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        controller = MagicMock(
            kp=wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"),
            kd=wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"),
        )
        actuator = MagicMock(
            controller=controller,
            indices=wp.array([0, 1, 2], dtype=wp.uint32, device="cpu"),
        )
        adapter = MagicMock(actuators=[actuator])
        object.__setattr__(art, "newton_actuator_adapter", adapter)

        getattr(art, method_name)(
            **{
                value_name: torch.tensor([[99.0]], dtype=torch.float32),
                "env_ids": torch.tensor([0], dtype=torch.int32),
                "joint_ids": torch.tensor([0], dtype=torch.int32),
            }
        )

        np.testing.assert_array_equal(
            getattr(controller, controller_attr).numpy(),
            np.asarray([99.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_physx_validate_cfg_reports_velocity_limits_in_public_joint_order(self):
        """Pair public default velocities with limits for the same named joint."""
        if "physx" not in BACKENDS:
            pytest.skip("PhysX backend is not available")
        art, raw_backend = get_articulation(
            "physx",
            1,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        backend_velocity_limits = np.asarray([[1.0, 100.0, 100.0]], dtype=np.float32)
        public_velocity_limits = backend_velocity_limits[:, [2, 1, 0]]
        raw_backend.set_mock_dof_max_velocities(wp.array(backend_velocity_limits, dtype=wp.float32, device="cpu"))
        art.data._joint_vel_limits.assign(wp.array(public_velocity_limits, dtype=wp.float32, device="cpu"))
        art.data._joint_pos_limits.assign(wp.array(np.tile((-100.0, 100.0), (1, 3, 1)), dtype=wp.vec2f, device="cpu"))
        art.data._default_joint_pos.assign(wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
        art.data._default_joint_vel.assign(
            wp.array(np.asarray([[50.0, 50.0, 2.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
        )

        with pytest.raises(ValueError, match=r"'joint_0': 2\.000 not in \[-1\.000, 1\.000\]"):
            art._validate_cfg()


# ---------------------------------------------------------------------------
# Tests: Root writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

_ROOT_POSE_METHODS = ["root_pose", "root_link_pose", "root_com_pose"]
_ROOT_VEL_METHODS = ["root_velocity", "root_link_velocity", "root_com_velocity"]


class TestArticulationWritersRoot:
    """Test root pose/velocity writers with all input combinations."""

    # -- index variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_index(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, subset
        method(root_pose=_make_data_torch((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # warp, all envs
        method(root_pose=_make_data_warp((num_instances,), device, wp.transformf))
        # warp, subset
        method(root_pose=_make_data_warp((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_index(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, subset
        method(root_velocity=_make_data_torch((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # warp, all envs
        method(root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf))
        # warp, subset
        method(root_velocity=_make_data_warp((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_torch((num_instances,), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_warp((num_instances,), device, wp.spatial_vectorf))

    # -- mask variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_mask(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask (all)
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, partial mask
        method(
            root_pose=_make_data_torch((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask
        method(root_pose=_make_data_warp((num_instances,), device, wp.transformf))
        # warp, partial mask
        method(
            root_pose=_make_data_warp((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_mask(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, partial mask
        method(
            root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask
        method(root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf))
        # warp, partial mask
        method(
            root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_torch((num_instances,), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_warp((num_instances,), device, wp.spatial_vectorf))


# ---------------------------------------------------------------------------
# Tests: Joint writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, accepts_float)
_JOINT_METHODS = [
    ("write_joint_position_to_sim", "position", wp.float32, False),
    ("write_joint_velocity_to_sim", "velocity", wp.float32, False),
    ("write_joint_stiffness_to_sim", "stiffness", wp.float32, True),
    ("write_joint_damping_to_sim", "damping", wp.float32, True),
    ("write_joint_position_limit_to_sim", "limits", wp.vec2f, True),
    ("write_joint_velocity_limit_to_sim", "limits", wp.float32, True),
    ("write_joint_effort_limit_to_sim", "limits", wp.float32, True),
    ("write_joint_armature_to_sim", "armature", wp.float32, True),
    ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", wp.float32, False),
    ("set_joint_position_target", "target", wp.float32, False),
    ("set_joint_velocity_target", "target", wp.float32, False),
    ("set_joint_effort_target", "target", wp.float32, False),
]


class TestArticulationWritersJoint:
    """Test joint writers/setters with all input combinations."""

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _JOINT_METHODS,
        ids=[m[0] for m in _JOINT_METHODS],
    )
    def test_joint_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_j = min(2, num_joints)
        sub_joint_ids = list(range(sub_j))

        # torch, all envs + all joints
        method(**{kwarg: _make_data_torch((num_instances, num_joints), device, wp_dtype)})
        # torch, subset envs + subset joints
        method(
            **{
                kwarg: _make_data_torch((1, sub_j), device, wp_dtype),
                "joint_ids": sub_joint_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all joints
        method(**{kwarg: _make_data_warp((num_instances, num_joints), device, wp_dtype)})
        # warp, subset
        method(
            **{
                kwarg: _make_data_warp((1, sub_j), device, wp_dtype),
                "joint_ids": sub_joint_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f position_limit)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # float scalar for vec2f position_limit should raise ValueError
        if accepts_float and wp_dtype == wp.vec2f:
            with pytest.raises((ValueError, TypeError)):
                method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_joints), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_joints), device, wp_dtype)})

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _JOINT_METHODS,
        ids=[m[0] for m in _JOINT_METHODS],
    )
    def test_joint_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_joint_sel = list(range(min(2, num_joints)))

        # torch, no mask
        method(**{kwarg: _make_data_torch((num_instances, num_joints), device, wp_dtype)})
        # torch, partial env_mask + joint_mask
        method(
            **{
                kwarg: _make_data_torch((num_instances, num_joints), device, wp_dtype),
                "joint_mask": _make_item_mask(num_joints, sub_joint_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_data_warp((num_instances, num_joints), device, wp_dtype)})
        # warp, partial env_mask + joint_mask
        method(
            **{
                kwarg: _make_data_warp((num_instances, num_joints), device, wp_dtype),
                "joint_mask": _make_item_mask(num_joints, sub_joint_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_joints), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_joints), device, wp_dtype)})


# ---------------------------------------------------------------------------
# Tests: Body writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, trailing_dim)
_BODY_METHODS = [
    ("set_masses", "masses", wp.float32, 0),
    ("set_coms", "coms", wp.transformf, 7),
    ("set_inertias", "inertias", wp.float32, 9),
]


class TestArticulationWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        trailing,
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")

        # For inertias, the shape is (N, B, 9) always (no structured warp type)
        # For coms, torch shape is (N, B, 7), warp shape is (N, B) transformf
        # For masses, shape is (N, B)

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _warp_shape(n_envs, n_bods):
            return (n_envs, n_bods)

        def _make_torch(n_envs, n_bods):
            shape = _torch_shape(n_envs, n_bods)
            data = torch.ones(shape, device=device, dtype=torch.float32)
            if wp_dtype == wp.transformf:
                data[..., :3] = 0.0
                data[..., 3:6] = 0.0
                data[..., 6] = 1.0
            return data

        def _make_warp(n_envs, n_bods):
            t = _make_torch(n_envs, n_bods)
            if wp_dtype == wp.transformf:
                return wp.from_torch(t.contiguous(), dtype=wp.transformf)
            return wp.from_torch(t.contiguous(), dtype=wp.float32)

        sub_b = min(2, num_bodies)
        sub_body_ids = list(range(sub_b))

        # torch, all envs + all bodies
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, subset
        method(
            **{
                kwarg: _make_torch(1, sub_b),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all bodies
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, subset
        method(
            **{
                kwarg: _make_warp(1, sub_b),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # negative: bad torch shape (extra env)
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_torch(num_instances + 1, num_bodies)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_warp(num_instances + 1, num_bodies)})

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        trailing,
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _make_torch(n_envs, n_bods):
            shape = _torch_shape(n_envs, n_bods)
            data = torch.ones(shape, device=device, dtype=torch.float32)
            if wp_dtype == wp.transformf:
                data[..., :3] = 0.0
                data[..., 3:6] = 0.0
                data[..., 6] = 1.0
            return data

        def _make_warp(n_envs, n_bods):
            t = _make_torch(n_envs, n_bods)
            if wp_dtype == wp.transformf:
                return wp.from_torch(t.contiguous(), dtype=wp.transformf)
            return wp.from_torch(t.contiguous(), dtype=wp.float32)

        sub_body_sel = list(range(min(2, num_bodies)))

        # torch, no mask
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_torch(num_instances, num_bodies),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_warp(num_instances, num_bodies),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_torch(num_instances + 1, num_bodies)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_warp(num_instances + 1, num_bodies)})


# ---------------------------------------------------------------------------
# Tests: Alias/shorthand properties
# ---------------------------------------------------------------------------


class TestArticulationDataAliases:
    """Test that alias properties return the same shape/dtype as their canonical counterparts."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_aliases(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        """root_pose_w == root_link_pose_w, root_vel_w == root_com_vel_w, etc."""
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        d = art.data

        assert d.root_pose_w.shape == d.root_link_pose_w.shape
        assert d.root_pose_w.dtype == d.root_link_pose_w.dtype
        assert d.root_pos_w.shape == d.root_link_pos_w.shape
        assert d.root_quat_w.shape == d.root_link_quat_w.shape

        assert d.root_vel_w.shape == d.root_com_vel_w.shape
        assert d.root_vel_w.dtype == d.root_com_vel_w.dtype
        assert d.root_lin_vel_w.shape == d.root_com_lin_vel_w.shape
        assert d.root_ang_vel_w.shape == d.root_com_ang_vel_w.shape

    @_backends
    @_default_dims
    @_default_devices
    def test_body_aliases(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        d = art.data

        assert d.body_pose_w.shape == d.body_link_pose_w.shape
        assert d.body_pos_w.shape == d.body_link_pos_w.shape
        assert d.body_quat_w.shape == d.body_link_quat_w.shape
        assert d.body_vel_w.shape == d.body_com_vel_w.shape
        assert d.body_lin_vel_w.shape == d.body_com_lin_vel_w.shape
        assert d.body_ang_vel_w.shape == d.body_com_ang_vel_w.shape

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_aliases(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        d = art.data

        assert d.joint_limits.shape == d.joint_pos_limits.shape
        assert d.joint_friction.shape == d.joint_friction_coeff.shape


# ---------------------------------------------------------------------------
# Tendon tests — parametrize, properties, finders, data, writers
# ---------------------------------------------------------------------------

# Newton does not support tendons (always 0), so exclude it from tendon tests.
_tendon_backends = pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "newton"], indirect=False)

_tendon_dims = pytest.mark.parametrize(
    "num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons",
    [
        (1, 2, 2, 1, 0),  # fixed only
        (2, 6, 7, 3, 2),  # both types
        (100, 8, 13, 4, 3),  # large, both types
    ],
)


class TestArticulationTendonProperties:
    """Test that tendon-related articulation properties return the correct types/values."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_num_fixed_tendons(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        assert art.num_fixed_tendons == num_fixed_tendons

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_num_spatial_tendons(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        assert art.num_spatial_tendons == num_spatial_tendons

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_names(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        names = art.fixed_tendon_names
        assert isinstance(names, list)
        assert len(names) == num_fixed_tendons
        assert all(isinstance(n, str) for n in names)

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_names(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        names = art.spatial_tendon_names
        assert isinstance(names, list)
        assert len(names) == num_spatial_tendons
        assert all(isinstance(n, str) for n in names)


class TestArticulationTendonFinders:
    """Test that tendon finder methods return (list[int], list[str]) tuples."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_fixed_tendons_all(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        indices, names = art.find_fixed_tendons(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_fixed_tendons
        assert len(names) == num_fixed_tendons
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_fixed_tendons_single(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        first = art.fixed_tendon_names[0]
        indices, names = art.find_fixed_tendons(first)
        assert indices == [0]
        assert names == [first]

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_spatial_tendons_all(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        indices, names = art.find_spatial_tendons(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_spatial_tendons
        assert len(names) == num_spatial_tendons

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_spatial_tendons_single(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        first = art.spatial_tendon_names[0]
        indices, names = art.find_spatial_tendons(first)
        assert indices == [0]
        assert names == [first]


class TestArticulationDataTendonState:
    """Test data properties for tendon state (fixed and spatial)."""

    # -- Fixed tendon data properties --

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_stiffness,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_damping(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_damping,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_damping",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_limit_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_limit_stiffness,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_limit_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_rest_length(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_rest_length,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_rest_length",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_offset(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_offset,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_offset",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_pos_limits(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        from isaaclab.utils.warp import ProxyArray

        arr = art.data.fixed_tendon_pos_limits
        assert isinstance(arr, ProxyArray), f"fixed_tendon_pos_limits: expected ProxyArray, got {type(arr)}"
        if num_fixed_tendons == 0:
            # When no tendons, shape is (N, 0, 2) float32
            assert arr.shape == (num_instances, 0, 2)
            assert arr.dtype == wp.float32
        else:
            # PhysX returns (N, T, 2) float32; Mock returns (N, T) vec2f
            assert arr.shape in ((num_instances, num_fixed_tendons), (num_instances, num_fixed_tendons, 2))
            assert arr.dtype in (wp.vec2f, wp.float32)

    # -- Spatial tendon data properties --

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_stiffness,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_damping(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_damping,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_damping",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_limit_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_limit_stiffness,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_limit_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_offset(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_offset,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_offset",
        )


# ---------------------------------------------------------------------------
# Tests: Fixed tendon writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, accepts_float)
_FIXED_TENDON_METHODS = [
    ("set_fixed_tendon_stiffness", "stiffness", wp.float32, True),
    ("set_fixed_tendon_damping", "damping", wp.float32, True),
    ("set_fixed_tendon_limit_stiffness", "limit_stiffness", wp.float32, True),
    ("set_fixed_tendon_rest_length", "rest_length", wp.float32, True),
    ("set_fixed_tendon_offset", "offset", wp.float32, True),
]
# Note: set_fixed_tendon_position_limit is excluded because the PhysX backend stores
# pos_limits as (N, T, 2) float32 while the setter validates (N, T) float32. This data
# layout mismatch prevents consistent testing across mock and PhysX backends.


class TestArticulationWritersFixedTendon:
    """Test fixed tendon writers/setters with all input combinations."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _FIXED_TENDON_METHODS,
        ids=[m[0] for m in _FIXED_TENDON_METHODS],
    )
    def test_fixed_tendon_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_t = min(2, num_fixed_tendons)
        sub_tendon_ids = list(range(sub_t))

        # torch, all envs + all tendons
        method(**{kwarg: _make_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # torch, subset envs + subset tendons
        method(
            **{
                kwarg: _make_data_torch((1, sub_t), device, wp_dtype),
                "fixed_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all tendons
        method(**{kwarg: _make_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})
        # warp, subset
        method(
            **{
                kwarg: _make_data_warp((1, sub_t), device, wp_dtype),
                "fixed_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # float scalar for vec2f should raise ValueError
        if accepts_float and wp_dtype == wp.vec2f:
            with pytest.raises((ValueError, TypeError)):
                method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _FIXED_TENDON_METHODS,
        ids=[m[0] for m in _FIXED_TENDON_METHODS],
    )
    def test_fixed_tendon_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_tendon_sel = list(range(min(2, num_fixed_tendons)))

        # torch, no mask
        method(**{kwarg: _make_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # torch, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_torch((num_instances, num_fixed_tendons), device, wp_dtype),
                "fixed_tendon_mask": _make_item_mask(num_fixed_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})
        # warp, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_warp((num_instances, num_fixed_tendons), device, wp_dtype),
                "fixed_tendon_mask": _make_item_mask(num_fixed_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})


# ---------------------------------------------------------------------------
# Tests: Spatial tendon writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

_SPATIAL_TENDON_METHODS = [
    ("set_spatial_tendon_stiffness", "stiffness", wp.float32, True),
    ("set_spatial_tendon_damping", "damping", wp.float32, True),
    ("set_spatial_tendon_limit_stiffness", "limit_stiffness", wp.float32, True),
    ("set_spatial_tendon_offset", "offset", wp.float32, True),
]


class TestArticulationWritersSpatialTendon:
    """Test spatial tendon writers/setters with all input combinations."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _SPATIAL_TENDON_METHODS,
        ids=[m[0] for m in _SPATIAL_TENDON_METHODS],
    )
    def test_spatial_tendon_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_t = min(2, num_spatial_tendons)
        sub_tendon_ids = list(range(sub_t))

        # torch, all envs + all tendons
        method(**{kwarg: _make_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # torch, subset envs + subset tendons
        method(
            **{
                kwarg: _make_data_torch((1, sub_t), device, wp_dtype),
                "spatial_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all tendons
        method(**{kwarg: _make_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})
        # warp, subset
        method(
            **{
                kwarg: _make_data_warp((1, sub_t), device, wp_dtype),
                "spatial_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # float scalar
        if accepts_float:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _SPATIAL_TENDON_METHODS,
        ids=[m[0] for m in _SPATIAL_TENDON_METHODS],
    )
    def test_spatial_tendon_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_tendon_sel = list(range(min(2, num_spatial_tendons)))

        # torch, no mask
        method(**{kwarg: _make_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # torch, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_torch((num_instances, num_spatial_tendons), device, wp_dtype),
                "spatial_tendon_mask": _make_item_mask(num_spatial_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})
        # warp, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_warp((num_instances, num_spatial_tendons), device, wp_dtype),
                "spatial_tendon_mask": _make_item_mask(num_spatial_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # float scalar
        if accepts_float:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})


# ---------------------------------------------------------------------------
# Tests: Tendon write-to-sim smoke tests
# ---------------------------------------------------------------------------


class TestArticulationWritersTendonToSim:
    """Smoke test write_*_tendon_properties_to_sim_index/mask methods."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_fixed_tendon_properties_to_sim_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        # all envs
        art.write_fixed_tendon_properties_to_sim_index()
        # subset envs
        art.write_fixed_tendon_properties_to_sim_index(env_ids=_make_env_ids(device, True))

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_fixed_tendon_properties_to_sim_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        # no mask
        art.write_fixed_tendon_properties_to_sim_mask()
        # partial env mask
        art.write_fixed_tendon_properties_to_sim_mask(env_mask=_make_env_mask(num_instances, device, True))

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_spatial_tendon_properties_to_sim_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        # all envs
        art.write_spatial_tendon_properties_to_sim_index()
        # subset envs
        art.write_spatial_tendon_properties_to_sim_index(env_ids=_make_env_ids(device, True))

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_spatial_tendon_properties_to_sim_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        # no mask
        art.write_spatial_tendon_properties_to_sim_mask()
        # partial env mask
        art.write_spatial_tendon_properties_to_sim_mask(env_mask=_make_env_mask(num_instances, device, True))
