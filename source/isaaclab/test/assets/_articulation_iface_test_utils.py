# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared mocked articulation backend factories for interface tests."""

from unittest.mock import MagicMock

from _iface_test_boot import simulation_app

import numpy as np
import warp as wp

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

BACKENDS = ["Mock"]  # Mock backend is always available.
BACKEND_UNAVAILABLE_REASONS: dict[str, str] = {}

try:
    from isaaclab_physx.assets.articulation.articulation import Articulation as PhysXArticulation
    from isaaclab_physx.assets.articulation.articulation_data import ArticulationData as PhysXArticulationData
    from isaaclab_physx.physics import PhysxManager as SimulationManager
    from isaaclab_physx.test.mock_interfaces.views import MockArticulationViewWarp as PhysXMockArticulationViewWarp
except ImportError as error:
    BACKEND_UNAVAILABLE_REASONS["physx"] = f"{type(error).__name__}: {error}"
else:
    # PhysX data classes need gravity even though interface tests do not create a physics scene.
    _mock_physics_sim_view = MagicMock()
    _mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)
    SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)

    BACKENDS.append("physx")

try:
    from isaaclab_newton.assets.articulation.articulation import Articulation as NewtonArticulation
    from isaaclab_newton.assets.articulation.articulation_data import ArticulationData as NewtonArticulationData
    from isaaclab_newton.test.mock_interfaces.views import MockNewtonArticulationView as NewtonMockArticulationView
except ImportError as error:
    BACKEND_UNAVAILABLE_REASONS["newton"] = f"{type(error).__name__}: {error}"
else:
    BACKENDS.append("newton")

try:
    import ovphysx  # noqa: F401

    from isaaclab_ovphysx.assets.articulation.articulation import Articulation as OvPhysxArticulation
    from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData as OvPhysxArticulationData
    from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet
except ImportError as error:
    BACKEND_UNAVAILABLE_REASONS["ovphysx"] = f"{type(error).__name__}: {error}"
else:
    BACKENDS.append("ovphysx")


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
    articulation._ordering_configure_backend_staging()

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
    object.__setattr__(articulation, "_sim_env_ids", wp.empty(N, dtype=wp.int32, device=device))
    object.__setattr__(articulation, "_sim_env_ids_views", {})
    cpu_env_ids = wp.array(np.arange(N, dtype=np.int32), device="cpu")
    object.__setattr__(articulation, "_cpu_env_ids_all", cpu_env_ids)
    object.__setattr__(articulation, "_cpu_env_ids", wp.empty(N, dtype=wp.int32, device="cpu", pinned=True))
    object.__setattr__(articulation, "_cpu_env_ids_views", {})
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

    from isaaclab_ovphysx import tensor_types as TT

    object.__setattr__(articulation, "_can_write_effort", articulation._get_binding(TT.DOF_ACTUATION_FORCE) is not None)
    object.__setattr__(articulation, "_can_write_pos_target", articulation._get_binding(TT.DOF_POSITION_TARGET) is not None)
    object.__setattr__(articulation, "_can_write_vel_target", articulation._get_binding(TT.DOF_VELOCITY_TARGET) is not None)

    return articulation, mock_bindings


def create_newton_articulation(
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
    """Create a test Newton Articulation instance with mocked dependencies."""
    import isaaclab_newton.assets.articulation.articulation_data as newton_data_module

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    fixed_tendon_names = [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]

    # Create Newton mock view
    mock_view = NewtonMockArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
        tendon_names=fixed_tendon_names,
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
    mock_view._tendon_count = num_fixed_tendons

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

    # Newton supports fixed tendons but not spatial tendons.
    object.__setattr__(articulation, "_fixed_tendon_names", fixed_tendon_names)
    object.__setattr__(articulation, "_spatial_tendon_names", [])
    data.fixed_tendon_names = fixed_tendon_names
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
    articulation._ordering_configure_backend_staging()

    # Tendon arrays
    object.__setattr__(
        articulation,
        "_ALL_FIXED_TENDON_INDICES",
        wp.array(np.arange(num_fixed_tendons, dtype=np.int32), device=device),
    )
    object.__setattr__(
        articulation, "_ALL_FIXED_TENDON_MASK", wp.ones((num_fixed_tendons,), dtype=wp.bool, device=device)
    )
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
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    from isaaclab.test.mock_interfaces.assets.mock_articulation import MockArticulation

    if joint_ordering is not None or body_ordering is not None:
        raise ValueError("The mock backend does not support explicit joint or body ordering.")

    art = MockArticulation(
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=num_bodies,
        is_fixed_base=is_fixed_base,
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
            num_fixed_tendons,
            num_spatial_tendons,
            device,
            is_fixed_base=is_fixed_base,
            joint_ordering=joint_ordering,
            body_ordering=body_ordering,
        )
    elif backend.lower() == "mock":
        return create_mock_articulation(
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
    else:
        raise ValueError(f"Invalid backend: {backend}")
