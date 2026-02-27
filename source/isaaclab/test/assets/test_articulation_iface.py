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

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import pytest
from unittest.mock import MagicMock
import warp as wp
import torch

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

"""
Check which backends are available.
"""

BACKENDS = ["Mock"] # Mock backend is always available.

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
    BACKENDS.append("newton")
except ImportError:
    pass

def create_physx_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
):
    """Create a test Articulation instance with mocked dependencies."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(PhysXArticulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )

    # Create PhysX mock view
    mock_view = PhysXMockArticulationViewWarp(
        count=num_instances,
        num_links=num_bodies,
        num_dofs=num_joints,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Set up the mock view's metatype for accessing names/counts
    mock_metatype = MagicMock()
    mock_metatype.fixed_base = False
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

    # Create mock wrench composers (pass articulation which has num_instances, num_bodies, device properties)
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    # Set up other required attributes
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)
    object.__setattr__(articulation, "_ALL_INDICES", torch.arange(num_instances, dtype=torch.int32, device=device))
    object.__setattr__(articulation, "_ALL_BODY_INDICES", torch.arange(num_bodies, dtype=torch.int32, device=device))
    object.__setattr__(articulation, "_ALL_JOINT_INDICES", torch.arange(num_joints, dtype=torch.int32, device=device))

    # Warp arrays for set_external_force_and_torque
    all_indices = torch.arange(num_instances, dtype=torch.int32, device=device)
    all_body_indices = torch.arange(num_bodies, dtype=torch.int32, device=device)
    object.__setattr__(articulation, "_ALL_INDICES_WP", wp.from_torch(all_indices, dtype=wp.int32))
    object.__setattr__(articulation, "_ALL_BODY_INDICES_WP", wp.from_torch(all_body_indices, dtype=wp.int32))

    # Initialize joint targets
    object.__setattr__(articulation, "_joint_pos_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_vel_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_effort_target_sim", torch.zeros(num_instances, num_joints, device=device))

    return articulation, mock_view

def create_newton_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
):
    raise NotImplementedError("Newton articulation is not supported yet")

def create_mock_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
):
    raise NotImplementedError("Mock articulation is not supported yet")

def get_articulation(backend: str, num_instances: int = 2, num_joints: int = 6, num_bodies: int = 7, device: str = "cuda:0"):
    if backend == "physx":
        return create_physx_articulation(num_instances, num_joints, num_bodies, device)
    elif backend == "newton":
        return create_newton_articulation(num_instances, num_joints, num_bodies, device)
    elif backend == "mock":
        return create_mock_articulation(num_instances, num_joints, num_bodies, device)
    else:
        raise ValueError(f"Invalid backend: {backend}")

@pytest.fixture
def articulation_iface(request):
    backend = request.getfixturevalue("backend")
    num_instances = request.getfixturevalue("num_instances")
    num_joints = request.getfixturevalue("num_joints")
    num_bodies = request.getfixturevalue("num_bodies")
    device = request.getfixturevalue("device")
    return get_articulation(backend, num_instances, num_joints, num_bodies, device)

