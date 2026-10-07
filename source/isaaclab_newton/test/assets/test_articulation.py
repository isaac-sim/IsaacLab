# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices
from isaaclab.utils import clone

HEADLESS = True

launch_test_simulation()


import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelBuilder
from newton.selection import ArticulationView

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_NEWTON_CFG  # isort:skip

_FRANKA_PANDA_NEWTON_CFG = clone(FRANKA_PANDA_CFG)
_FRANKA_PANDA_NEWTON_CFG.spawn.variants = {"Physics": "mujoco", "Colliders": "gripper_only"}
_FRANKA_PANDA_HIGH_PD_NEWTON_CFG = clone(FRANKA_PANDA_HIGH_PD_NEWTON_CFG)
_FRANKA_PANDA_HIGH_PD_NEWTON_CFG.spawn.variants = {"Physics": "mujoco", "Colliders": "gripper_only"}


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_world_hinged_root_has_no_base_dofs(monkeypatch, device):
    """A root link hinged to the world adds no floating-base DoF columns to the Jacobian or mass matrix."""
    builder = ModelBuilder()
    builder.begin_world()
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="Robot/base")
    builder.add_articulation([builder.add_joint_revolute(-1, base, axis=(0.0, 1.0, 0.0))], label="Robot")
    builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    monkeypatch.setattr(SimulationManager, "get_model", lambda: model)
    monkeypatch.setattr(SimulationManager, "get_state_0", lambda: state)
    monkeypatch.setattr(SimulationManager, "get_control", lambda: control)
    data = ArticulationData(ArticulationView(model, "Robot"), device)
    data._apply_ordering_maps_after_resolve()

    assert data.body_link_jacobian_w.torch.shape[-1] == data.mass_matrix.torch.shape[-1] == 1
