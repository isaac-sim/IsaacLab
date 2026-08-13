# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import pytest
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.integration


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    with build_simulation_context(device=device) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("usd_default", [False, True])
def test_implicit_actuator_init_minimum(sim, num_envs, num_joints, device, usd_default):
    """Test initialization of implicit actuator with minimum configuration."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = None if usd_default else 200
    damping = None if usd_default else 10

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    # faux usd defaults
    stiffness_default = 300
    damping_default = 20

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=stiffness_default,
        damping=damping_default,
    )

    # check initialized actuator
    assert actuator.is_implicit_model is True
    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))

    torch.testing.assert_close(actuator.effort_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.velocity_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))

    if not usd_default:
        torch.testing.assert_close(actuator.stiffness, stiffness * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.damping, damping * torch.ones(num_envs, num_joints, device=device))
    else:
        torch.testing.assert_close(
            actuator.stiffness, stiffness_default * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(actuator.damping, damping_default * torch.ones(num_envs, num_joints, device=device))


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300])
def test_implicit_actuator_init_effort_limits(sim, num_envs, num_joints, device, effort_lim):
    """Test initialization of implicit actuator with effort limits."""
    effort_limit_default = 5000

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        effort_limit=effort_lim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        effort_limit=effort_limit_default,
    )
    effort_lim_expected = effort_lim if effort_lim is not None else effort_limit_default
    torch.testing.assert_close(
        actuator.effort_limit, effort_lim_expected * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("velocity_lim", [None, 300])
def test_implicit_actuator_init_velocity_limits(sim, num_envs, num_joints, device, velocity_lim):
    """Test initialization of implicit actuator with velocity limits."""
    velocity_limit_default = 1000
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        velocity_limit=velocity_lim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        velocity_limit=velocity_limit_default,
    )
    vel_lim_expected = velocity_lim if velocity_lim is not None else velocity_limit_default

    torch.testing.assert_close(
        actuator.velocity_limit, vel_lim_expected * torch.ones(num_envs, num_joints, device=device)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
