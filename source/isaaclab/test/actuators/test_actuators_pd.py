# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import torch

import pytest

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import build_simulation_context


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    if "gravity_enabled" in request.fixturenames:
        gravity_enabled = request.getfixturevalue("gravity_enabled")
    else:
        gravity_enabled = True  # default to gravity enabled
    if "add_ground_plane" in request.fixturenames:
        add_ground_plane = request.getfixturevalue("add_ground_plane")
    else:
        add_ground_plane = False  # default to no ground plane
    with build_simulation_context(
        device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled, add_ground_plane=add_ground_plane
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_implicit_actuator_init_minimum(sim, num_envs, num_joints, device):
    """Test initialization of implicit actuator with minimum configuration."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
    )

    # check initialized actuator
    assert actuator.is_implicit_model is True
    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))

    torch.testing.assert_close(actuator.effort_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.effort_limit_sim, torch.inf * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.velocity_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.velocity_limit_sim, torch.inf * torch.ones(num_envs, num_joints, device=device))

    torch.testing.assert_close(actuator.stiffness, stiffness * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.damping, damping * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.armature, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.friction, torch.zeros(num_envs, num_joints, device=device))


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300])
@pytest.mark.parametrize("effort_lim_sim", [None, 400])
def test_implicit_actuator_init_effort_limits(sim, num_envs, num_joints, device, effort_lim, effort_lim_sim):
    """Test initialization of implicit actuator with effort limits."""
    effort_limit_default = 5000

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        effort_limit=effort_lim,
        effort_limit_sim=effort_lim_sim,
    )

    if effort_lim is not None and effort_lim_sim is not None:
        with pytest.raises(ValueError):
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
    else:
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
        if effort_lim is not None and effort_lim_sim is None:
            assert actuator.cfg.effort_limit_sim == actuator.cfg.effort_limit
            torch.testing.assert_close(
                actuator.effort_limit, effort_lim * torch.ones(num_envs, num_joints, device=device)
            )
            torch.testing.assert_close(
                actuator.effort_limit_sim, effort_lim * torch.ones(num_envs, num_joints, device=device)
            )
        elif effort_lim is None and effort_lim_sim is not None:
            assert actuator.cfg.effort_limit_sim == actuator.cfg.effort_limit
            torch.testing.assert_close(
                actuator.effort_limit, effort_lim_sim * torch.ones(num_envs, num_joints, device=device)
            )
            torch.testing.assert_close(
                actuator.effort_limit_sim, effort_lim_sim * torch.ones(num_envs, num_joints, device=device)
            )
        else:
            assert actuator.cfg.effort_limit_sim is None
            assert actuator.cfg.effort_limit is None
            torch.testing.assert_close(
                actuator.effort_limit, effort_limit_default * torch.ones(num_envs, num_joints, device=device)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
