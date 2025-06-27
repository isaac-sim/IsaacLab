# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
    friction = None if usd_default else 0.1
    armature = None if usd_default else 0.2

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        friction=friction,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    # faux usd defaults
    stiffness_default = 300
    damping_default = 20
    friction_default = 0.0
    armature_default = 0.0

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=stiffness_default,
        damping=damping_default,
        friction=friction_default,
        armature=armature_default,
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

    if not usd_default:
        torch.testing.assert_close(actuator.stiffness, stiffness * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.damping, damping * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.armature, armature * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.friction, friction * torch.ones(num_envs, num_joints, device=device))
    else:
        torch.testing.assert_close(
            actuator.stiffness, stiffness_default * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(actuator.damping, damping_default * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(
            actuator.armature, armature_default * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(
            actuator.friction, friction_default * torch.ones(num_envs, num_joints, device=device)
        )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300, 200])
@pytest.mark.parametrize("effort_lim_sim", [None, 400, 200])
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

    if effort_lim is not None and effort_lim_sim is not None and effort_lim != effort_lim_sim:
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
            effort_lim_expected = effort_lim
            effort_lim_sim_expected = effort_lim

        elif effort_lim is None and effort_lim_sim is not None:
            assert actuator.cfg.effort_limit_sim == actuator.cfg.effort_limit
            effort_lim_expected = effort_lim_sim
            effort_lim_sim_expected = effort_lim_sim

        elif effort_lim is None and effort_lim_sim is None:
            assert actuator.cfg.effort_limit_sim is None
            assert actuator.cfg.effort_limit is None
            effort_lim_expected = effort_limit_default
            effort_lim_sim_expected = effort_limit_default

        elif effort_lim is not None and effort_lim_sim is not None:
            assert actuator.cfg.effort_limit_sim == actuator.cfg.effort_limit
            effort_lim_expected = effort_lim
            effort_lim_sim_expected = effort_lim_sim

        torch.testing.assert_close(
            actuator.effort_limit, effort_lim_expected * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(
            actuator.effort_limit_sim, effort_lim_sim_expected * torch.ones(num_envs, num_joints, device=device)
        )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("velocity_lim", [None, 300, 200])
@pytest.mark.parametrize("velocity_lim_sim", [None, 400, 200])
def test_implicit_actuator_init_velocity_limits(sim, num_envs, num_joints, device, velocity_lim, velocity_lim_sim):
    """Test initialization of implicit actuator with velocity limits.

    Note implicit actuators do no use velocity limits in computation, they are passed to physics via articulations.
    """
    velocity_limit_default = 1000
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        velocity_limit=velocity_lim,
        velocity_limit_sim=velocity_lim_sim,
    )

    if velocity_lim is not None and velocity_lim_sim is not None and velocity_lim != velocity_lim_sim:
        with pytest.raises(ValueError):
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
    else:
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
        if velocity_lim is not None and velocity_lim_sim is None:
            assert actuator.cfg.velocity_limit is None
            vel_lim_expected = velocity_limit_default
        elif velocity_lim is None and velocity_lim_sim is not None:
            assert actuator.cfg.velocity_limit == actuator.cfg.velocity_limit_sim
            vel_lim_expected = velocity_lim_sim
        elif velocity_lim is None and velocity_lim_sim is None:
            assert actuator.cfg.velocity_limit is None
            assert actuator.cfg.velocity_limit_sim is None
            vel_lim_expected = velocity_limit_default
        else:
            assert actuator.cfg.velocity_limit == actuator.cfg.velocity_limit_sim
            vel_lim_expected = velocity_lim_sim

        torch.testing.assert_close(
            actuator.velocity_limit, vel_lim_expected * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(
            actuator.velocity_limit_sim, vel_lim_expected * torch.ones(num_envs, num_joints, device=device)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
