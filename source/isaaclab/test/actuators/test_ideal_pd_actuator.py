# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("usd_default", [False, True])
def test_ideal_pd_actuator_init_minimum(num_envs, num_joints, device, usd_default):
    """Test initialization of ideal pd actuator with minimum configuration."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = None if usd_default else 200
    damping = None if usd_default else 10

    actuator_cfg = IdealPDActuatorCfg(
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
    assert actuator.is_implicit_model is False
    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))

    torch.testing.assert_close(
        actuator.actuator_effort_limit, torch.inf * torch.ones(num_envs, num_joints, device=device)
    )
    with pytest.warns(DeprecationWarning, match="actuator_effort_limit"):
        torch.testing.assert_close(actuator.effort_limit, actuator.actuator_effort_limit)
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
def test_ideal_pd_actuator_init_effort_limits(num_envs, num_joints, device, effort_lim):
    """Test initialization of ideal pd actuator with effort limits."""
    # used as a standin for the usd default value read in by articulation.
    # This value should not be propagated for ideal pd actuators
    effort_lim_default = 5000

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        actuator_effort_limit=effort_lim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        actuator_effort_limit=effort_lim_default,
    )

    effort_lim_expected = effort_lim if effort_lim is not None else effort_lim_default

    torch.testing.assert_close(
        actuator.actuator_effort_limit, effort_lim_expected * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("velocity_lim", [None, 300])
def test_ideal_pd_actuator_init_velocity_limits(num_envs, num_joints, device, velocity_lim):
    """Test initialization of ideal pd actuator with velocity limits.

    Note Ideal PD actuator does not use velocity limits in computation, they are passed to physics via articulations.
    """
    velocity_limit_default = 1000
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = IdealPDActuatorCfg(
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


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300])
def test_ideal_pd_compute(num_envs, num_joints, device, effort_lim):
    """Test the computation of the ideal pd actuator."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        actuator_effort_limit=effort_lim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
    )
    desired_pos = 10.0
    desired_vel = 0.1
    measured_joint_pos = 1.0
    measured_joint_vel = -0.1

    desired_control_action = ArticulationActions()
    desired_control_action.joint_positions = desired_pos * torch.ones(num_envs, num_joints, device=device)
    desired_control_action.joint_velocities = desired_vel * torch.ones(num_envs, num_joints, device=device)
    desired_control_action.joint_efforts = torch.zeros(num_envs, num_joints, device=device)

    expected_comp_joint_effort = stiffness * (desired_pos - measured_joint_pos) + damping * (
        desired_vel - measured_joint_vel
    )

    computed_control_action = actuator.compute(
        desired_control_action,
        measured_joint_pos * torch.ones(num_envs, num_joints, device=device),
        measured_joint_vel * torch.ones(num_envs, num_joints, device=device),
    )

    torch.testing.assert_close(
        expected_comp_joint_effort * torch.ones(num_envs, num_joints, device=device), actuator.computed_effort
    )

    if effort_lim is None:
        torch.testing.assert_close(
            expected_comp_joint_effort * torch.ones(num_envs, num_joints, device=device), actuator.applied_effort
        )
    else:
        torch.testing.assert_close(
            effort_lim * torch.ones(num_envs, num_joints, device=device), actuator.applied_effort
        )
    torch.testing.assert_close(
        actuator.applied_effort,
        computed_control_action.joint_efforts,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
