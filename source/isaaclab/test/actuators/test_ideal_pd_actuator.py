# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import math
import torch

import pytest

from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.types import ArticulationActions


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
def test_ideal_pd_actuator_init_minimum(sim, num_envs, num_joints, device, usd_default):
    """Test initialization of ideal pd actuator with minimum configuration."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = None if usd_default else 200
    damping = None if usd_default else 10
    friction = None if usd_default else 0.1
    armature = None if usd_default else 0.2

    actuator_cfg = IdealPDActuatorCfg(
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
    assert actuator.is_implicit_model is False
    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))

    torch.testing.assert_close(
        actuator.effort_limit, actuator._DEFAULT_MAX_EFFORT_SIM * torch.ones(num_envs, num_joints, device=device)
    )
    torch.testing.assert_close(
        actuator.effort_limit_sim, actuator._DEFAULT_MAX_EFFORT_SIM * torch.ones(num_envs, num_joints, device=device)
    )
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
@pytest.mark.parametrize("effort_lim", [None, 300])
@pytest.mark.parametrize("effort_lim_sim", [None, 400])
def test_ideal_pd_actuator_init_effort_limits(sim, num_envs, num_joints, device, effort_lim, effort_lim_sim):
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
        effort_limit=effort_lim,
        effort_limit_sim=effort_lim_sim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        effort_limit=effort_lim_default,
    )

    if effort_lim is not None and effort_lim_sim is None:
        effort_lim_expected = effort_lim
        effort_lim_sim_expected = actuator._DEFAULT_MAX_EFFORT_SIM

    elif effort_lim is None and effort_lim_sim is not None:
        effort_lim_expected = effort_lim_sim
        effort_lim_sim_expected = effort_lim_sim

    elif effort_lim is None and effort_lim_sim is None:
        effort_lim_expected = actuator._DEFAULT_MAX_EFFORT_SIM
        effort_lim_sim_expected = actuator._DEFAULT_MAX_EFFORT_SIM

    elif effort_lim is not None and effort_lim_sim is not None:
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
@pytest.mark.parametrize("velocity_lim", [None, 300])
@pytest.mark.parametrize("velocity_lim_sim", [None, 400])
def test_ideal_pd_actuator_init_velocity_limits(sim, num_envs, num_joints, device, velocity_lim, velocity_lim_sim):
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
        velocity_limit_sim=velocity_lim_sim,
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
    if velocity_lim is not None and velocity_lim_sim is None:
        vel_lim_expected = velocity_lim
        vel_lim_sim_expected = velocity_limit_default
    elif velocity_lim is None and velocity_lim_sim is not None:
        vel_lim_expected = velocity_lim_sim
        vel_lim_sim_expected = velocity_lim_sim
    elif velocity_lim is None and velocity_lim_sim is None:
        vel_lim_expected = velocity_limit_default
        vel_lim_sim_expected = velocity_limit_default
    elif velocity_lim is not None and velocity_lim_sim is not None:
        vel_lim_expected = velocity_lim
        vel_lim_sim_expected = velocity_lim_sim

    torch.testing.assert_close(
        actuator.velocity_limit, vel_lim_expected * torch.ones(num_envs, num_joints, device=device)
    )
    torch.testing.assert_close(
        actuator.velocity_limit_sim, vel_lim_sim_expected * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300])
def test_ideal_pd_compute(sim, num_envs, num_joints, device, effort_lim):
    """Test the computation of the ideal pd actuator."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
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


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 60])
@pytest.mark.parametrize("saturation_effort", [None, 100])
@pytest.mark.parametrize("mirror_t_s", [1.0, -1.0])
@pytest.mark.parametrize("test_point", range(4))
def test_dc_motor_clip(sim, num_envs, num_joints, device, effort_lim, saturation_effort, mirror_t_s, test_point):
    r"""Test the computation of the dc motor actuator 4 quadrant torque speed curve.

    torque_speed_pairs of interest:
    0 - fully inside torque speed curve and effort limit (quadrant 1)
    1 - greater than effort limit but under torque-speed curve (quadrant 1)
    2 - greater than effort limit and outside torque-speed curve (quadrant 1)
    3 - less than effort limit but outside torque speed curve (quadrant 1)
    4 - less than effort limit but outside torque speed curve (quadrant 2)
    5 - fully inside torque speed curve and effort limit (quadrant 2)
    6 - fully outside torque speed curve and -effort limit (quadrant 2)

    e - effort_limit
    s - saturation_effort
    v - velocity_limit
    \ - torque-speed linear boundary between v and s

    each torque_speed_point will be tested in quadrant 3 and 4

    ===========Speed==============
    |\  6  |     |     |         |
    |  \   |     |     |         |
    |    \ |     |     |         |
    |      \  4  |     |         |
    |  Q2  | \   |     |     Q1  |
    |      |   \ |  3  |         |
    |      |     v     |         |
    |      |     | \   | 2       |
    |      |     |   \ |         |
    |      |     |     \         |
    |\     |  5  |  0  |1\       |
    |--s---e-----o-----e---s-----| Torque
    |    \ |     |     |     \   |
    |      \     |     |       \ |
    |      | \   |     |         |
    |  Q3  |   \ |     |     Q4  |
    |      |     v     |         |
    |      |     | \   |         |
    ==============================
    """
    torque_speed_pairs = [
        (30.0, 10.0),
        (70.0, 10.0),
        (80.0, 40.0),
        (30.0, 40.0),
        (-20.0, 90.0),
        (-30.0, 10.0),
        (-80.0, 110.0),
    ]

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    velocity_limit = 50
    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_lim,
        velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
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

    i = test_point
    ts = torque_speed_pairs[test_point]
    torque = ts[0] * mirror_t_s
    speed = ts[1] * mirror_t_s
    actuator._joint_vel[:] = speed * torch.ones(num_envs, num_joints, device=device)
    effort = torque * torch.ones(num_envs, num_joints, device=device)
    clipped_effort = actuator._clip_effort(effort)

    if saturation_effort is not None:
        torque_speed_curve = saturation_effort * (mirror_t_s * 1 - speed / velocity_limit)

    if i == 0 or i == 5:
        expected_clipped_effort = torque
    elif i == 1:
        expected_clipped_effort = math.copysign(effort_lim, torque) if effort_lim is not None else torque
    elif i == 2:
        if saturation_effort is not None:
            expected_clipped_effort = torque_speed_curve
        elif effort_lim is not None:
            expected_clipped_effort = math.copysign(effort_lim, torque)
        else:
            expected_clipped_effort = torque
    elif i == 3:
        if saturation_effort is not None:
            expected_clipped_effort = torque_speed_curve
        else:
            expected_clipped_effort = torque
    elif i == 4:
        if effort_lim is not None:
            expected_clipped_effort = math.copysign(effort_lim, torque)
        elif saturation_effort is not None:
            expected_clipped_effort = torque_speed_curve
        else:
            expected_clipped_effort = torque
        print("expected: ", expected_clipped_effort)
        print("clipped:", clipped_effort)
    elif i == 6:
        if effort_lim is not None:
            expected_clipped_effort = math.copysign(effort_lim, torque)
        elif saturation_effort is not None:
            expected_clipped_effort = torque_speed_curve
        else:
            expected_clipped_effort = torque

    torch.testing.assert_close(
        expected_clipped_effort * torch.ones(num_envs, num_joints, device=device), clipped_effort
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
