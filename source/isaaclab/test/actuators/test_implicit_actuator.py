# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.test.utils import test_devices

pytestmark = pytest.mark.integration

# Shapes broadcast the same scalar parameters, so one multi-env, multi-joint shape covers them.
NUM_ENVS, NUM_JOINTS = 2, 2


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("usd_default", [False, True])
def test_implicit_actuator_init_minimum(device, usd_default):
    """Test initialization of implicit actuator with minimum configuration."""
    num_envs, num_joints = NUM_ENVS, NUM_JOINTS

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

    torch.testing.assert_close(actuator.joint_effort_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))
    with pytest.warns(DeprecationWarning, match="actuator_effort_limit"):
        torch.testing.assert_close(actuator.effort_limit, actuator.joint_effort_limit)
    torch.testing.assert_close(
        actuator.actuator_velocity_limit, torch.inf * torch.ones(num_envs, num_joints, device=device)
    )

    if not usd_default:
        torch.testing.assert_close(actuator.stiffness, stiffness * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.damping, damping * torch.ones(num_envs, num_joints, device=device))
    else:
        torch.testing.assert_close(
            actuator.stiffness, stiffness_default * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(actuator.damping, damping_default * torch.ones(num_envs, num_joints, device=device))


@pytest.mark.parametrize("cfg_limit", [None, 300])
@pytest.mark.parametrize(
    "limit_name",
    ["joint_effort_limit", "actuator_velocity_limit"],
)
def test_implicit_actuator_init_limits(cfg_limit, limit_name):
    """Test that a cfg-provided limit wins over the constructor default for effort and velocity limits."""
    num_envs, num_joints, device = NUM_ENVS, NUM_JOINTS, "cpu"
    # used as a standin for the usd default value read in by articulation.
    limit_default = 5000

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        **{limit_name: cfg_limit},
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        **{limit_name: limit_default},
    )
    limit_expected = cfg_limit if cfg_limit is not None else limit_default
    torch.testing.assert_close(
        getattr(actuator, limit_name), limit_expected * torch.ones(num_envs, num_joints, device=device)
    )


def test_implicit_actuator_deprecated_effort_aliases_resolve_rated_and_solver():
    """Deprecated ``effort_limit``/``effort_limit_sim`` map to the rated and solver limits."""
    device = "cpu"
    joint_names = ["joint_0"]
    actuator_cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        effort_limit=87.0,
        effort_limit_sim=870.0,
    )
    with pytest.warns(DeprecationWarning):
        actuator = actuator_cfg.class_type(
            actuator_cfg,
            joint_names=joint_names,
            joint_ids=[0],
            num_envs=2,
            device=device,
            stiffness=actuator_cfg.stiffness,
            damping=actuator_cfg.damping,
        )
    torch.testing.assert_close(actuator.actuator_effort_limit, 87.0 * torch.ones(2, 1, device=device))
    torch.testing.assert_close(actuator.joint_effort_limit, 870.0 * torch.ones(2, 1, device=device))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
