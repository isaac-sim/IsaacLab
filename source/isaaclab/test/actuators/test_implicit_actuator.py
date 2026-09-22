# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.actuators import ImplicitActuatorCfg

pytestmark = pytest.mark.unit

_DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
]
_JOINT_NAMES = ["joint_0", "joint_1"]
_NUM_ENVS = 2


def _make_actuator(device: str, cfg_kwargs: dict | None = None, **constructor_kwargs):
    cfg = ImplicitActuatorCfg(
        joint_names_expr=_JOINT_NAMES, **({"stiffness": 200.0, "damping": 10.0} | (cfg_kwargs or {}))
    )
    return cfg.class_type(
        cfg, joint_names=_JOINT_NAMES, joint_ids=[0, 1], num_envs=_NUM_ENVS, device=device, **constructor_kwargs
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("usd_default", [False, True])
def test_implicit_actuator_init_minimum(device, usd_default):
    """Configured gains win over the authored (USD) defaults passed to the constructor; limits default to inf."""
    cfg_gains = {"stiffness": None, "damping": None} if usd_default else {}
    actuator = _make_actuator(device, cfg_gains, stiffness=300.0, damping=20.0)

    zeros = torch.zeros(_NUM_ENVS, len(_JOINT_NAMES), device=device)
    assert actuator.is_implicit_model is True
    torch.testing.assert_close(actuator.computed_effort, zeros)
    torch.testing.assert_close(actuator.applied_effort, zeros)
    torch.testing.assert_close(actuator.joint_effort_limit, torch.full_like(zeros, torch.inf))
    torch.testing.assert_close(actuator.actuator_velocity_limit, torch.full_like(zeros, torch.inf))
    with pytest.warns(DeprecationWarning, match="actuator_effort_limit"):
        torch.testing.assert_close(actuator.effort_limit, actuator.joint_effort_limit)
    expected_stiffness, expected_damping = (300.0, 20.0) if usd_default else (200.0, 10.0)
    torch.testing.assert_close(actuator.stiffness, torch.full_like(zeros, expected_stiffness))
    torch.testing.assert_close(actuator.damping, torch.full_like(zeros, expected_damping))


@pytest.mark.parametrize("cfg_limit", [None, 300.0])
@pytest.mark.parametrize("limit_name", ["joint_effort_limit", "actuator_velocity_limit"])
def test_implicit_actuator_init_limits(cfg_limit, limit_name):
    """A cfg-provided limit wins over the constructor default for effort and velocity limits."""
    # used as a stand-in for the usd default value read in by articulation.
    limit_default = 5000.0
    actuator = _make_actuator("cpu", {limit_name: cfg_limit}, **{limit_name: limit_default})
    limit_expected = cfg_limit if cfg_limit is not None else limit_default
    torch.testing.assert_close(
        getattr(actuator, limit_name), torch.full((_NUM_ENVS, len(_JOINT_NAMES)), limit_expected)
    )


@pytest.mark.parametrize(
    ("cfg_limits", "expected_rated", "expected_solver"),
    [
        ({"actuator_effort_limit": 87.0, "joint_effort_limit": 870.0}, 87.0, 870.0),
        ({"effort_limit": 87.0, "effort_limit_sim": 870.0}, 87.0, 870.0),
        ({"effort_limit": 87.0}, 87.0, 87.0),
    ],
    ids=["separate_limits", "deprecated_aliases", "deprecated_effort_limit_alone_reaches_solver"],
)
def test_implicit_actuator_rated_and_solver_effort_limits(cfg_limits, expected_rated, expected_solver):
    """The rated limit stays on the actuator while the solver keeps its own clamp, including via deprecated aliases."""
    if any(name in cfg_limits for name in ("effort_limit", "effort_limit_sim")):
        with pytest.warns(DeprecationWarning):
            actuator = _make_actuator("cpu", cfg_limits)
    else:
        actuator = _make_actuator("cpu", cfg_limits)
    shape = (_NUM_ENVS, len(_JOINT_NAMES))
    torch.testing.assert_close(actuator.actuator_effort_limit, torch.full(shape, expected_rated))
    torch.testing.assert_close(actuator.joint_effort_limit, torch.full(shape, expected_solver))
