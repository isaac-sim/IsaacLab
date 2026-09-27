# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend-independent asset property events."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import events as events_module

_NUM_ENVS = 2


class _ArticulationWithoutDynamicFriction:
    def __init__(self):
        self.device = "cpu"
        joint_properties = torch.zeros((_NUM_ENVS, 2))
        self.data = SimpleNamespace(
            joint_friction_coeff=SimpleNamespace(torch=joint_properties.clone()),
            joint_viscous_friction_coeff=SimpleNamespace(torch=joint_properties.clone()),
            joint_armature=SimpleNamespace(torch=joint_properties.clone()),
            joint_pos_limits=SimpleNamespace(torch=torch.zeros((_NUM_ENVS, 2, 2))),
        )
        self.static_friction_writes = []
        self.viscous_friction_writes = []

    def write_joint_friction_coefficient_to_sim_index(
        self, *, joint_dynamic_friction_coeff=None, joint_viscous_friction_coeff=None, **kwargs
    ):
        """Record static friction and forward an optional viscous-friction write."""
        if joint_viscous_friction_coeff is not None:
            self.write_joint_viscous_friction_coefficient_to_sim_index(
                joint_viscous_friction_coeff=joint_viscous_friction_coeff,
                joint_ids=kwargs["joint_ids"],
                env_ids=kwargs["env_ids"],
            )
        self.static_friction_writes.append(kwargs)

    def write_joint_viscous_friction_coefficient_to_sim_index(self, **kwargs):
        """Record viscous-friction writes."""
        self.viscous_friction_writes.append(kwargs)


class _FakeScene(dict):
    """Dictionary-backed scene with the attributes used by joint randomization."""

    def __init__(self, **assets):
        super().__init__(assets)
        self.num_envs = _NUM_ENVS


@pytest.mark.parametrize("env_ids", [torch.tensor([0], dtype=torch.int32), slice(0, 1)])
def test_joint_parameter_randomization_writes_static_and_viscous_friction(env_ids):
    """Joint randomization writes supported friction properties separately."""
    asset = _ArticulationWithoutDynamicFriction()
    asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    cfg = SimpleNamespace(
        params={
            "asset_cfg": asset_cfg,
            "operation": "abs",
            "friction_distribution_params": (0.5, 0.5),
        }
    )
    env = SimpleNamespace(scene=_FakeScene(robot=asset))

    term = events_module.randomize_joint_parameters(cfg, env)
    term(
        env,
        env_ids,
        asset_cfg,
        friction_distribution_params=(0.5, 0.5),
    )

    assert len(asset.static_friction_writes) == 1
    assert len(asset.viscous_friction_writes) == 1
    static_write = asset.static_friction_writes[0]
    viscous_write = asset.viscous_friction_writes[0]
    assert set(static_write) == {"joint_friction_coeff", "joint_ids", "env_ids"}
    assert set(viscous_write) == {"joint_viscous_friction_coeff", "joint_ids", "env_ids"}
    torch.testing.assert_close(static_write["joint_friction_coeff"], torch.full((1, 2), 0.5))
    torch.testing.assert_close(viscous_write["joint_viscous_friction_coeff"], torch.full((1, 2), 0.5))
    assert static_write["env_ids"] is env_ids
    assert viscous_write["env_ids"] is env_ids


def test_fixed_tendon_randomization_writes_limit_stiffness_and_rest_length():
    """Fixed tendon randomization writes every requested property through the asset setters."""
    tendon_values = torch.zeros((_NUM_ENVS, 2))
    writes = {}
    asset = SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(
            fixed_tendon_limit_stiffness=SimpleNamespace(torch=tendon_values.clone()),
            fixed_tendon_rest_length=SimpleNamespace(torch=tendon_values.clone()),
        ),
        set_fixed_tendon_limit_stiffness_index=lambda **kwargs: writes.update(limit_stiffness=kwargs),
        set_fixed_tendon_rest_length_index=lambda **kwargs: writes.update(rest_length=kwargs),
        write_fixed_tendon_properties_to_sim_index=lambda **kwargs: writes.update(sim=kwargs),
    )
    asset_cfg = SimpleNamespace(name="robot", fixed_tendon_ids=slice(None))
    cfg = SimpleNamespace(params={"asset_cfg": asset_cfg, "operation": "abs"})
    env = SimpleNamespace(scene=_FakeScene(robot=asset))

    term = events_module.randomize_fixed_tendon_parameters(cfg, env)
    term(
        env,
        torch.tensor([0]),
        asset_cfg,
        limit_stiffness_distribution_params=(2.0, 2.0),
        rest_length_distribution_params=(0.5, 0.5),
    )

    torch.testing.assert_close(writes["limit_stiffness"]["limit_stiffness"], torch.full((1, 2), 2.0))
    torch.testing.assert_close(writes["rest_length"]["rest_length"], torch.full((1, 2), 0.5))
    assert "sim" in writes
