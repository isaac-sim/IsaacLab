# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native PhysX conveyor device and authored geometry contracts."""

import pytest

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_physx_env_cfg import physx_belt_section_specs
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import BELT_TURN_RADIUS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_physx_preserves_cpu_policy_and_pivot_conventions():
    """Unspecified parser devices retain CPU; native twists/materials preserve the trained interface."""
    cfg = parse_env_cfg("IsaacContrib-Conveyor-Racetrack-Transfer-PhysX-CPU-v0", device=None, num_envs=2)
    cfg.validate()
    base = ConveyorFrankaEnvCfg()
    assert cfg.sim.device == "cpu" and cfg.scene.num_envs == 2
    assert cfg.actions == base.actions and cfg.observations == base.observations
    assert cfg.sim.dt == base.sim.dt == 1 / 120 and cfg.decimation == base.decimation == 2
    cfg.scene.configure_conveyor(friction_coefficient=0.42)
    specs = cfg.scene.build_conveyor_belt_specs(velocity=0.21, friction_coefficient=0.42, contact_threshold=0.95)
    assert len(specs) == 8
    assert all(s.velocity == 0.21 and s.friction_coefficient == 0.42 and s.contact_threshold == 0.95 for s in specs)
    for side in ("Left", "Right"):
        for section, root in physx_belt_section_specs(side)[2:]:
            assert section.belt.curved
            assert section.belt.pivot_point == (0, 0, 0) and section.belt.radius == BELT_TURN_RADIUS
            assert root[2] == 0
        for key in ("top_straight", "bottom_straight", "right_turn", "left_turn"):
            spawn = getattr(cfg.scene, f"conveyor_{side.lower()}_{key}_collision").spawn
            assert spawn.physics_material.static_friction == spawn.physics_material.dynamic_friction == 0.42
            assert spawn.rigid_props.kinematic_enabled
        assert (
            getattr(cfg.scene, f"conveyor_{side.lower()}_right_turn_collision").spawn.collision_approximation == "sdf"
        )
    cfg.sim.device = "cuda:0"
    with pytest.raises(ValueError, match="CPU-only.*--device cpu"):
        cfg.validate()
