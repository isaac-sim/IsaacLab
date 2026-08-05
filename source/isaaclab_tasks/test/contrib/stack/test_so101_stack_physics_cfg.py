# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free unit tests for the SO-101 stack physics preset.

Regression coverage for :class:`SO101StackPhysicsCfg`: the preset must construct
(``@configclass`` presets expose their members on instances, not on the class), it
must enable ``solve_articulation_contact_last`` (contacts solved after the position
drive, so the closing jaw stalls at the object surface instead of tunneling into
it), and it must preserve the stack-family PhysX tuning. Both SO-101 stack tasks
must actually carry the preset. No gym.make, USD, or GPU.
"""

# Both tasks expose the same class name from different modules; alias for clarity.
from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import (
    SO101CubeStackEnvCfg as SO101CubeStackIKAbsEnvCfg,
)
from isaaclab_tasks.contrib.stack.config.so101.stack_joint_pos_env_cfg import (
    SO101CubeStackEnvCfg,
    SO101StackPhysicsCfg,
)


def test_preset_enables_contact_last_and_keeps_stack_tuning():
    preset = SO101StackPhysicsCfg()
    assert preset.default == preset.isaacsim_physx
    for physx in (preset.isaacsim_physx, preset.physx.isaacsim_physx):
        assert physx.solve_articulation_contact_last is True
        assert physx.bounce_threshold_velocity == 0.01
        assert physx.friction_correlation_distance == 0.00625


def test_both_so101_stack_tasks_carry_the_preset():
    for cfg_cls in (SO101CubeStackEnvCfg, SO101CubeStackIKAbsEnvCfg):
        cfg = cfg_cls()
        assert isinstance(cfg.sim.physics, SO101StackPhysicsCfg)
        assert cfg.sim.physics.physx.isaacsim_physx.solve_articulation_contact_last is True
