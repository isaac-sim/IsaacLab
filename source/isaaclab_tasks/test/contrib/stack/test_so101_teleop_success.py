# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free unit tests for the SO-101 joint-teleop success termination.

The joint-teleop pipeline mirrors the leader arm's raw gripper encoder angle onto the follower 1:1,
so the follower jaw stops wherever the operator's leader jaw does. These tests pin the gripper-open
window that :func:`~isaaclab_tasks.contrib.stack.mdp.cubes_stacked` accepts, using the success term
as the environment actually wires it, so a tolerance that is too tight to reach in practice cannot
be reintroduced silently.
"""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.stack.config.so101.stack_joint_teleop_env_cfg import SO101CubeStackEnvCfg

# Resting center-to-center distance of two stacked cubes [m], measured in sim.
_STACK_DZ = 0.0477


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    """Wrap a tensor in the ``.torch`` accessor the MDP terms read."""
    return SimpleNamespace(torch=tensor)


class _Scene:
    """Minimal scene mapping. Deliberately has no ``surface_grippers`` attribute."""

    def __init__(self, entities: dict):
        self._entities = entities

    def __getitem__(self, key: str):
        return self._entities[key]


class _Robot:
    def __init__(self, gripper_pos: float):
        self.data = SimpleNamespace(joint_pos=_proxy(torch.tensor([[gripper_pos]])))

    def find_joints(self, names):
        return [0], names


def _make_env(gripper_pos: float, stacked: bool = True) -> SimpleNamespace:
    """Build a fake env holding one stack of three cubes and a gripper at ``gripper_pos`` [rad]."""
    # cube_1 bottom, cube_2 middle, cube_3 top, sharing an xy column.
    heights = [0.0, _STACK_DZ, 2 * _STACK_DZ]
    if not stacked:
        # Swap the top two so the required bottom-to-top order no longer holds.
        heights = [0.0, 2 * _STACK_DZ, _STACK_DZ]
    cubes = {
        f"cube_{i + 1}": SimpleNamespace(data=SimpleNamespace(root_pos_w=_proxy(torch.tensor([[0.3, 0.0, z]]))))
        for i, z in enumerate(heights)
    }
    return SimpleNamespace(
        scene=_Scene({"robot": _Robot(gripper_pos), **cubes}),
        cfg=SimpleNamespace(gripper_joint_names=["gripper"], gripper_open_val=1.745),
        device="cpu",
    )


def _success(gripper_pos: float, stacked: bool = True) -> bool:
    """Evaluate the success term exactly as the joint-teleop env configures it."""
    term = SO101CubeStackEnvCfg().terminations.success
    return bool(term.func(_make_env(gripper_pos, stacked=stacked), **term.params)[0])


@pytest.mark.parametrize("gripper_pos", [1.745, 1.5, 1.3])
def test_success_fires_once_the_gripper_is_open_enough(gripper_pos):
    """A completed stack terminates without opening the jaw to the very top of its range.

    1.3 rad fails under the previous 0.2 rad tolerance, which is the regression this guards.
    """
    assert _success(gripper_pos)


@pytest.mark.parametrize("gripper_pos", [0.0, 0.5, 1.2])
def test_success_does_not_fire_while_the_gripper_is_still_closing(gripper_pos):
    """The window still has a lower bound, so success cannot fire while the cube is held."""
    assert not _success(gripper_pos)


def test_success_requires_the_documented_stack_order():
    """Cubes out of the bottom-to-top cube_1, cube_2, cube_3 order never count as stacked."""
    assert not _success(1.745, stacked=False)
