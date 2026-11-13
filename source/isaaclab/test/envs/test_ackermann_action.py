# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from isaaclab.controllers import AckermannControllerCfg
from isaaclab.envs.mdp.actions import AckermannAction, AckermannActionCfg


class _FakeArticulation:
    """Minimal articulation interface used by the action term."""

    def __init__(self):
        self.joint_names = [
            "rear_right_wheel",
            "left_steering",
            "front_left_wheel",
            "right_steering",
            "rear_left_wheel",
            "front_right_wheel",
        ]
        self.position_target = None
        self.position_joint_ids = None
        self.velocity_target = None
        self.velocity_joint_ids = None

    def find_joints(self, name_keys: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        ids = []
        names = []
        search_keys = name_keys if preserve_order else self.joint_names
        for key in search_keys:
            for joint_id, joint_name in enumerate(self.joint_names):
                if re.fullmatch(key, joint_name):
                    ids.append(joint_id)
                    names.append(joint_name)
        return ids, names

    def set_joint_position_target_index(self, *, target: torch.Tensor, joint_ids: list[int]):
        self.position_target = target.clone()
        self.position_joint_ids = list(joint_ids)

    def set_joint_velocity_target_index(self, *, target: torch.Tensor, joint_ids: list[int]):
        self.velocity_target = target.clone()
        self.velocity_joint_ids = list(joint_ids)


class _FakeMarkerRegistry:
    def clear_debug_vis_callback(self, action) -> None:
        pass


def _make_action_cfg(**kwargs) -> AckermannActionCfg:
    controller_cfg = AckermannControllerCfg(
        wheel_radius=0.25,
        wheel_base=1.5,
        track_width=1.0,
        non_steerable_wheel_offsets=(0.5, -0.5),
        max_linear_speed=3.0,
        max_steering_angle=0.6,
    )
    values = {
        "asset_name": "vehicle",
        "steering_joint_names": ["left_steering", "right_steering"],
        "wheel_joint_names": [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ],
        "steering_joint_directions": (1.0, -1.0),
        "wheel_joint_directions": (1.0, -1.0, 1.0, -1.0),
        "controller": controller_cfg,
    }
    values.update(kwargs)
    return AckermannActionCfg(**values)


def _make_action(cfg: AckermannActionCfg) -> tuple[AckermannAction, _FakeArticulation]:
    asset = _FakeArticulation()
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={"vehicle": asset},
        sim=SimpleNamespace(vis_marker_registry=_FakeMarkerRegistry()),
    )
    return AckermannAction(cfg, env), asset


def test_action_resolves_order_and_applies_direction_multipliers() -> None:
    action, asset = _make_action(_make_action_cfg())
    actions = torch.tensor([[1.0, 0.2], [-0.5, -0.1]])

    action.process_actions(actions)
    reference_steering, reference_wheels = action._controller.compute(actions)
    reference_steering = reference_steering.clone()
    reference_wheels = reference_wheels.clone()
    action.apply_actions()

    assert asset.position_joint_ids == [1, 3]
    assert asset.velocity_joint_ids == [2, 5, 4, 0]
    torch.testing.assert_close(asset.position_target, reference_steering * torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(asset.velocity_target, reference_wheels * torch.tensor([[1.0, -1.0, 1.0, -1.0]]))


def test_action_scales_offsets_and_clips_commands() -> None:
    cfg = _make_action_cfg(
        scale=(2.0, 0.5),
        offset=(0.1, -0.05),
        clip={"linear_speed": (-0.5, 0.75), "steering_angle": (-0.2, 0.2)},
    )
    action, _ = _make_action(cfg)

    action.process_actions(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))

    torch.testing.assert_close(action.processed_actions, torch.tensor([[0.75, 0.2], [-0.5, -0.2]]))


def test_action_reset_clears_only_selected_environments() -> None:
    action, _ = _make_action(_make_action_cfg())
    commands = torch.tensor([[1.0, 0.2], [-0.5, -0.1]])
    action.process_actions(commands)
    action.apply_actions()

    action.reset(torch.tensor([0]))

    torch.testing.assert_close(action.raw_actions, torch.tensor([[0.0, 0.0], [-0.5, -0.1]]))
    torch.testing.assert_close(action.processed_actions, torch.tensor([[0.0, 0.0], [-0.5, -0.1]]))
    torch.testing.assert_close(action._controller._command, torch.tensor([[0.0, 0.0], [-0.5, -0.1]]))


def test_action_clip_cannot_saturate_non_finite_commands() -> None:
    """NaN or infinity produces safe zero targets even when action clipping is configured."""
    cfg = _make_action_cfg(
        clip={"linear_speed": (-1.0, 1.0), "steering_angle": (-0.4, 0.4)},
    )
    action, asset = _make_action(cfg)

    action.process_actions(torch.tensor([[torch.inf, 0.2], [1.0, torch.nan]]))
    action.apply_actions()

    torch.testing.assert_close(action.processed_actions, torch.zeros_like(action.processed_actions))
    torch.testing.assert_close(asset.position_target, torch.zeros_like(asset.position_target))
    torch.testing.assert_close(asset.velocity_target, torch.zeros_like(asset.velocity_target))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("steering_joint_names", ["left_steering"], "exactly two"),
        ("steering_joint_names", ["left_steering", "left_steering"], "duplicate joints"),
        ("wheel_joint_names", ["front_left_wheel", "front_right_wheel"], "exactly 4"),
        ("steering_joint_directions", (1.0, 0.0), "either -1.0 or 1.0"),
        ("wheel_joint_directions", (1.0, -1.0), "must contain 4"),
    ],
)
def test_action_rejects_invalid_joint_mapping(field: str, value: list[str] | tuple[float, ...], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _make_action(_make_action_cfg(**{field: value}))


def test_mdp_aggregate_exports_ackermann_action() -> None:
    """Task configs can reach the Ackermann action API through ``isaaclab.envs.mdp``."""
    from isaaclab.envs import mdp

    assert mdp.AckermannAction is AckermannAction
    assert mdp.AckermannActionCfg is AckermannActionCfg
