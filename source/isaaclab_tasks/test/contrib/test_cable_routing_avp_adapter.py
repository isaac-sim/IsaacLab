# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for the right-YAM AVP action adapter and teleoperation script hooks."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaacteleop")

from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.cable_routing.avp_action_adapter import (
    CableRoutingAVPActionAdapter,
    CableRoutingAVPActionAdapterCfg,
)


class _RobotStub:
    """Minimal fixed-base YAM surface used by the adapter contract tests."""

    def __init__(self, *, joint_pos: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor):
        identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        body_pos_w = torch.zeros(1, 2, 3)
        body_pos_w[:, 1, 2] = 0.80
        body_quat_w = identity_quat[:, None, :].repeat(1, 2, 1)
        self.data = SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=root_quat_w),
            body_pos_w=SimpleNamespace(torch=body_pos_w),
            body_quat_w=SimpleNamespace(torch=body_quat_w),
            joint_pos=SimpleNamespace(torch=joint_pos),
        )
        self.is_fixed_base = True
        self.num_base_dofs = 0

    def find_joints(self, _patterns, preserve_order=False):
        del preserve_order
        return list(range(6)), [f"joint{index}" for index in range(1, 7)]

    def find_bodies(self, _patterns, preserve_order=False):
        del preserve_order
        return [1], ["link_6"]


class _SceneStub(dict):
    """Dictionary scene with the environment-origin attribute used by adapters."""

    def __init__(self, **assets):
        super().__init__(assets)
        self.env_origins = torch.zeros(1, 3)


class _EnvStub:
    """Single-environment manager surface accepted by the AVP adapter."""

    def __init__(self):
        half_sqrt = 2.0**-0.5
        self.num_envs = 1
        self.device = "cpu"
        self.action_manager = SimpleNamespace(
            active_terms=("left_arm", "left_gripper", "right_arm", "right_gripper"),
            action_term_dim=(6, 1, 7, 1),
        )
        self.scene = _SceneStub(
            yam_left=_RobotStub(
                joint_pos=torch.tensor([[0.10, -0.20, 0.30, -0.40, 0.50, -0.60]]),
                root_pos_w=torch.tensor([[0.25, 0.10, 0.70]]),
                root_quat_w=torch.tensor([[0.0, 0.0, half_sqrt, half_sqrt]]),
            ),
            yam_right=_RobotStub(
                joint_pos=torch.tensor([[-0.10, 0.20, -0.30, 0.40, -0.50, 0.60]]),
                root_pos_w=torch.tensor([[-0.25, 0.15, 0.72]]),
                root_quat_w=torch.tensor([[0.0, 0.0, -half_sqrt, half_sqrt]]),
            ),
        )

    @property
    def unwrapped(self):
        return self


def _make_adapter() -> tuple[CableRoutingAVPActionAdapter, _EnvStub]:
    env = _EnvStub()
    return CableRoutingAVPActionAdapter(CableRoutingAVPActionAdapterCfg(), env), env


def _raw_command(
    right_pos=(0.20, -0.10, 1.00),
    right_quat=(0.0, 0.0, 0.0, 1.0),
    right_grip=-1.0,
) -> torch.Tensor:
    return torch.tensor([*right_pos, *right_quat, right_grip])


def test_avp_adapter_clutches_position_but_tracks_semantic_tool_orientation_absolutely() -> None:
    """Engagement cannot translate, while the hand-authored YAM tool basis remains absolute."""
    adapter, env = _make_adapter()
    first_raw = _raw_command(
        right_pos=(0.20, -0.10, 1.00),
        # Fingers point world +X and the hand plane spans world X/Y. Contact
        # +X follows index-to-little (world -Y), choosing the negative roll.
        right_quat=(-0.5, 0.5, -0.5, 0.5),
        right_grip=-1.0,
    )

    first_action = adapter.process(first_raw)

    assert adapter.RAW_ACTION_DIM == 8
    assert adapter.ENV_ACTION_DIM == 15
    assert first_action.shape == (15,)
    torch.testing.assert_close(first_action[0:6], env.scene["yam_left"].data.joint_pos.torch[0])
    assert first_action[6].item() == adapter.cfg.gripper_open_action

    right = env.scene["yam_right"]
    contact_pos_w, _ = math_utils.combine_frame_transforms(
        right.data.body_pos_w.torch[:, 1],
        right.data.body_quat_w.torch[:, 1],
        adapter._contact_offset_pos,
        adapter._contact_offset_quat,
    )
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        right.data.root_pos_w.torch,
        right.data.root_quat_w.torch,
        contact_pos_w,
        first_raw[3:7].reshape(1, 4),
    )
    torch.testing.assert_close(first_action[7:14], torch.cat((target_pos_b, target_quat_b), dim=-1)[0])

    reconstructed_pos_w, reconstructed_quat_w = math_utils.combine_frame_transforms(
        right.data.root_pos_w.torch,
        right.data.root_quat_w.torch,
        first_action[7:10].reshape(1, 3),
        first_action[10:14].reshape(1, 4),
    )
    torch.testing.assert_close(reconstructed_pos_w, contact_pos_w)
    torch.testing.assert_close(reconstructed_quat_w, first_raw[3:7].reshape(1, 4))
    semantic_forward_w = math_utils.quat_apply(
        reconstructed_quat_w,
        torch.tensor([[0.0, 0.0, 1.0]]),
    )
    semantic_pad_tangent_w = math_utils.quat_apply(
        reconstructed_quat_w,
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    semantic_pad_normal_w = math_utils.quat_apply(
        reconstructed_quat_w,
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(semantic_forward_w, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(semantic_pad_tangent_w, torch.tensor([[0.0, -1.0, 0.0]]))
    torch.testing.assert_close(semantic_pad_normal_w, torch.tensor([[0.0, 0.0, -1.0]]))
    assert first_action[14].item() == adapter.cfg.gripper_close_action

    hand_delta_pos_w = torch.tensor([[0.02, -0.01, 0.03]])
    hand_delta_quat_w = torch.tensor([[0.13052619, 0.0, 0.0, 0.99144486]])
    second_hand_quat_w = math_utils.quat_mul(hand_delta_quat_w, first_raw[3:7].reshape(1, 4))
    second_raw = _raw_command(
        right_pos=tuple(first_raw[0:3] + hand_delta_pos_w[0]),
        right_quat=tuple(second_hand_quat_w[0]),
        right_grip=1.0,
    )
    second_action = adapter.process(second_raw)
    expected_pos_b, expected_quat_b = math_utils.subtract_frame_transforms(
        right.data.root_pos_w.torch,
        right.data.root_quat_w.torch,
        contact_pos_w + hand_delta_pos_w,
        second_hand_quat_w,
    )
    torch.testing.assert_close(second_action[7:14], torch.cat((expected_pos_b, expected_quat_b), dim=-1)[0])
    assert second_action[14].item() == adapter.cfg.gripper_open_action


def test_avp_adapter_never_uses_right_hand_to_move_left_yam_or_gripper() -> None:
    """Changing every right-hand value leaves the inactive YAM's absolute hold unchanged."""
    adapter, env = _make_adapter()
    first = adapter.process(_raw_command(right_grip=-1.0))

    # The articulation may move between manager steps, but the adapter must
    # continue emitting the reset-captured absolute target, not the live pose.
    env.scene["yam_left"].data.joint_pos.torch[:] = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    second = adapter.process(
        _raw_command(
            right_pos=(0.25, -0.05, 1.03),
            right_quat=(0.0, 0.0, 0.08715574, 0.99619470),
            right_grip=1.0,
        )
    )

    torch.testing.assert_close(first[0:7], second[0:7])
    torch.testing.assert_close(second[0:6], torch.tensor([0.10, -0.20, 0.30, -0.40, 0.50, -0.60]))
    assert second[6].item() == adapter.cfg.gripper_open_action
    assert second[14].item() == adapter.cfg.gripper_open_action


def test_avp_adapter_reset_recaptures_left_absolute_joint_hold() -> None:
    """An explicit environment reset adopts the newly reset left-YAM pose as its hold target."""
    adapter, env = _make_adapter()
    adapter.process(_raw_command())
    reset_home = torch.tensor([[-0.35, 0.25, -0.15, 0.05, 0.10, -0.20]])
    env.scene["yam_left"].data.joint_pos.torch[:] = reset_home

    adapter.reset()
    action = adapter.process(_raw_command())

    torch.testing.assert_close(action[0:6], reset_home[0])


def test_avp_tracking_loss_holds_then_reclutches_position_without_stale_orientation() -> None:
    """A dropout holds safely; reacquisition keeps position but restores semantic axes."""
    adapter, env = _make_adapter()
    adapter.process(_raw_command(right_grip=-1.0))
    right = env.scene["yam_right"]
    right.data.body_pos_w.torch[:, 1] += torch.tensor([[0.015, -0.025, 0.010]])
    invalid = _raw_command(right_grip=1.0)
    invalid[0:7] = torch.nan

    hold_action = adapter.process(invalid)
    reacquired_raw = _raw_command(
        right_pos=(-0.20, 0.30, 1.20),
        right_quat=(0.27059805, -0.27059805, 0.65328148, 0.65328148),
        right_grip=1.0,
    )
    reacquired_action = adapter.process(reacquired_raw)

    assert torch.isfinite(hold_action).all()
    torch.testing.assert_close(reacquired_action[7:10], hold_action[7:10])
    reconstructed_pos_w, reconstructed_quat_w = math_utils.combine_frame_transforms(
        right.data.root_pos_w.torch,
        right.data.root_quat_w.torch,
        reacquired_action[7:10].reshape(1, 3),
        reacquired_action[10:14].reshape(1, 4),
    )
    expected_hold_pos_w, _ = math_utils.combine_frame_transforms(
        right.data.root_pos_w.torch,
        right.data.root_quat_w.torch,
        hold_action[7:10].reshape(1, 3),
        hold_action[10:14].reshape(1, 4),
    )
    torch.testing.assert_close(reconstructed_pos_w, expected_hold_pos_w)
    torch.testing.assert_close(reconstructed_quat_w, reacquired_raw[3:7].reshape(1, 4))
    assert hold_action[14].item() == adapter.cfg.gripper_close_action
    assert reacquired_action[14].item() == adapter.cfg.gripper_open_action


@pytest.mark.parametrize("shape", [(7,), (9,), (2, 8), (1, 1, 8)])
def test_avp_adapter_rejects_raw_commands_outside_the_8d_contract(shape: tuple[int, ...]) -> None:
    """Malformed pipeline output fails before it can be routed into a manager action term."""
    adapter, _ = _make_adapter()
    with pytest.raises(ValueError, match="expects an 8-D command"):
        adapter.process(torch.zeros(shape))


def test_avp_adapter_prewarms_right_newton_ik_at_the_current_safe_hold() -> None:
    """CUDA-graph capture happens before the first live hand command is exposed."""
    adapter, env = _make_adapter()
    processed_actions = []
    right_term = SimpleNamespace(apply_actions=lambda: processed_actions.append("right_applied"))
    env.action_manager.process_action = lambda action: processed_actions.append(action.clone())
    env.action_manager.get_term = lambda name: right_term if name == "right_arm" else None

    returned_hold_action = adapter.prewarm()

    assert len(processed_actions) == 2
    hold_action = processed_actions[0]
    torch.testing.assert_close(returned_hold_action, hold_action[0])
    assert hold_action.shape == (1, adapter.ENV_ACTION_DIM)
    torch.testing.assert_close(hold_action[0, 0:6], env.scene["yam_left"].data.joint_pos.torch[0])
    assert hold_action[0, 6].item() == adapter.cfg.gripper_open_action
    assert hold_action[0, 14].item() == adapter.cfg.gripper_open_action
    assert processed_actions[1] == "right_applied"


def test_both_teleop_scripts_apply_and_reset_environment_action_adapters() -> None:
    """Playback and demonstration recording share process/reset and automatic-reset hooks."""
    repository_root = Path(__file__).resolve().parents[4]
    scripts = {
        "teleop_se3_agent.py": repository_root / "scripts/environments/teleoperation/teleop_se3_agent.py",
        "record_demos.py": repository_root / "scripts/tools/record_demos.py",
    }
    for name, path in scripts.items():
        source = path.read_text(encoding="utf-8")
        assert source.index("import warp.fem") < source.index("from isaaclab.app import AppLauncher"), name
        assert "def _create_teleop_action_adapter(" in source, name
        assert "teleop_action_adapter.process(action)" in source, name
        assert source.count("teleop_action_adapter.reset()") >= 4, name
        assert "teleop_action_adapter.prewarm()" in source, name
        assert "env.step(prewarm_action.repeat(env.num_envs, 1))" in source, name

    teleop_source = scripts["teleop_se3_agent.py"].read_text(encoding="utf-8")
    recorder_source = scripts["record_demos.py"].read_text(encoding="utf-8")
    assert "torch.any(terminated | truncated)" in teleop_source
    assert "torch.any(obv[2] | obv[3])" in recorder_source


def _find_argument_call(source: str, flag: str) -> ast.Call:
    """Return the parser.add_argument call that declares ``flag``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == flag:
            return node
    raise AssertionError(f"No add_argument call declares {flag}.")


def test_both_teleop_scripts_offer_explicit_safe_auto_start() -> None:
    """XR auto-start is opt-in while standalone/configured starts still request RUNNING."""
    repository_root = Path(__file__).resolve().parents[4]
    script_paths = (
        repository_root / "scripts/environments/teleoperation/teleop_se3_agent.py",
        repository_root / "scripts/tools/record_demos.py",
    )

    for path in script_paths:
        source = path.read_text(encoding="utf-8")
        call = _find_argument_call(source, "--auto_start_teleop")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.unparse(keywords["action"]) == "argparse.BooleanOptionalAction", path.name
        assert isinstance(keywords["default"], ast.Constant) and keywords["default"].value is False, path.name

        assert "not args_cli.xr" in source, path.name
        assert "args_cli.auto_start_teleop" in source, path.name
        assert "env_cfg.isaac_teleop.teleoperation_active_default" in source, path.name
        assert "if should_auto_start:" in source, path.name
        assert "teleop_interface.request_start()" in source, path.name


def _load_status_reporter(source: str):
    """Load the script's pure status reporter without importing and launching Kit."""
    tree = ast.parse(source)
    reporter_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_IsaacTeleopStatusReporter"
    )
    reporter_module = ast.Module(body=[reporter_node], type_ignores=[])
    ast.fix_missing_locations(reporter_module)
    namespace = {"torch": torch}
    exec(compile(reporter_module, filename="teleop_se3_agent.py", mode="exec"), namespace)
    return namespace["_IsaacTeleopStatusReporter"]


def test_teleop_status_reporter_tracks_only_the_right_8d_pose(capsys) -> None:
    """The single-hand stream reports right tracking transitions without phantom left-hand messages."""
    repository_root = Path(__file__).resolve().parents[4]
    source = (repository_root / "scripts/environments/teleoperation/teleop_se3_agent.py").read_text(encoding="utf-8")
    reporter_type = _load_status_reporter(source)
    reporter = reporter_type(report_bimanual_tracking=True)
    command = _raw_command()

    reporter.update(None, False)
    reporter.update(command, False)
    lost = command.clone()
    lost[0:7] = torch.nan
    reporter.update(lost, False)
    reporter.update(command, True)

    assert capsys.readouterr().out.splitlines() == [
        "IsaacTeleop status: WAITING_FOR_XR_SESSION",
        "IsaacTeleop status: XR_SESSION_READY",
        "IsaacTeleop status: TRACKING_RIGHT_ACQUIRED",
        "IsaacTeleop status: CONTROL_PAUSED",
        "IsaacTeleop status: TRACKING_RIGHT_LOST",
        "IsaacTeleop status: TRACKING_RIGHT_ACQUIRED",
        "IsaacTeleop status: CONTROL_RUNNING",
    ]
