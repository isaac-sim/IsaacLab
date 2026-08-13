# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless configuration tests for the bimanual cable-routing environments."""

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoJointCfg, MujocoRigidBodyCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

from pxr import UsdUtils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

from isaaclab_contrib.coupling import CouplerProxyCfg
from isaaclab_contrib.deformable import VBDSolverCfg

import isaaclab_tasks.contrib.cable_routing  # noqa: F401
from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (
    BOARD_SIZE,
    BOARD_THICKNESS,
    BOARD_TOP_Z,
    BOARD_USD_PATH,
    CABLE_BEND_MODULUS,
    CABLE_LENGTH,
    CABLE_NUM_SEGMENTS,
    CABLE_SEGMENT_LENGTH,
    CABLE_STRETCH_MODULUS,
    CABLE_THICKNESS,
    ROUND_PEG_USD_PATH,
    ROUTE_AXIAL_CUTOFF,
    TABLE_TOP_Z,
    YAM_BASE_COLLISION_DEPTH,
    YAM_BASE_Z,
    YAM_BOARD_GAP,
    YAM_FRONT_X,
    YAM_GRIPPER_CLOSED_POS,
    YAM_GRIPPER_OPEN_POS,
    YAM_LATERAL_OFFSET,
    YAM_USD_PATH,
    YAM_VISUAL_BASE_DEPTH,
    YAM_VISUAL_BASE_WIDTH,
    CableRoutingEnvCfg,
    CableRoutingPeg0CCWEnvCfg,
    CableRoutingPeg1CWEnvCfg,
    CableRoutingSevenGoalsEnvCfg,
    CableRoutingTier1PegsEnvCfg,
)
from isaaclab_tasks.contrib.cable_routing.mdp.commands import (
    CableRoutingCommand,
    CableRoutingCommandCfg,
    _route_goal_marker_data,
)

_TASK_CONFIGS = {
    "IsaacContrib-CableRouting-YAM": "CableRoutingEnvCfg",
    "IsaacContrib-CableRouting-YAM-Peg0-CCW": "CableRoutingPeg0CCWEnvCfg",
    "IsaacContrib-CableRouting-YAM-Peg1-CW": "CableRoutingPeg1CWEnvCfg",
    "IsaacContrib-CableRouting-YAM-Tier1-Pegs": "CableRoutingTier1PegsEnvCfg",
    "IsaacContrib-CableRouting-YAM-SevenGoals": "CableRoutingSevenGoalsEnvCfg",
}


def test_cable_routing_gym_registrations_use_manager_based_env_and_shared_agent() -> None:
    """Test the base and staged task registrations."""
    for task_id, config_name in _TASK_CONFIGS.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.disable_env_checker
        assert spec.kwargs == {
            "env_cfg_entry_point": (f"isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg:{config_name}"),
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.contrib.cable_routing.agents.rsl_rl_ppo_cfg:CableRoutingPPORunnerCfg"
            ),
        }


def test_cable_routing_actions_are_relative_joint_position_with_binary_grippers() -> None:
    """Test the two six-joint arms and two scalar gripper commands form 14 policy actions."""
    cfg = CableRoutingEnvCfg()
    arm_actions = (cfg.actions.left_arm, cfg.actions.right_arm)
    gripper_actions = (cfg.actions.left_gripper, cfg.actions.right_gripper)

    assert all(isinstance(action, RelativeJointPositionActionCfg) for action in arm_actions)
    assert not any(
        isinstance(action, (DifferentialInverseKinematicsActionCfg, OperationalSpaceControllerActionCfg))
        for action in (*arm_actions, *gripper_actions)
    )
    for action, asset_name in zip(arm_actions, ("yam_left", "yam_right")):
        assert action.asset_name == asset_name
        assert action.joint_names == ["joint[1-6]"]
        assert action.scale == 0.04
        assert action.use_zero_offset
        assert action.preserve_order

    assert all(isinstance(action, BinaryJointPositionActionCfg) for action in gripper_actions)
    for action, asset_name in zip(gripper_actions, ("yam_left", "yam_right")):
        assert action.asset_name == asset_name
        assert action.joint_names == ["left_finger"]
        assert YAM_GRIPPER_OPEN_POS == 0.0375
        assert YAM_GRIPPER_CLOSED_POS == 0.0
        assert action.open_command_expr == {"left_finger": YAM_GRIPPER_OPEN_POS}
        assert action.close_command_expr == {"left_finger": YAM_GRIPPER_CLOSED_POS}

    resolved_policy_action_dim = 6 + 1 + 6 + 1
    assert resolved_policy_action_dim == 14


def test_cable_routing_manager_terms_use_menagerie_joint_and_body_names() -> None:
    """Test observations, events, and rewards resolve against the canonical YAM names."""
    cfg = CableRoutingEnvCfg()

    active_geometry = cfg.observations.policy.active_geometry.params
    assert active_geometry["left_ee_cfg"].body_names == ["link_6"]
    assert active_geometry["right_ee_cfg"].body_names == ["link_6"]

    expected_proprio_joints = ["joint[1-6]", "left_finger", "right_finger"]
    for term in (
        cfg.observations.proprio.left_joint_pos,
        cfg.observations.proprio.left_joint_vel,
        cfg.observations.proprio.right_joint_pos,
        cfg.observations.proprio.right_joint_vel,
    ):
        assert term.params["asset_cfg"].joint_names == expected_proprio_joints

    for term in (
        cfg.events.reset_left_arm,
        cfg.events.reset_right_arm,
        cfg.rewards.left_joint_velocity,
        cfg.rewards.right_joint_velocity,
    ):
        assert term.params["asset_cfg"].joint_names == ["joint[1-6]"]

    assert cfg.events.reset_cable.params["full_scene_replay_command_name"] == "route"


def test_cable_routing_rewards_are_sparse_success_with_penalties_only() -> None:
    """Test every route variant excludes dense geometric reward shaping."""
    expected_reward_names = {
        "success",
        "stretch",
        "action_rate",
        "left_joint_velocity",
        "right_joint_velocity",
    }
    configs = (
        CableRoutingEnvCfg(),
        CableRoutingPeg0CCWEnvCfg(),
        CableRoutingPeg1CWEnvCfg(),
        CableRoutingTier1PegsEnvCfg(),
        CableRoutingSevenGoalsEnvCfg(),
    )

    for cfg in configs:
        active_reward_names = {
            name for name, term in vars(cfg.rewards).items() if not name.startswith("_") and term is not None
        }
        assert active_reward_names == expected_reward_names


def test_cable_routing_goal_variants_use_seven_padded_route_programs() -> None:
    """Test seven route choices, staged subsets, and the fixed 18-D encoder input."""
    base = CableRoutingEnvCfg()
    command = base.commands.route

    assert command.route_options == (
        ((0, -1),),
        ((1, 1),),
        ((0, -1), (1, 1)),
        ((0, 1),),
        ((1, -1),),
        ((0, 1), (1, 1)),
        ((0, -1), (1, -1)),
    )
    assert command.allowed_route_ids == (0, 1, 2)
    assert command.max_route_steps == 3
    assert command.max_route_steps * 6 == 18
    assert command.radial_cutoff == 0.05
    assert command.axial_cutoff == ROUTE_AXIAL_CUTOFF

    variants = (
        (CableRoutingPeg0CCWEnvCfg(), (0,)),
        (CableRoutingPeg1CWEnvCfg(), (1,)),
        (CableRoutingTier1PegsEnvCfg(), (2,)),
        (CableRoutingSevenGoalsEnvCfg(), tuple(range(7))),
    )
    for cfg, allowed_route_ids in variants:
        assert cfg.commands.route.route_options == command.route_options
        assert cfg.commands.route.allowed_route_ids == allowed_route_ids
        assert cfg.commands.route.max_route_steps * 6 == 18


@pytest.mark.parametrize(
    ("route_options", "allowed_route_ids", "message"),
    (
        ((((0, -1),),), (0, 0), "allowed_route_ids"),
        ((((0, -1), (0, -1)),), (0,), "cannot repeat a peg"),
        ((((0, -1), (1, 1)), ((1, 1), (0, -1))), (0, 1), "distinct terminal"),
    ),
)
def test_cable_routing_route_programs_reject_ambiguous_goal_definitions(
    route_options: tuple[tuple[tuple[int, int], ...], ...],
    allowed_route_ids: tuple[int, ...],
    message: str,
) -> None:
    """Routes must map one-to-one onto physically representable terminal goals."""
    with pytest.raises(ValueError, match=message):
        CableRoutingCommandCfg(route_options=route_options, allowed_route_ids=allowed_route_ids)


def test_cable_routing_post_step_command_update_does_not_duplicate_route_refresh(monkeypatch) -> None:
    """The command update reuses route geometry already refreshed for this policy step."""
    command = object.__new__(CableRoutingCommand)
    command._env = SimpleNamespace(common_step_counter=7)
    command._route_state_step = 7

    def unexpected_refresh(*args, **kwargs) -> None:
        raise AssertionError("_update_command must not recompute route geometry")

    monkeypatch.setattr(command, "refresh_route_state", unexpected_refresh)
    assert command._update_command() is None


def test_cable_routing_command_masks_nonfinite_geometry_before_route_rewards() -> None:
    """A numerically failed cable row terminates with finite command state and zero progress reward."""
    command = object.__new__(CableRoutingCommand)
    cable_points = torch.tensor(
        (
            ((-0.04, 0.0, 0.02), (0.0, 0.04, 0.02), (0.04, 0.0, 0.02)),
            ((-0.04, 0.0, 0.02), (torch.nan, 0.04, 0.02), (0.04, 0.0, 0.02)),
        )
    )
    segment_poses = torch.nn.functional.pad(cable_points, (0, 4))

    def tensor_proxy(value: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(torch=value)

    command.cable = SimpleNamespace(data=SimpleNamespace(segment_pose_w=tensor_proxy(segment_poses)))
    peg_positions = torch.tensor(((0.0, 0.0, 0.02), (0.0, 0.0, 0.02)))
    command.pegs = [
        SimpleNamespace(data=SimpleNamespace(root_pos_w=tensor_proxy(peg_positions))),
        SimpleNamespace(data=SimpleNamespace(root_pos_w=tensor_proxy(peg_positions + torch.tensor((0.2, 0.0, 0.0))))),
    ]
    command.cfg = SimpleNamespace(
        radial_cutoff=0.08,
        axial_cutoff=0.05,
        maximum_local_cable_length=1.0,
        completion_winding=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
    )
    command.peg_indices = torch.zeros((2, 1), dtype=torch.long)
    command.directions = torch.ones((2, 1))
    command.valid_steps = torch.ones((2, 1), dtype=torch.bool)
    command.directed_progress = torch.zeros((2, 1))
    command.completed_steps = torch.zeros((2, 1), dtype=torch.bool)
    command.prefix_length = torch.zeros(2, dtype=torch.long)
    command.succeeded = torch.zeros(2, dtype=torch.bool)
    command.active_step = torch.zeros(2, dtype=torch.long)
    command.winding = torch.zeros((2, 2))
    command.route_progress_delta = torch.zeros(2)
    command._reward_progress_baseline = torch.tensor((0.25, 0.75))
    command.route_id = torch.zeros(2, dtype=torch.long)
    command.metrics = {
        name: torch.zeros(2)
        for name in (
            "route_progress",
            "success",
            "completed_fraction",
            "active_directed_winding_rad",
            "max_abs_winding_rad",
            "route_id",
        )
    }

    command.refresh_route_state(update_reward_delta=True)

    state_tensors = (
        command.winding,
        command.directed_progress,
        command.route_progress_delta,
        command._reward_progress_baseline,
        *command.metrics.values(),
    )
    assert all(bool(torch.isfinite(value).all()) for value in state_tensors)
    assert command.route_progress_delta[1] == 0.0
    assert not bool(command.succeeded[1])


def test_cable_routing_command_reuses_cached_authored_rest_length(monkeypatch) -> None:
    """Reset-bank chunks must not rebuild or synchronize the authored cable spacing."""
    command = object.__new__(CableRoutingCommand)
    command._cable_rest_length_m = CABLE_SEGMENT_LENGTH

    def unexpected_tensor_creation(*args, **kwargs) -> None:
        raise AssertionError("cached cable rest length must not recreate a tensor")

    monkeypatch.setattr(torch, "as_tensor", unexpected_tensor_creation)

    assert command._cable_rest_length() == CABLE_SEGMENT_LENGTH
    assert command._cable_rest_length() == CABLE_SEGMENT_LENGTH


def test_cable_routing_goal_markers_encode_order_state_and_direction() -> None:
    """Test dense marker data highlights active order and tangent CW/CCW motion."""
    peg_positions_w = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        ]
    )
    peg_indices = torch.tensor([[0, 1, 0], [1, 0, 0]])
    directions = torch.tensor([[-1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    valid_steps = torch.tensor([[True, True, False], [True, False, False]])
    completed_steps = torch.tensor([[True, False, False], [False, False, False]])
    active_step = torch.tensor([1, 0])

    markers = _route_goal_marker_data(
        peg_positions_w,
        peg_indices,
        directions,
        valid_steps,
        completed_steps,
        active_step,
        (0.040, 0.016, 0.035),
    )

    assert markers.step_positions_w.shape == (6, 3)
    assert markers.direction_positions_w.shape == (6, 3)
    assert markers.arc_positions_w.shape == (96, 3)
    assert markers.step_marker_indices.tolist() == [0, 1, 2, 1, 2, 2]
    assert markers.direction_marker_indices.tolist() == [2, 0, 1, 0, 1, 1]
    torch.testing.assert_close(markers.step_positions_w[0], torch.tensor([1.0, 2.0, 3.040]))
    torch.testing.assert_close(markers.step_positions_w[1], torch.tensor([4.0, 5.0, 6.056]))
    torch.testing.assert_close(
        markers.direction_positions_w[:, 0],
        markers.step_positions_w[:, 0] + 0.035,
    )

    sqrt_half = math.sqrt(0.5)
    torch.testing.assert_close(
        markers.direction_orientations_w[0],
        torch.tensor([0.0, 0.0, sqrt_half, sqrt_half]),
    )
    torch.testing.assert_close(
        markers.direction_orientations_w[1],
        torch.tensor([0.0, 0.0, -sqrt_half, sqrt_half]),
    )
    torch.testing.assert_close(markers.step_scales[:, 0], torch.tensor([0.75, 1.276, 0.0, 1.45, 0.0, 0.0]))
    torch.testing.assert_close(markers.direction_scales[:, 0], torch.tensor([0.65, 1.10, 0.0, 1.25, 0.0, 0.0]))
    assert markers.arc_marker_indices.shape == (96,)
    assert markers.arc_scales.shape == (96, 3)
    # The first ghost bead sits at +X on the first peg's target wrap.
    torch.testing.assert_close(markers.arc_positions_w[0], torch.tensor([1.030, 2.0, 3.022]))
    assert bool((markers.arc_scales[32:48] == 0.0).all())


def test_cable_routing_goal_marker_configs_use_newton_supported_prototypes() -> None:
    """Test the command uses primitive beads and arrow-x assets understood by Newton markers."""
    command = CableRoutingEnvCfg().commands.route

    assert not command.debug_vis
    assert command.route_step_visualizer_cfg.prim_path == "/Visuals/Command/cable_route_steps"
    assert list(command.route_step_visualizer_cfg.markers) == ["completed", "active", "pending"]
    assert all(isinstance(marker, sim_utils.SphereCfg) for marker in command.route_step_visualizer_cfg.markers.values())
    assert command.route_direction_visualizer_cfg.prim_path == "/Visuals/Command/cable_route_directions"
    assert list(command.route_direction_visualizer_cfg.markers) == ["clockwise", "counterclockwise", "completed"]
    assert all(
        isinstance(marker, sim_utils.UsdFileCfg) and marker.usd_path.endswith("/Props/UIElements/arrow_x.usd")
        for marker in command.route_direction_visualizer_cfg.markers.values()
    )
    assert command.route_arc_visualizer_cfg.prim_path == "/Visuals/Command/cable_route_target_arcs"
    assert list(command.route_arc_visualizer_cfg.markers) == ["clockwise", "counterclockwise", "completed"]
    assert all(isinstance(marker, sim_utils.SphereCfg) for marker in command.route_arc_visualizer_cfg.markers.values())

    play_cfg = CableRoutingEnvCfg()
    play_cfg.play_mode()
    assert play_cfg.commands.route.debug_vis


def test_cable_routing_table_board_and_yams_have_front_edge_reachable_placement() -> None:
    """Test the board rests on the table and both YAMs face it from the front."""
    scene = CableRoutingEnvCfg().scene

    assert isinstance(scene.table, RigidObjectCfg)
    assert scene.table.spawn.rigid_props.kinematic_enabled
    assert scene.table.spawn.size == (1.10, 0.80, TABLE_TOP_Z)
    assert scene.table.init_state.pos == (0.0, 0.0, 0.5 * TABLE_TOP_Z)
    table_top = scene.table.init_state.pos[2] + 0.5 * scene.table.spawn.size[2]
    assert table_top == TABLE_TOP_Z

    assert isinstance(scene.board, RigidObjectCfg)
    assert scene.board.spawn.rigid_props.kinematic_enabled
    assert isinstance(scene.board.spawn, sim_utils.UsdFileCfg)
    assert scene.board.spawn.usd_path == BOARD_USD_PATH
    assert not scene.board.spawn.copy_from_source
    board_bottom = scene.board.init_state.pos[2] - 0.5 * BOARD_THICKNESS
    board_top = scene.board.init_state.pos[2] + 0.5 * BOARD_THICKNESS
    assert math.isclose(board_bottom, table_top)
    assert math.isclose(board_top, BOARD_TOP_Z)
    for peg in (scene.peg_0, scene.peg_1):
        assert isinstance(peg.spawn, sim_utils.UsdFileCfg)
        assert peg.spawn.usd_path == ROUND_PEG_USD_PATH
        assert not peg.spawn.copy_from_source
        assert peg.spawn.rigid_props.kinematic_enabled

    left = scene.yam_left
    right = scene.yam_right
    assert left.spawn.usd_path == right.spawn.usd_path == YAM_USD_PATH
    assert not left.spawn.variants
    assert not right.spawn.variants
    assert left.spawn.func.__name__ == right.spawn.func.__name__ == "spawn_from_usd"
    for yam in (left, right):
        assert len(yam.spawn.rigid_props) == 1
        assert isinstance(yam.spawn.rigid_props[0], MujocoRigidBodyCfg)
        assert yam.spawn.rigid_props[0].gravcomp == 1.0
        assert len(yam.spawn.joint_drive_props) == 1
        assert isinstance(yam.spawn.joint_drive_props[0], MujocoJointCfg)
        assert yam.spawn.joint_drive_props[0].actuatorgravcomp is True
        assert set(yam.actuators) == {"arm", "gripper_drive", "gripper_passive"}
        assert yam.actuators["arm"].joint_names_expr == ["joint[1-6]"]
        assert yam.actuators["gripper_drive"].joint_names_expr == ["left_finger"]
        assert yam.actuators["gripper_drive"].stiffness == 2000.0
        assert yam.actuators["gripper_passive"].joint_names_expr == ["right_finger"]
        assert yam.actuators["gripper_passive"].stiffness == 0.0
        assert yam.init_state.joint_pos["left_finger"] == YAM_GRIPPER_OPEN_POS
        assert yam.init_state.joint_pos["right_finger"] == -YAM_GRIPPER_OPEN_POS
    assert math.isclose(YAM_BASE_Z - YAM_BASE_COLLISION_DEPTH, TABLE_TOP_Z)
    assert left.init_state.pos == (YAM_FRONT_X, YAM_LATERAL_OFFSET, YAM_BASE_Z)
    assert right.init_state.pos == (YAM_FRONT_X, -YAM_LATERAL_OFFSET, YAM_BASE_Z)
    board_front_edge = -0.5 * BOARD_SIZE[0]
    yam_board_facing_edge = YAM_FRONT_X + 0.5 * YAM_VISUAL_BASE_DEPTH
    assert math.isclose(YAM_BOARD_GAP, 0.15)
    assert math.isclose(board_front_edge - yam_board_facing_edge, YAM_BOARD_GAP)
    left_inner_edge = left.init_state.pos[1] - 0.5 * YAM_VISUAL_BASE_WIDTH
    right_inner_edge = right.init_state.pos[1] + 0.5 * YAM_VISUAL_BASE_WIDTH
    assert math.isclose(left_inner_edge, 0.5 * BOARD_SIZE[1])
    assert math.isclose(right_inner_edge, -0.5 * BOARD_SIZE[1])
    assert math.isclose(left.init_state.pos[1] + 0.5 * YAM_VISUAL_BASE_WIDTH, 0.5 * scene.table.spawn.size[1])
    assert math.isclose(right.init_state.pos[1] - 0.5 * YAM_VISUAL_BASE_WIDTH, -0.5 * scene.table.spawn.size[1])
    # Identity yaw keeps the 200 mm asset-local Y base axes parallel to the
    # board's 400 mm world-Y edge and points both arms' +X reach over the board.
    assert left.init_state.rot == right.init_state.rot == (0.0, 0.0, 0.0, 1.0)

    cfg = CableRoutingEnvCfg()
    cfg.play_mode()
    assert cfg.scene.num_envs == 4


def test_cable_routing_uses_complete_credential_free_yam_snapshot() -> None:
    """Test the default YAM is the pinned, dependency-complete Menagerie package."""
    yam_path = Path(YAM_USD_PATH)

    assert yam_path.is_file()
    assert yam_path.name == "i2rt_yam_default.usda"
    expected_hashes = {
        "i2rt_yam_default.usda": "c1bedf1d978d1147f82d1c2cb5e56da1b5003eb14ec78bd0be89258c021404bc",
        "i2rt_yam/i2rt_yam.usda": "d51c03bf78f1724c09ce46ae4c95cf64269871af25450fbc075bc2eb7cf08b55",
        "i2rt_yam/Payload/Contents.usda": "2d414f8266987073cc307feb750b3706393e7d2a2db34d6c27644ca570dbe79a",
        "i2rt_yam/Payload/Geometry.usda": "8116d69c971e70468ed7bd1662e5cc7869232aa96c61aeea44ba533624e504a5",
        "i2rt_yam/Payload/GeometryLibrary.usdc": ("46289c905bb19085b47b195976530aada09ea3fa04f79d741e083b34b0910993"),
        "i2rt_yam/Payload/Materials.usda": "6499795564ffa39af36488f0489e92e73f92fe59f5931b9dd4499fd16f4a0c99",
        "i2rt_yam/Payload/MaterialsLibrary.usdc": ("c20c4d119a844c7efbb74905589bef2bcab5eaeb0b5518f81d16f997d9da9adb"),
        "i2rt_yam/Payload/Physics.usda": "51f028b6a305fc788eb0286a7e67bf5d43cab54727ffcfeb20ef6502230b10c9",
    }
    usd_files = {path.relative_to(yam_path.parent).as_posix(): path for path in yam_path.parent.rglob("*.usd*")}
    assert usd_files.keys() == expected_hashes.keys()
    for relative_path, expected_hash in expected_hashes.items():
        assert hashlib.sha256(usd_files[relative_path].read_bytes()).hexdigest() == expected_hash

    for usd_path in yam_path.parent.rglob("*.usd*"):
        if usd_path.suffix == ".usda":
            contents = usd_path.read_text(encoding="utf-8")
            assert "omniverse://" not in contents
            assert "https://" not in contents

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(yam_path))
    assert assets == []
    assert unresolved == []
    package_root = yam_path.parent.resolve()
    resolved_layers = {Path(layer.identifier).resolve().relative_to(package_root).as_posix() for layer in layers}
    assert resolved_layers == {
        "i2rt_yam_default.usda",
        "i2rt_yam/Payload/Contents.usda",
        "i2rt_yam/Payload/Geometry.usda",
        "i2rt_yam/Payload/GeometryLibrary.usdc",
        "i2rt_yam/Payload/Materials.usda",
        "i2rt_yam/Payload/MaterialsLibrary.usdc",
        "i2rt_yam/Payload/Physics.usda",
    }


def test_cable_routing_uses_disjoint_newton_collision_ownership() -> None:
    """Test rigid fixtures use MJWarp and enter the cable's VBD view only as proxies."""
    cfg = CableRoutingEnvCfg()
    physics = cfg.sim.physics

    assert isinstance(physics, NewtonCfg)
    assert physics.num_substeps == 10
    assert physics.use_cuda_graph
    assert physics.default_shape_cfg.ke == 4.0e4
    assert physics.default_shape_cfg.gap == 0.001
    assert physics.default_shape_cfg.mu == 5.0

    coupler = physics.solver_cfg
    assert isinstance(coupler, CouplerProxyCfg)
    assert coupler.iterations == 1
    assert [entry.name for entry in coupler.entries] == ["rigid", "cable"]
    rigid_entry, cable_entry = coupler.entries
    assert isinstance(rigid_entry.solver_cfg, MJWarpSolverCfg)
    assert rigid_entry.solver_cfg.use_mujoco_contacts
    assert not rigid_entry.solver_cfg.disable_contacts
    assert rigid_entry.solver_cfg.njmax == 300
    assert rigid_entry.solver_cfg.nconmax == 200
    assert rigid_entry.bodies == [
        r"/World/envs/env_.*/YamLeft",
        r"/World/envs/env_.*/YamRight",
        r"/World/envs/env_.*/Table",
        r"/World/envs/env_.*/Board",
        r"/World/envs/env_.*/Peg0",
        r"/World/envs/env_.*/Peg1",
    ]
    assert not rigid_entry.include_static_shapes
    assert isinstance(cable_entry.solver_cfg, VBDSolverCfg)
    assert cable_entry.solver_cfg.iterations == 10
    assert cable_entry.solver_cfg.rigid_body_contact_buffer_size == 256
    assert cable_entry.bodies == [r"/World/envs/env_.*/Cable"]
    assert cable_entry.include_static_shapes

    assert len(coupler.proxies) == 1
    proxy = coupler.proxies[0]
    assert proxy.source == "rigid"
    assert proxy.destination == "cable"
    assert proxy.mode == "lagged"
    assert proxy.mass_scale == 1.0
    assert proxy.collide_interval == 2
    assert physics.num_substeps % proxy.collide_interval == 0
    assert proxy.bodies == [
        r"/World/envs/env_.*/Yam(Left|Right)",
        r"/World/envs/env_.*/(Table|Board)",
        r"/World/envs/env_.*/Peg(0|1)",
    ]
    assert coupler.model_cfg is None

    for asset_cfg, expected_friction in (
        (cfg.scene.yam_left, 5.0),
        (cfg.scene.yam_right, 5.0),
        (cfg.scene.table, 0.5),
        (cfg.scene.board, 0.5),
        (cfg.scene.peg_0, 0.5),
        (cfg.scene.peg_1, 0.5),
        (cfg.scene.ground, 0.5),
    ):
        usd_material, newton_material = asset_cfg.spawn.physics_material
        assert isinstance(usd_material, UsdPhysicsRigidBodyMaterialCfg)
        assert usd_material.static_friction == expected_friction
        assert usd_material.dynamic_friction == expected_friction
        assert isinstance(newton_material, NewtonMaterialCfg)
        assert newton_material.contact_stiffness == 4.0e4
        assert newton_material.contact_damping == 1.0e-5


def test_cable_routing_cable_matches_length_resolution_and_material_targets() -> None:
    """Test the 1 m smooth cable resolution and native Newton material configuration."""
    cable_cfg = CableRoutingEnvCfg().scene.cable.spawn
    material = cable_cfg.physics_material

    assert CABLE_LENGTH == 1.0
    assert CABLE_SEGMENT_LENGTH == 0.01
    assert CABLE_NUM_SEGMENTS == 100
    assert len(cable_cfg.positions) == CABLE_NUM_SEGMENTS + 1
    polyline_length = sum(math.dist(start, end) for start, end in zip(cable_cfg.positions, cable_cfg.positions[1:]))
    assert math.isclose(polyline_length, CABLE_LENGTH, abs_tol=1.0e-12)
    assert min(position[0] for position in cable_cfg.positions) >= -0.145
    assert max(position[0] for position in cable_cfg.positions) <= 0.145
    assert min(position[1] for position in cable_cfg.positions) >= -0.195
    assert max(position[1] for position in cable_cfg.positions) <= 0.195
    headings = [
        math.atan2(end[1] - start[1], end[0] - start[0])
        for start, end in zip(cable_cfg.positions, cable_cfg.positions[1:])
    ]
    heading_changes = [
        abs(math.atan2(math.sin(end - start), math.cos(end - start))) for start, end in zip(headings, headings[1:])
    ]
    assert max(heading_changes) <= math.pi / 12.0 + 1.0e-12
    assert math.dist(cable_cfg.positions[0], cable_cfg.positions[-1]) > 0.10
    assert material.thickness == CABLE_THICKNESS == 0.006
    assert material.density == 1200.0
    assert material.stretch_stiffness == CABLE_STRETCH_MODULUS
    assert material.bend_stiffness == CABLE_BEND_MODULUS


def test_cable_routing_reset_preserves_rest_shape_and_uses_matching_route_cutoff() -> None:
    """Test reset randomization cannot inject bend strain or use a mismatched topology metric."""
    cfg = CableRoutingEnvCfg()
    reset_params = cfg.events.reset_cable.params

    assert reset_params["max_heading_offset"] == 0.0
    assert reset_params["winding_radial_cutoff"] == cfg.commands.route.radial_cutoff == 0.05
    assert reset_params["winding_axial_cutoff"] == cfg.commands.route.axial_cutoff == ROUTE_AXIAL_CUTOFF
    assert reset_params["translation_jitter"] == ((-0.002, 0.002), (-0.002, 0.002))
    assert reset_params["yaw_jitter"] == (-0.02, 0.02)
