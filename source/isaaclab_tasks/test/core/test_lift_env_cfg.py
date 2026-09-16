# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.adr_curriculum import CurriculumCfg
from isaaclab_tasks.core.lift.lift_env_cfg import RewardsCfg
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import (
    CableUniformPoseCommand,
    DeformableUniformPoseCommand,
    ObjectUniformPoseCommand,
)


class _MarkerSpy:
    def __init__(self, _cfg=None):
        self.calls: list[tuple[tuple, dict]] = []

    def set_visibility(self, _visible: bool) -> None:
        pass

    def visualize(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeScene(dict):
    def __init__(self, environment_ids: torch.Tensor, **assets):
        super().__init__(assets)
        self._ALL_INDICES = environment_ids
        self.env_origins = torch.zeros((len(environment_ids), 3))


def _tensor_data(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def test_point_cloud_noise_curriculum_is_symmetric() -> None:
    """Point-cloud noise should widen evenly around the uncorrupted observation."""
    curriculum = CurriculumCfg()

    minimum = curriculum.object_obs_unoise_min_adr.params["modify_params"]
    maximum = curriculum.object_obs_unoise_max_adr.params["modify_params"]

    assert minimum["initial_value"] == maximum["initial_value"] == 0.0
    assert minimum["final_value"] == -maximum["final_value"] == -0.01


def test_rigid_lift_regularizes_action_rate_and_joint_velocity() -> None:
    """Rigid Lift and Reorient should retain their legacy motion-smoothing schedule."""
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()

    assert rewards.action_rate.func is mdp.action_rate_l2
    assert rewards.action_rate.weight == pytest.approx(-1e-4)
    assert rewards.joint_vel.func is mdp.joint_vel_l2
    assert rewards.joint_vel.weight == pytest.approx(-1e-4)
    assert rewards.joint_vel.params["asset_cfg"].name == "robot"
    assert curriculum.action_rate.params == {
        "term_name": "action_rate",
        "weight": -1e-1,
        "num_steps": 10000,
    }
    assert curriculum.joint_vel.params == {
        "term_name": "joint_vel",
        "weight": -1e-1,
        "num_steps": 10000,
    }


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)


def test_lift_pose_markers_forward_environment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lift pose, goal, and success marker should retain its environment ownership."""
    num_envs = 3
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    root_pose_w = torch.cat((root_pos_w, identity_quat), dim=-1)

    robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        ),
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
            root_link_pose_w=SimpleNamespace(torch=root_pose_w),
        )
    )
    success_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_pos_w)))
    scene = _FakeScene(environment_ids, robot=robot, object=object_asset, table=success_asset)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
    cfg = SimpleNamespace(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="table",
        success_visualizer_cfg=object(),
        goal_pose_visualizer_cfg=object(),
        curr_pose_visualizer_cfg=object(),
        position_only=True,
        cmd_kind=None,
        element_names=None,
    )

    def _initialize_command_term(command, command_cfg, command_env) -> None:
        command.cfg = command_cfg
        command._env = command_env
        command.metrics = {}

    monkeypatch.setattr(CommandTerm, "__init__", _initialize_command_term)
    monkeypatch.setattr("isaaclab.markers.VisualizationMarkers", _MarkerSpy)

    command = ObjectUniformPoseCommand(cfg, env)
    command._set_debug_vis_impl(True)
    command._debug_vis_callback(None)
    command.cfg.position_only = False
    command._debug_vis_callback(None)
    command._update_metrics()
    DeformableUniformPoseCommand._update_metrics(command)
    command._segment_position_w = lambda: root_pos_w
    CableUniformPoseCommand._update_metrics(command)
    CableUniformPoseCommand._debug_vis_callback(command, None)

    expected_call_counts = {
        command.success_visualizer: 4,
        command.goal_visualizer: 3,
        command.curr_visualizer: 3,
    }
    for visualizer, expected_count in expected_call_counts.items():
        assert len(visualizer.calls) == expected_count
        for _, kwargs in visualizer.calls:
            assert torch.equal(kwargs["environment_ids"], environment_ids)


def test_lift_point_cloud_markers_repeat_environment_ids_per_point() -> None:
    """Flattened point-cloud markers should retain env-major ownership."""
    num_envs = 3
    num_points = 4
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 0] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    points_local = torch.arange(num_envs * num_points * 3, dtype=torch.float32).view(num_envs, num_points, 3)

    term = object.__new__(mdp.object_point_cloud_b)
    term.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.ref_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.points_local = points_local
    term.points_w = torch.zeros_like(points_local)
    term.visualizer = _MarkerSpy()
    env = SimpleNamespace(num_envs=num_envs)

    term(env, num_points=num_points, visualize=True)

    assert len(term.visualizer.calls) == 1
    _, kwargs = term.visualizer.calls[0]
    assert torch.equal(kwargs["translations"], term.points_w.view(-1, 3))
    assert torch.equal(kwargs["environment_ids"], torch.arange(num_envs).repeat_interleave(num_points))


def test_rigid_object_terms_handle_nonfinite_terminal_state() -> None:
    """A non-finite rigid-object transition should terminate with finite zero rewards."""
    num_envs = 2
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 3] = 1.0
    object_positions = torch.tensor([[0.5, 0.0, 0.2], [float("nan"), 0.0, 0.2]])
    object_quaternions = identity_quat.clone()
    object_quaternions[1, 0] = float("nan")
    object_velocities = torch.zeros((num_envs, 6))
    object_velocities[1, 0] = float("inf")
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=_tensor_data(object_positions.nan_to_num()[:, None, :]),
            root_pos_w=_tensor_data(torch.zeros((num_envs, 3))),
            root_quat_w=_tensor_data(identity_quat),
            root_link_quat_w=_tensor_data(identity_quat),
        )
    )
    rigid_object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_tensor_data(object_positions),
            root_quat_w=_tensor_data(object_quaternions),
            root_vel_w=_tensor_data(object_velocities),
        )
    )
    scene = _FakeScene(environment_ids, robot=robot, object=rigid_object)
    contact_forces = torch.zeros((num_envs, 3))
    contact_forces[:, 0] = 1.0
    contact_sensor = SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=_tensor_data(contact_forces)))
    scene.sensors = {"thumb": contact_sensor, "finger": contact_sensor}
    command = torch.zeros((num_envs, 7))
    command[:, :3] = object_positions.nan_to_num()
    command[:, 3:] = identity_quat
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda _name: command),
    )
    robot_cfg = SimpleNamespace(name="robot", body_ids=[0])
    object_cfg = SimpleNamespace(name="object")

    termination = object.__new__(mdp.out_of_bound)
    termination._object = rigid_object
    termination._origins = scene.env_origins
    termination._lower = scene.env_origins.clone()
    termination._upper = scene.env_origins.clone()
    termination._cached_axis = [None, None, None]
    assert torch.equal(
        termination(env, in_bound_range={"x": (0.0, 1.0), "y": (-0.5, 0.5), "z": (-0.02, 1.0)}),
        torch.tensor([False, True]),
    )

    success = object.__new__(mdp.success_reward)
    success.succeeded = torch.zeros(num_envs, dtype=torch.bool)
    position_progress = object.__new__(mdp.position_command_progress)
    position_progress.best_error = torch.full((num_envs,), float("inf"))
    position_progress._prev_command = None
    rewards = {
        "object_ee_distance": mdp.object_ee_distance(
            env,
            std=0.4,
            thumb_name="thumb",
            finger_names=["finger"],
            asset_cfg=robot_cfg,
            object_cfg=object_cfg,
        ),
        "success": success(
            env,
            command_name="object_pose",
            asset_cfg=robot_cfg,
            align_asset_cfg=object_cfg,
            pos_std=0.05,
            rot_std=0.5,
            thumb_name="thumb",
            finger_names=["finger"],
        ),
        "position_progress": position_progress(
            env,
            command_name="object_pose",
            asset_cfg=robot_cfg,
            align_asset_cfg=object_cfg,
            min_improvement=0.0025,
            thumb_name="thumb",
            finger_names=["finger"],
        ),
    }
    for reward in rewards.values():
        assert torch.isfinite(reward).all()
        assert reward[1] == 0.0
    assert torch.isinf(position_progress.best_error[1])
