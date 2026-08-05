# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused unit tests for the canonical Franka Pour reset-dataset MDP."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction
from isaaclab.managers import TerminationTermCfg
from isaaclab.test.mock_interfaces.assets import MockArticulation

from isaaclab_tasks.contrib.franka_pour import _state as state_utils
from isaaclab_tasks.contrib.franka_pour.mdp import observations, rewards, terminations
from isaaclab_tasks.contrib.franka_pour.mdp.actions import CurriculumGripperPositionAction
from isaaclab_tasks.contrib.franka_pour.mdp.actions_cfg import CurriculumGripperPositionActionCfg


class FakeTerminationManager:
    """Minimal vectorized termination manager."""

    def __init__(self, num_envs: int):
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.time_outs = torch.zeros(num_envs, dtype=torch.bool)
        self._terms = {"success": torch.zeros(num_envs, dtype=torch.bool)}

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms)

    @property
    def dones(self) -> torch.Tensor:
        return self.terminated | self.time_outs

    def get_term(self, name: str) -> torch.Tensor:
        return self._terms[name]


class FakeGripperAction:
    """Minimal symmetric gripper term with explicit contact state."""

    def __init__(self, num_envs: int):
        self.commanded_position = torch.full((num_envs, 1), 0.04)
        self.bilateral_contact = torch.zeros(num_envs, dtype=torch.bool)
        self.contact_deflection = torch.zeros((num_envs, 2))

    @property
    def contact_quality(self) -> torch.Tensor:
        return self.bilateral_contact.float()


class FakeActionManager:
    """Expose only the gripper action consumed by the MDP terms."""

    def __init__(self, num_envs: int):
        self._gripper = FakeGripperAction(num_envs)

    def get_term(self, name: str) -> FakeGripperAction:
        assert name == "gripper_action"
        return self._gripper


class FakeEnv:
    """Small vectorized state used by reset-learning and safety terms."""

    def __init__(self):
        self.num_envs = 4
        self.num_particles = 245
        self.device = "cpu"
        self.step_dt = 1.0 / 60.0
        self.max_episode_length = 12
        self.cfg = SimpleNamespace(
            cup_grasp_height=0.032,
            lost_grasp_dwell_time_s=0.05,
            max_spill_fraction=0.10,
            success_min_lift_height=0.05,
        )
        self.termination_manager = FakeTerminationManager(self.num_envs)
        self.action_manager = FakeActionManager(self.num_envs)
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self._success_dwell_count = torch.zeros(self.num_envs, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, dtype=torch.long)
        self._lifted_grasp_seen = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_dataset_row_id = torch.arange(self.num_envs, dtype=torch.long)
        self._reset_dataset_states = {"difficulty": torch.zeros(self.num_envs)}

        self.gripper_open_width = 0.08
        self.gripper_grasp_width = 0.06
        self.cup_reset_height = 0.0
        self.pour_target_frac = torch.full((self.num_envs,), 0.70)
        self._cup = torch.tensor([[0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self._target = torch.tensor([[0.50, -0.18, 0.00, 0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self._grasp = torch.tensor([[0.50, 0.00, 0.032]]).repeat(self.num_envs, 1)
        self._tcp = self._grasp + torch.tensor([0.20, 0.0, 0.0])
        self._tcp_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self._width = torch.full((self.num_envs,), self.gripper_open_width)
        self._source_count = torch.tensor([245.0, 183.0, 98.0, 0.0])
        self._target_count = torch.tensor([0.0, 61.0, 122.0, 233.0])
        self._spill_count = torch.tensor([0.0, 0.0, 24.0, 12.0])

    @property
    def gripper(self) -> FakeGripperAction:
        return self.action_manager.get_term("gripper_action")

    def tcp_pos_e(self) -> torch.Tensor:
        return self._tcp

    def tcp_pose_e(self) -> torch.Tensor:
        return torch.cat((self._tcp, self._tcp_quat), dim=-1)

    def cup_pose_e(self) -> torch.Tensor:
        return self._cup

    def target_pose_e(self) -> torch.Tensor:
        return self._target

    def cup_grasp_point_e(self) -> torch.Tensor:
        return self._grasp

    def desired_grasp_tcp_quat_c(self) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)

    def gripper_width(self) -> torch.Tensor:
        return self._width

    def count_in_source(self) -> torch.Tensor:
        return self._source_count

    def count_in_target(self) -> torch.Tensor:
        return self._target_count

    def count_spilled(self) -> torch.Tensor:
        return self._spill_count

    def spilled_fraction(self) -> torch.Tensor:
        return self._spill_count / self.num_particles


def _physical_progress_params() -> dict[str, float]:
    return {
        "approach_position_std": 0.20,
        "approach_orientation_std": 0.75,
        "approach_open_hand_fraction": 0.15,
        "grasp_target_height": 0.10,
        "grasp_reach_std": 0.025,
        "grasp_preload_position": 0.024,
        "grasp_fraction": 0.40,
        "capture_orientation_tolerance": math.radians(25.0),
        "capture_intent_gain": 3.0,
        "source_mouth_height": 0.119,
        "target_rim_height": 0.074,
        "target_clearance_height": 0.15,
        "alignment_std": 0.20,
        "tilt_alignment_radius": 0.15,
        "target_tilt": math.radians(140.0),
    }


def _learning_progress_params() -> dict[str, object]:
    potential_params = _physical_progress_params()
    potential_params.pop("capture_orientation_tolerance")
    potential_params.pop("capture_intent_gain")
    return {
        "potential_params": potential_params,
        "minimum_progress": 0.08,
        "minimum_episode_steps": 3,
    }


def _place_held_pose(env: FakeEnv, tilt: float) -> None:
    """Place every source cup in an exact held pose above the receiver."""
    source_mouth_height = 0.119
    target_rim_height = 0.074
    target_clearance_height = 0.16
    source_grasp_height = 0.083
    source_quaternion = torch.tensor([math.sin(0.5 * tilt), 0.0, 0.0, math.cos(0.5 * tilt)])
    open_axis = torch.tensor([0.0, -math.sin(tilt), math.cos(tilt)])
    mouth_position = env._target[0, :3].clone()
    mouth_position[2] += target_rim_height + target_clearance_height
    source_position = mouth_position - source_mouth_height * open_axis
    grasp_position = source_position + source_grasp_height * open_axis

    env._cup[:] = torch.cat((source_position, source_quaternion))
    env._grasp[:] = grasp_position
    env._tcp[:] = grasp_position
    env._tcp_quat[:] = source_quaternion
    env._width[:] = env.gripper_grasp_width
    env.gripper.commanded_position[:] = 0.024
    env.gripper.bilateral_contact[:] = True


def test_terminal_failure_has_success_precedence_and_can_exclude_timeouts():
    env = FakeEnv()
    env.termination_manager.terminated[:] = torch.tensor([True, True, False, False])
    env.termination_manager.time_outs[:] = torch.tensor([False, False, True, False])
    env.termination_manager._terms["success"][:] = torch.tensor([False, True, False, False])

    all_failures = rewards.terminal_failure(env) * env.step_dt
    true_failures = rewards.terminal_failure(env, include_time_out=False) * env.step_dt

    torch.testing.assert_close(all_failures, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(true_failures, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_immediate_success_uses_current_fraction_with_failure_precedence_and_245_boundary():
    env = FakeEnv()
    env._target_count[:] = torch.tensor([171.0, 172.0, 245.0, 245.0])
    env.termination_manager.terminated[3] = True

    success = terminations.immediate_pour_success(env)

    assert success.tolist() == [False, True, True, False]
    assert env.episode_succeeded.tolist() == [False, True, True, False]
    boundary = terminations.immediate_pour_success_mask(
        torch.tensor([171.0 / 245.0, 172.0 / 245.0]),
        0.70,
        torch.zeros(2, dtype=torch.bool),
    )
    assert boundary.tolist() == [False, True]


def test_unsuccessful_timeout_excludes_same_step_success():
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([12, 12, 11, 12]),
        max_episode_length=12,
        episode_succeeded=torch.tensor([True, False, False, True]),
    )

    assert terminations.unsuccessful_time_out(env).tolist() == [False, True, False, False]


def test_spill_threshold_is_strict_and_monitor_mode_never_terminates():
    env = FakeEnv()
    env._spill_count[:] = torch.tensor([0.0, 24.0, 25.0, 245.0])

    assert terminations.excessive_spill(env).tolist() == [False, False, True, True]
    assert not bool(torch.any(terminations.excessive_spill(env, terminate=False)))


def test_lost_grasp_monitor_tracks_dwell_before_optional_termination():
    env = FakeEnv()
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env.gripper.commanded_position[:] = 0.024
    env.gripper.bilateral_contact[:] = True
    terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt, terminate=False)

    env._tcp[1, 0] += 0.10
    env.gripper.bilateral_contact[1] = False
    for _ in range(3):
        monitored = terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt, terminate=False)

    assert not bool(torch.any(monitored))
    assert env._lifted_grasp_seen.tolist() == [True, True, True, True]
    assert env._lost_grasp_dwell_count.tolist() == [0, 3, 0, 0]
    terminated = terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)
    assert terminated.tolist() == [False, True, False, False]


def test_actor_geometry_observations_are_cup_relative_and_quaternion_sign_invariant():
    half_sqrt = math.sqrt(0.5)
    cup_pose = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt],
        ]
    )
    tcp_pose = torch.tensor(
        [
            [1.0, 2.0, 0.03, 0.0, 0.0, 0.0, -1.0],
            [1.02, 2.01, 0.03, 0.0, 0.0, -half_sqrt, -half_sqrt],
        ]
    )
    target_pose = torch.tensor(
        [
            [1.2, 2.1, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.9, 2.2, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(cup_grasp_height=0.03),
        cup_pose_e=lambda: cup_pose,
        tcp_pose_e=lambda: tcp_pose,
        tcp_pos_e=lambda: tcp_pose[:, :3],
        target_pose_e=lambda: target_pose,
        desired_grasp_tcp_quat_c=lambda: torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    )

    torch.testing.assert_close(
        observations.tcp_to_grasp_position_c_obs(env),
        torch.tensor([[0.0, 0.0, 0.0], [-0.01, 0.02, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        observations.grasp_to_tcp_quat_obs(env),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        observations.target_position_c_obs(env),
        torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert bool(torch.all(observations.tcp_pose_obs(env)[:, 6] >= 0.0))


def test_actor_gripper_and_horizon_observations_preserve_current_state():
    env = FakeEnv()
    env.episode_length_buf[:] = torch.tensor([0, 3, 6, 12])
    env.pour_target_frac[:] = torch.tensor([0.1, 0.2, 0.4, 0.7])
    env.gripper.commanded_position[:] = torch.tensor([[0.021], [0.024], [0.030], [0.040]])
    env.gripper.contact_deflection[:] = torch.tensor([[0.002, 0.002], [0.003, 0.0002], [0.0, 0.0], [0.001, 0.004]])
    finger_position = torch.tensor([[0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.04]])
    finger_velocity = torch.tensor([[0.1, -0.2], [0.0, 0.0], [0.3, -0.4], [0.1, 0.1]])
    env.finger_joint_pos = lambda: finger_position
    env.finger_joint_vel = lambda: finger_velocity

    torch.testing.assert_close(observations.time_remaining_obs(env).flatten(), torch.tensor([1.0, 0.75, 0.5, 0.0]))
    torch.testing.assert_close(observations.pour_target_fraction_obs(env).flatten(), env.pour_target_frac)
    torch.testing.assert_close(observations.finger_position_obs(env), finger_position)
    torch.testing.assert_close(observations.finger_velocity_obs(env), finger_velocity)
    torch.testing.assert_close(observations.gripper_target_obs(env), env.gripper.commanded_position)
    torch.testing.assert_close(observations.gripper_contact_obs(env), env.gripper.contact_deflection)


def test_gripper_action_reuses_binary_mapping_with_identical_filter_and_reset_state():
    robot = MockArticulation(
        num_instances=2,
        num_joints=2,
        num_bodies=1,
        joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        device="cpu",
    )
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu")
    alpha = 1.0 - (1.0 - 0.2) ** (1.0 / 3.0)
    cfg = CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        alpha=alpha,
        close_position=0.021,
        neutral_position=0.04,
        default_position=0.024,
    )

    action = CurriculumGripperPositionAction(cfg, env)
    assert isinstance(action, BinaryJointPositionAction)
    raw = torch.tensor(((-1.0,), (1.0,)))
    action.process_actions(raw)
    initial = torch.full((2, 2), 0.024)
    expected = initial.lerp(torch.tensor(((0.021, 0.021), (0.04, 0.04))), alpha)
    torch.testing.assert_close(action.raw_actions, raw)
    torch.testing.assert_close(action.processed_actions, expected)

    action.set_reset_position(torch.tensor(((0.03,),)), env_ids=torch.tensor((1,)))
    torch.testing.assert_close(action.processed_actions[1], torch.tensor((0.03, 0.03)))
    action.reset(torch.tensor((1,)))
    torch.testing.assert_close(action.raw_actions[1], torch.zeros(1))
    torch.testing.assert_close(action.processed_actions[1], torch.tensor((0.03, 0.03)))

    robot.data.joint_pos.torch[:] = action.processed_actions + torch.tensor(((0.002, 0.001), (0.002, 0.0005)))
    assert action.bilateral_contact.tolist() == [True, False]
    torch.testing.assert_close(action.contact_quality, torch.tensor((1.0, 0.5)))


def test_media_velocity_and_fraction_observations_are_bounded_and_permutation_invariant():
    env = FakeEnv()
    velocity = torch.tensor(
        [
            [0.25, -0.5, 0.0, 1.0, -2.0, 0.0],
            [float("inf"), 0.0, 0.0, 0.0, float("nan"), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    env.cup_velocity_w = lambda: velocity

    normalized_velocity = observations.normalized_cup_velocity_obs(env, max_surface_speed=0.5, surface_radius=0.13)
    fractions = observations.particle_fractions_obs(env)

    torch.testing.assert_close(normalized_velocity[0], torch.tensor([0.5, -1.0, 0.0, 0.26, -0.52, 0.0]))
    torch.testing.assert_close(normalized_velocity[1], torch.zeros(6))
    torch.testing.assert_close(
        fractions,
        torch.stack(
            (
                env._source_count / env.num_particles,
                env._target_count / env.num_particles,
                env._spill_count / env.num_particles,
                torch.zeros(env.num_envs),
            ),
            dim=-1,
        ),
    )
    torch.testing.assert_close(observations.held_delivery_history_obs(env), torch.zeros((env.num_envs, 1)))
    with pytest.raises(ValueError, match="surface_radius"):
        observations.normalized_cup_velocity_obs(env, surface_radius=0.0)


def test_particle_source_and_transfer_summaries_handle_rigid_motion_and_empty_sets():
    source = torch.tensor([[True, True, False, False], [False, False, False, False]])
    target = torch.tensor([[False, False, False, True], [True, True, True, True]])
    spilled = torch.zeros_like(source)
    positions = torch.zeros((2, 4, 3))
    positions[0, 0] = torch.tensor([1.012, 2.0, 3.06])
    positions[0, 1] = torch.tensor([0.988, 2.0, 3.06])
    positions[0, 2] = torch.tensor([1.30, 2.0, 3.0])
    velocities = torch.zeros_like(positions)
    velocities[0, :2] = torch.tensor([1.12, 0.0, 2.0])
    velocities[0, 2] = torch.tensor([2.0, 0.0, 0.0])
    cup_pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    cup_velocity = torch.zeros((2, 6))
    cup_velocity[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    target_pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env = SimpleNamespace(
        num_particles=4,
        particle_region_masks=lambda: (source, target, spilled),
        particle_pos_e=lambda: positions,
        particle_vel_e=lambda: velocities,
        cup_pose_e=lambda: cup_pose,
        cup_velocity_w=lambda: cup_velocity,
        target_pose_e=lambda: target_pose,
    )

    source_summary = observations.particle_source_state_obs(env)
    transfer_summary = observations.particle_transfer_obs(env)

    torch.testing.assert_close(source_summary[0], torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(source_summary[1], torch.zeros(6))
    torch.testing.assert_close(transfer_summary[0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]))
    torch.testing.assert_close(transfer_summary[1], torch.zeros(7))


def test_particle_masks_exclude_cups_and_enforce_workspace_bounds():
    points = torch.zeros((2, 4, 3))
    points[:, :, 2] = torch.tensor([0.003, 0.0031, 0.0, -0.01])
    in_source = torch.tensor([[False, False, True, False], [False, False, False, False]])
    in_target = torch.tensor([[False, False, False, True], [False, False, False, False]])

    spilled = state_utils.spilled_particle_mask(points, in_source, in_target, max_height=0.003)
    delivered = state_utils.delivered_particle_mask(in_source, in_target)

    assert spilled.tolist() == [[True, False, False, False], [True, False, True, True]]
    assert delivered.tolist() == [[False, False, False, True], [False, False, False, False]]
    assert not bool(torch.any(delivered & in_source))

    workspace_points = torch.zeros((3, 4, 3))
    workspace_points[0] = torch.tensor([0.5, 0.0, 0.2])
    workspace_points[1, 0] = torch.tensor([1.51, 0.0, 0.2])
    workspace_points[2, 0] = torch.tensor([0.5, 0.0, -0.51])
    inside = state_utils.particles_in_workspace(
        workspace_points,
        lower_bound=(-0.5, -1.0, -0.5),
        upper_bound=(1.5, 1.0, 1.5),
    )
    assert inside.tolist() == [True, False, False]


def test_reset_learning_progress_latches_local_advancement_without_terminating():
    env = FakeEnv()
    params = _learning_progress_params()
    term = rewards.PourResetLearningProgress(
        TerminationTermCfg(func=rewards.PourResetLearningProgress, params=params),
        env,
    )
    term.reset()

    env.episode_length_buf[:] = params["minimum_episode_steps"]
    term(env, **params)
    assert not bool(term.ever_success.any())

    env._tcp[:] = env._grasp
    no_termination = term(env, **params)
    assert not bool(torch.any(no_termination))
    assert bool(term.ever_success.all())

    env._tcp[:] = env._grasp + torch.tensor([0.20, 0.0, 0.0])
    term(env, **params)
    assert bool(term.ever_success.all())


def test_reset_learning_progress_requires_terminal_transfer_and_rejects_unsafe_progress():
    params = _learning_progress_params()
    terminal_env = FakeEnv()
    terminal_env._reset_dataset_states["difficulty"].fill_(0.99)
    _place_held_pose(terminal_env, tilt=float(params["potential_params"]["target_tilt"]))
    terminal_term = rewards.PourResetLearningProgress(
        TerminationTermCfg(func=rewards.PourResetLearningProgress, params=params),
        terminal_env,
    )
    terminal_term.reset()
    terminal_term(terminal_env, **params)
    assert not bool(terminal_term.ever_success.any())
    terminal_env.episode_succeeded[:] = True
    terminal_term(terminal_env, **params)
    assert bool(terminal_term.ever_success.all())

    unsafe_env = FakeEnv()
    unsafe_term = rewards.PourResetLearningProgress(
        TerminationTermCfg(func=rewards.PourResetLearningProgress, params=params),
        unsafe_env,
    )
    unsafe_term.reset()
    unsafe_env.episode_length_buf[:] = params["minimum_episode_steps"]
    unsafe_env._tcp[:] = unsafe_env._grasp
    unsafe_env.termination_manager.terminated[:] = True
    unsafe_term(unsafe_env, **params)
    assert not bool(unsafe_term.ever_success.any())


def test_physical_potential_ignores_dataset_labels_and_requires_contact_for_downstream_progress():
    env = FakeEnv()
    params = _physical_progress_params()
    _place_held_pose(env, tilt=params["target_tilt"])

    held = rewards._reset_dataset_physical_potential(env, **params)
    env.reset_dataset_row_id[:] = torch.tensor([3, 2, 1, 0])
    env._reset_dataset_states["difficulty"][:] = torch.tensor([1.0, 0.75, 0.25, 0.0])
    relabeled = rewards._reset_dataset_physical_potential(env, **params)
    env.gripper.bilateral_contact[:] = False
    unheld = rewards._reset_dataset_physical_potential(env, **params)

    torch.testing.assert_close(relabeled, held)
    assert bool(torch.all(held > 0.95))
    assert bool(torch.all(unheld <= 0.32 + 1.0e-6))
    assert bool(torch.all(held > unheld))


def test_reset_learning_progress_requires_bilateral_contact_at_grasp_crossing():
    env = FakeEnv()
    env._reset_dataset_states["difficulty"].fill_(0.21)
    env._tcp[:] = env._grasp
    params = _learning_progress_params()
    term = rewards.PourResetLearningProgress(
        TerminationTermCfg(func=rewards.PourResetLearningProgress, params=params),
        env,
    )
    term.reset()

    env.episode_length_buf[:] = params["minimum_episode_steps"]
    env._width[:] = env.gripper_grasp_width
    env.gripper.commanded_position[:] = 0.024
    term(env, **params)
    assert not bool(term.ever_success.any())

    env.gripper.bilateral_contact[:] = True
    term(env, **params)
    assert bool(term.ever_success.all())


def test_state_finite_and_termination_wrappers_reject_raw_nonfinite_values():
    robot_joint_pos = torch.zeros((6, 7))
    robot_joint_vel = torch.zeros((6, 7))
    tcp_body_q = torch.tensor([[0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]).repeat(6, 1)
    cup_body_q = tcp_body_q.clone()
    cup_lin_vel = torch.zeros((6, 3))
    cup_ang_vel = torch.zeros((6, 3))
    particle_pos = torch.zeros((6, 16, 3))
    robot_joint_pos[1, 0] = float("nan")
    robot_joint_vel[2, 0] = float("inf")
    tcp_body_q[3, 0] = float("nan")
    cup_lin_vel[4, 0] = float("inf")
    particle_pos[5, 0, 0] = float("nan")

    finite = state_utils.state_finite(
        robot_joint_pos,
        robot_joint_vel,
        tcp_body_q,
        cup_body_q,
        cup_lin_vel,
        cup_ang_vel,
        particle_pos,
    )

    assert finite.tolist() == [True, False, False, False, False, False]
    env = SimpleNamespace(state_finite=lambda: finite, rigid_state_in_bounds=lambda: finite)
    assert terminations.nonfinite_failure(env).tolist() == [False, True, True, True, True, True]
    assert terminations.extreme_rigid_state(env).tolist() == [False, True, True, True, True, True]


def test_rigid_state_bounds_reject_each_extreme_observation_source():
    count = 8
    robot_joint_pos = torch.zeros((count, 9))
    robot_joint_vel = torch.zeros((count, 9))
    joint_pos_limits = torch.tensor([[[-1.0, 1.0]]]).repeat(count, 9, 1)
    tcp_body_q = torch.tensor([[0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)
    cup_body_q = tcp_body_q.clone()
    cup_lin_vel = torch.zeros((count, 3))
    cup_ang_vel = torch.zeros((count, 3))
    env_origins = torch.zeros((count, 3))

    robot_joint_pos[1, 0] = 1.051
    robot_joint_vel[2, 0] = 20.01
    tcp_body_q[3, 0] = 1.501
    cup_body_q[4, 2] = -0.501
    cup_lin_vel[5, 0] = 10.01
    cup_ang_vel[6, 0] = 50.01
    cup_body_q[7, 3:7] = 0.0

    in_bounds = state_utils.rigid_state_in_bounds(
        robot_joint_pos,
        robot_joint_vel,
        joint_pos_limits,
        tcp_body_q,
        cup_body_q,
        cup_lin_vel,
        cup_ang_vel,
        env_origins,
        lower_bound=(-0.5, -1.0, -0.5),
        upper_bound=(1.5, 1.0, 1.5),
        joint_position_margin=0.05,
        max_joint_velocity=20.0,
        max_cup_linear_velocity=10.0,
        max_cup_angular_velocity=50.0,
    )

    assert in_bounds.tolist() == [True, False, False, False, False, False, False, False]
