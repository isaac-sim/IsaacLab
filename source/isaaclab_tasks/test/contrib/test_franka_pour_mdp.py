# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka pour reward terms (no simulator)."""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg

import isaaclab_tasks.contrib.franka_pour.mdp as mdp_api
from isaaclab_tasks.contrib.franka_pour.mdp import observations, rewards, terminations


class FakeActionManager:
    def __init__(self, num_envs: int):
        self.action = torch.zeros((num_envs, 8))
        self._terms = {
            "gripper_action": SimpleNamespace(
                commanded_position=torch.full((num_envs, 1), 0.04),
                bilateral_preload=torch.ones(num_envs, dtype=torch.bool),
                bilateral_contact=torch.ones(num_envs, dtype=torch.bool),
                contact_deflection=torch.zeros((num_envs, 2)),
            )
        }

    def get_term(self, name: str):
        return self._terms[name]


def test_terminal_failure_pulses_once_for_overlapping_failure_predicates():
    termination_manager = FakeTerminationManager(4)
    termination_manager.terminated[:] = torch.tensor([True, True, False, False])
    termination_manager.time_outs[:] = torch.tensor([False, False, True, False])
    termination_manager._terms["success"][:] = torch.tensor([False, True, False, False])
    env = SimpleNamespace(step_dt=0.02, termination_manager=termination_manager)

    penalty = rewards.terminal_failure(env)

    torch.testing.assert_close(penalty, torch.tensor([50.0, 0.0, 50.0, 0.0]))


def test_general_reach_reward_is_bounded_monotonic_and_finite():
    env = SimpleNamespace(
        tcp_pos_e=lambda: torch.tensor(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (float("nan"), 0.0, 0.0))),
        cup_grasp_point_e=lambda: torch.zeros((3, 3)),
    )

    quality = rewards.tcp_cup_distance_tanh(env, std=1.0)

    assert quality[0] == 1.0
    assert 0.0 < quality[1] < quality[0]
    assert quality[2] == 0.0
    with pytest.raises(ValueError, match="std must be finite and positive"):
        rewards.tcp_cup_distance_tanh(env, std=0.0)


def test_general_media_goal_reward_measures_distance_to_target_cavity():
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0))
    target_pose = torch.cat((torch.zeros((3, 3)), identity.repeat(3, 1)), dim=-1)
    particles = torch.tensor(
        (
            ((0.0, 0.0, 0.05), (0.02, -0.02, 0.08)),
            ((0.20, 0.0, 0.05), (0.20, 0.0, 0.05)),
            ((float("inf"), 0.0, 0.05), (float("inf"), 0.0, 0.05)),
        )
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(particle_count_margin=0.0),
        _source_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        _target_inner_lo_t=torch.tensor((-0.05, -0.05, 0.01)),
        _target_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        cup_pose_e=lambda: target_pose,
        target_pose_e=lambda: target_pose,
        particle_pos_e=lambda: particles,
        particle_region_masks=lambda: tuple(torch.zeros((3, 2), dtype=torch.bool) for _ in range(3)),
    )

    quality = rewards.media_target_distance_tanh(env, std=1.0)

    assert quality[0] == 1.0
    assert 0.0 < quality[1] < quality[0]
    assert quality[2] == 0.0
    with pytest.raises(ValueError, match="std must be finite and positive"):
        rewards.media_target_distance_tanh(env, std=float("nan"))


def test_general_media_goal_reward_requires_release_from_nested_source():
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0))
    pose = torch.cat((torch.zeros(3), identity)).unsqueeze(0).repeat(2, 1)
    particles = torch.tensor((((0.0, 0.0, 0.05),), ((0.0, 0.0, 0.05),)))
    in_source = torch.tensor(((True,), (False,)))
    env = SimpleNamespace(
        cfg=SimpleNamespace(particle_count_margin=0.0),
        _source_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        _target_inner_lo_t=torch.tensor((-0.10, -0.10, 0.0)),
        _target_inner_hi_t=torch.tensor((0.10, 0.10, 0.15)),
        cup_pose_e=lambda: pose,
        target_pose_e=lambda: pose,
        particle_pos_e=lambda: particles,
        particle_region_masks=lambda: (
            in_source,
            ~in_source,
            torch.zeros_like(in_source),
        ),
    )

    quality = rewards.media_target_distance_tanh(env, std=0.1)

    assert quality[0] == pytest.approx(1.0 - math.tanh(0.5))
    assert quality[0] < quality[1]
    assert quality[1] == 1.0


def test_general_media_goal_reward_zeroes_spilled_particles():
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0))
    pose = torch.cat((torch.zeros(3), identity)).unsqueeze(0)
    particles = torch.tensor((((0.0, 0.0, 0.05), (0.0, 0.0, 0.05)),))
    false = torch.zeros((1, 2), dtype=torch.bool)
    env = SimpleNamespace(
        cfg=SimpleNamespace(particle_count_margin=0.0),
        _source_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        _target_inner_lo_t=torch.tensor((-0.10, -0.10, 0.0)),
        _target_inner_hi_t=torch.tensor((0.10, 0.10, 0.15)),
        cup_pose_e=lambda: pose,
        target_pose_e=lambda: pose,
        particle_pos_e=lambda: particles,
        particle_region_masks=lambda: (false, false, torch.tensor(((False, True),))),
    )

    quality = rewards.media_target_distance_tanh(env, std=0.1)

    assert quality[0] == pytest.approx(0.5)


def test_general_media_goal_reward_uses_rotated_target_frame():
    half_sqrt = math.sqrt(0.5)
    target_pose = torch.tensor(((0.0, 0.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt),))
    identity_pose = torch.tensor(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),))
    particles = torch.tensor((((0.0, 0.03, 0.05),),))
    false = torch.zeros((1, 1), dtype=torch.bool)
    env = SimpleNamespace(
        cfg=SimpleNamespace(particle_count_margin=0.0),
        _source_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        _target_inner_lo_t=torch.tensor((-0.02, -0.10, 0.0)),
        _target_inner_hi_t=torch.tensor((0.02, 0.10, 0.10)),
        cup_pose_e=lambda: identity_pose,
        target_pose_e=lambda: target_pose,
        particle_pos_e=lambda: particles,
        particle_region_masks=lambda: (false, false, false),
    )

    quality = rewards.media_target_distance_tanh(env, std=0.1)

    assert quality[0] == pytest.approx(1.0 - math.tanh(0.1))


def test_general_distance_rewards_are_independent_of_reset_stage():
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0))
    env = SimpleNamespace(
        cfg=SimpleNamespace(particle_count_margin=0.0),
        curriculum_stage=torch.tensor((15, 7, 6, 2)),
        _source_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        _target_inner_lo_t=torch.tensor((-0.05, -0.05, 0.01)),
        _target_inner_hi_t=torch.tensor((0.05, 0.05, 0.10)),
        tcp_pos_e=lambda: torch.full((4, 3), 0.1),
        cup_grasp_point_e=lambda: torch.zeros((4, 3)),
        cup_pose_e=lambda: torch.cat((torch.zeros((4, 3)), identity.repeat(4, 1)), dim=-1),
        target_pose_e=lambda: torch.cat((torch.zeros((4, 3)), identity.repeat(4, 1)), dim=-1),
        particle_pos_e=lambda: torch.full((4, 2, 3), 0.2),
        particle_region_masks=lambda: tuple(torch.zeros((4, 2), dtype=torch.bool) for _ in range(3)),
    )

    reach = rewards.tcp_cup_distance_tanh(env)
    goal_distance = rewards.media_target_distance_tanh(env)

    torch.testing.assert_close(reach, reach[0].expand_as(reach))
    torch.testing.assert_close(goal_distance, goal_distance[0].expand_as(goal_distance))


def test_general_joint_velocity_penalty_is_finite_on_invalid_terminal_state():
    velocity = torch.tensor(((1.0, 2.0), (float("nan"), float("inf")), (100.0, -100.0)))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=velocity)))}
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])

    penalty = rewards.finite_joint_velocity_l2(env, asset_cfg=asset_cfg, max_velocity=20.0)

    torch.testing.assert_close(penalty, torch.tensor((5.0, 800.0, 800.0)))
    assert torch.isfinite(penalty).all()


class FakeTerminationManager:
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


class FakeHeldDeliveryTracker:
    """Small in-memory implementation of the task's held-delivery interface."""

    def _init_held_delivery_tracker(self) -> None:
        self.common_step_counter = 0
        self._target_entry_seen = torch.zeros((self.num_envs, self.num_particles), dtype=torch.bool)
        self._held_delivered = torch.zeros_like(self._target_entry_seen)
        self._held_delivery_tracker_step = -1

    def update_held_delivery_tracker(self, held_pour: torch.Tensor) -> None:
        if held_pour.shape != (self.num_envs,):
            raise ValueError
        if self._held_delivery_tracker_step == self.common_step_counter:
            return
        in_target = self.particles_in_target_mask()
        first_entry = in_target & ~self._target_entry_seen
        self._held_delivered |= first_entry & held_pour.unsqueeze(-1)
        self._target_entry_seen |= in_target
        self._held_delivery_tracker_step = self.common_step_counter

    def held_delivered_mask(self) -> torch.Tensor:
        return self._held_delivered

    def current_held_delivered_mask(self) -> torch.Tensor:
        return self._held_delivered & self.particles_in_target_mask()


class FakeEnv(FakeHeldDeliveryTracker):
    """Minimal vectorized stand-in exposing the interface consumed by pure reward terms."""

    def __init__(self):
        self.num_envs = 4
        self.num_particles = 1000
        self.device = "cpu"
        self.step_dt = 1.0 / 60.0
        self.pour_target_frac = torch.full((self.num_envs,), 0.9)
        self.curriculum_stage = torch.full((self.num_envs,), 2, dtype=torch.long)
        self.cfg = type(
            "Cfg",
            (),
            {
                "curriculum_stage_names": ("pour", "carry", "lift", "full"),
                "max_spill_fraction": 0.10,
            },
        )()
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool)
        self.ep_max_target_frac = torch.zeros(self.num_envs)
        self._success_dwell_count = torch.zeros(self.num_envs, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, dtype=torch.long)
        self._lifted_grasp_seen = torch.zeros(self.num_envs, dtype=torch.bool)
        self.termination_manager = FakeTerminationManager(self.num_envs)
        self.gripper_open_width = 0.08
        self.gripper_grasp_width = 0.060
        self.cup_reset_height = 0.0
        self.action_manager = FakeActionManager(self.num_envs)
        self._gripper_command = self.action_manager.get_term("gripper_action").commanded_position[:, 0]

        self._tcp = torch.tensor([[0.50, 0.00, 0.032], [0.50, 0.00, 0.032], [0.50, 0.00, 0.032], [0.20, 0.00, 0.032]])
        self._tcp_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self._grasp = torch.tensor([[0.50, 0.00, 0.032]]).repeat(self.num_envs, 1)
        self._cup = torch.tensor(
            [
                [0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
                [0.50, 0.00, 0.12, 0.0, 0.0, 0.0, 1.0],
                [0.50, -0.17, 0.12, 0.7071068, 0.0, 0.0, 0.7071068],
                [0.20, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._target = torch.tensor([[0.50, -0.18, 0.00, 0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self._width = torch.tensor([0.08, 0.060, 0.060, 0.060])
        self._src = torch.tensor([1000.0, 750.0, 400.0, 0.0])
        self._tgt = torch.tensor([0.0, 250.0, 500.0, 950.0])
        self._spill = torch.tensor([0.0, 0.0, 100.0, 50.0])
        self._arm_q = torch.zeros((self.num_envs, 7))
        self._init_held_delivery_tracker()

    def tcp_pos_e(self):
        return self._tcp

    def tcp_pose_e(self):
        return torch.cat((self._tcp, self._tcp_quat), dim=-1)

    def desired_grasp_tcp_quat_c(self):
        return torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)

    def cup_grasp_point_e(self):
        return self._grasp

    def cup_pose_e(self):
        return self._cup

    def target_pose_e(self):
        return self._target

    def gripper_width(self):
        return self._width

    def arm_joint_pos(self):
        return self._arm_q

    def count_in_target(self):
        return self._tgt

    def particles_in_target_mask(self):
        particle_ids = torch.arange(self.num_particles).unsqueeze(0)
        return particle_ids < self._tgt.to(dtype=torch.long).unsqueeze(-1)

    def count_in_source(self):
        return self._src

    def count_spilled(self):
        return self._spill

    def spilled_fraction(self):
        return self._spill / self.num_particles


def test_legacy_reward_api_remains_available_with_deprecation_warning():
    legacy_names = (
        "reach_cup",
        "grasp_cup",
        "lift_cup",
        "lift_command_progress",
        "align_cup_over_target",
        "align_command_progress",
        "tilt_over_target",
        "tilt_command_progress",
    )
    assert all(hasattr(mdp_api, name) for name in legacy_names)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mdp_api.reach_cup(FakeEnv())

    assert result.shape == (4,)
    assert any(item.category is DeprecationWarning for item in caught)


class FakeDeliveryEnv(FakeHeldDeliveryTracker):
    """Four-particle environment used to exercise entry and reward idempotence."""

    def __init__(self, num_envs: int = 2, num_particles: int = 4):
        self.num_envs = num_envs
        self.num_particles = num_particles
        self.device = "cpu"
        self.step_dt = 0.02
        self.pour_target_frac = torch.ones(num_envs)
        self.termination_manager = FakeTerminationManager(num_envs)
        self._target_mask = torch.zeros((num_envs, num_particles), dtype=torch.bool)
        self._init_held_delivery_tracker()

    def particles_in_target_mask(self) -> torch.Tensor:
        return self._target_mask


def _set_simple_env_held_state(env, held: torch.Tensor | None = None) -> None:
    """Attach the source-grasp interface required by held-delivery shaping."""
    if held is None:
        held = torch.ones(env.num_envs, dtype=torch.bool)
    cup_pose = torch.zeros((env.num_envs, 7))
    cup_pose[:, 2] = 0.06
    tcp = torch.zeros((env.num_envs, 3))
    command = torch.where(held, 0.0, 0.04).unsqueeze(-1)
    env.cup_reset_height = 0.0
    env.gripper_grasp_width = 0.06
    env.cup_pose_e = lambda: cup_pose
    env.tcp_pos_e = lambda: tcp
    env.cup_grasp_point_e = lambda: tcp
    env.gripper_width = lambda: torch.full((env.num_envs,), 0.06)
    env.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(commanded_position=command),
    )


def test_particle_fractions_spill_and_success():
    env = FakeEnv()
    assert torch.allclose(rewards.particles_in_target(env), torch.tensor([0.0, 0.25, 0.5, 0.95]))
    assert torch.allclose(rewards.particles_in_source(env), torch.tensor([1.0, 0.75, 0.4, 0.0]))
    assert torch.allclose(rewards.spilled_particles(env), torch.tensor([0.0, 0.0, 0.1, 0.05]), atol=1e-6)
    fractions = observations.particle_fractions_obs(env)
    torch.testing.assert_close(
        fractions,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.75, 0.25, 0.0, 0.0],
                [0.4, 0.5, 0.1, 0.0],
                [0.0, 0.95, 0.05, 0.0],
            ]
        ),
    )


def test_stable_success_requires_consecutive_valid_steps_and_rejects_failures():
    env = FakeEnv()
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    env._tgt[:] = torch.tensor([950.0, 950.0, 0.0, 950.0])
    first = terminations.stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)
    env.common_step_counter += 1
    env._tgt[:] = torch.tensor([950.0, 0.0, 950.0, 950.0])
    env.termination_manager.terminated[3] = True
    second = terminations.stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)

    assert first.tolist() == [False, False, False, False]
    assert second.tolist() == [True, False, False, False]
    assert env.episode_succeeded.tolist() == [True, False, False, False]
    torch.testing.assert_close(env.ep_max_target_frac, torch.tensor([0.95, 0.95, 0.95, 0.95]))


def test_immediate_pour_success_uses_only_current_target_fraction_and_failure_precedence():
    env = FakeEnv()
    env.pour_target_frac[:] = 0.30
    env._tgt[:] = torch.tensor([299.0, 300.0, 950.0, 950.0])
    env.termination_manager.terminated[3] = True

    success = terminations.immediate_pour_success(env)

    assert success.tolist() == [False, True, True, False]
    assert env.episode_succeeded.tolist() == [False, True, True, False]
    assert env._success_dwell_count.tolist() == [0, 1, 1, 0]
    torch.testing.assert_close(env.ep_max_target_frac, torch.tensor([0.299, 0.300, 0.950, 0.950]))


def test_stable_success_counter_resets_selected_worlds_only():
    env = FakeEnv()
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    env._tgt[:] = 950.0
    assert not bool(torch.any(terminations.stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)))
    env._success_dwell_count[0] = 0
    env.common_step_counter += 1
    success = terminations.stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)

    assert success.tolist() == [False, True, True, True]


def test_stable_success_is_directly_reusable_as_a_state_reward():
    env = FakeEnv()
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    env._tgt[:] = 950.0

    first = terminations.stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)
    env.common_step_counter += 1
    second = terminations.stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)

    assert not bool(torch.any(first))
    assert bool(torch.all(second))
    assert env._success_dwell_count.tolist() == [2, 2, 2, 2]
    assert env.episode_succeeded.tolist() == [True, True, True, True]


def test_nonterminating_success_context_tracks_state_and_remains_replay_discoverable():
    env = FakeEnv()
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    env._tgt[:] = 950.0

    first = terminations.nonterminating_stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)
    env.common_step_counter += 1
    second = terminations.nonterminating_stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)

    assert not bool(torch.any(first))
    assert not bool(torch.any(second))
    assert env.episode_succeeded.tolist() == [True, True, True, True]
    assert rewards.sustained_pour_success(env, dwell_time_s=2.0 * env.step_dt).tolist() == [1.0, 1.0, 1.0, 1.0]

    # Record/replay tools remove the managed term before invoking its configured callable.
    env.termination_manager._terms.pop("success")
    replay_success = terminations.nonterminating_stable_pour_success(env, dwell_time_s=2.0 * env.step_dt)
    assert bool(torch.all(replay_success))


def test_sustained_success_reward_is_current_unit_state_not_terminal_pulse():
    env = FakeEnv()
    env._success_dwell_count[:] = torch.tensor([0, 1, 2, 3])

    success = rewards.sustained_pour_success(env, dwell_time_s=2.0 * env.step_dt)

    assert success.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert float(success.max()) == 1.0


def test_unsuccessful_timeout_excludes_same_step_success():
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([10, 10, 9, 10]),
        max_episode_length=10,
        episode_succeeded=torch.tensor([True, False, False, True]),
    )

    timed_out = terminations.unsuccessful_time_out(env)

    assert timed_out.tolist() == [False, True, False, False]


def test_stable_success_requires_a_preloaded_held_and_lifted_source():
    env = FakeEnv()
    env._tgt[:] = 950.0
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0

    env._cup[0, 2] = 0.0
    env._tcp[1] = env._grasp[1] + torch.tensor([0.04, 0.0, 0.0])
    env._gripper_command[2] = 0.04

    success = terminations.stable_pour_success(env, dwell_time_s=env.step_dt)

    assert success.tolist() == [False, False, False, True]


def test_stable_success_rejects_a_geometrically_plausible_unilateral_grasp():
    env = FakeEnv()
    env._tgt[:] = 950.0
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.025
    env.action_manager.get_term("gripper_action").bilateral_contact[0] = False

    success = terminations.stable_pour_success(env, dwell_time_s=env.step_dt)

    assert success.tolist() == [False, True, True, True]


def test_gripper_contact_observation_preserves_per_finger_asymmetry():
    env = FakeEnv()
    env.action_manager.get_term("gripper_action").contact_deflection[:] = torch.tensor(
        [[0.002, 0.002], [0.003, 0.0002], [0.0, 0.0], [0.001, 0.004]]
    )

    contact = observations.gripper_contact_obs(env)

    torch.testing.assert_close(
        contact,
        torch.tensor([[0.002, 0.002], [0.003, 0.0002], [0.0, 0.0], [0.001, 0.004]]),
    )


def test_policy_grasp_geometry_is_cup_relative_and_quaternion_sign_invariant():
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
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        observations.grasp_to_tcp_quat_obs(env),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        observations.target_position_c_obs(env),
        torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]]),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        observations.tcp_pose_obs(env)[:, 3:7],
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt]]),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_individual_finger_and_held_delivery_observations_preserve_state():
    finger_position = torch.tensor([[0.01, 0.02], [0.03, 0.04]])
    finger_velocity = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    held = torch.tensor([[True, False, True, False], [False, False, True, False]])
    env = SimpleNamespace(
        num_particles=4,
        finger_joint_pos=lambda: finger_position,
        finger_joint_vel=lambda: finger_velocity,
        held_delivered_mask=lambda: held,
    )

    torch.testing.assert_close(observations.finger_position_obs(env), finger_position)
    torch.testing.assert_close(observations.finger_velocity_obs(env), finger_velocity)
    torch.testing.assert_close(observations.held_delivery_history_obs(env), torch.tensor([[0.5], [0.25]]))


def test_grasp_command_comparison_tolerates_float32_filter_roundoff():
    env = FakeEnv()
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.025 + 5.0e-7

    _, preloaded, lifted = terminations.source_grasp_milestones(
        env,
        min_lift_height=0.05,
        max_tcp_distance=0.018,
        max_gripper_width_error=0.006,
        max_gripper_command=0.025,
    )

    assert bool(torch.all(preloaded))
    assert bool(torch.all(lifted))
    env._gripper_command[0] = 0.025 + 2.0e-6
    _, preloaded, _ = terminations.source_grasp_milestones(
        env,
        min_lift_height=0.05,
        max_tcp_distance=0.018,
        max_gripper_width_error=0.006,
        max_gripper_command=0.025,
    )
    assert not bool(preloaded[0])


def test_stable_success_cannot_recover_particles_first_delivered_while_unheld():
    env = FakeEnv()
    env.pour_target_frac[:] = 0.4
    env._tgt[:] = 950.0
    env._cup[:, 2] = 0.0
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0

    assert not bool(torch.any(terminations.stable_pour_success(env, dwell_time_s=env.step_dt)))
    assert not bool(torch.any(env.held_delivered_mask()))

    # Re-grasping and lifting the now-empty source cannot retroactively validate the first entry.
    env.common_step_counter += 1
    env._cup[:, 2] = 0.06
    assert not bool(torch.any(terminations.stable_pour_success(env, dwell_time_s=env.step_dt)))
    assert not bool(torch.any(env.current_held_delivered_mask()))


def test_success_reward_mirrors_cached_success_terminal():
    env = FakeEnv()
    env._tgt[:] = 0.0
    env.termination_manager.terminated[:] = torch.tensor([True, True, False, False])
    env.termination_manager._terms["success"][:] = torch.tensor([True, False, False, False])

    success = rewards.pour_success_bonus(env) * env.step_dt

    assert success.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_success_reward_tolerates_recording_tools_removing_success_termination():
    env = FakeEnv()
    env.termination_manager._terms.pop("success")

    success = rewards.pour_success_bonus(env) * env.step_dt

    assert success.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_terminal_failure_tolerates_recording_tools_removing_success_termination():
    env = FakeEnv()
    env.termination_manager._terms.pop("success")
    env.termination_manager.terminated[:] = torch.tensor([True, False, True, False])

    failure = rewards.terminal_failure(env) * env.step_dt

    assert failure.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_terminal_failure_can_exclude_ordinary_fixed_horizon_timeout():
    env = FakeEnv()
    env.termination_manager.terminated[:] = torch.tensor([True, False, True, False])
    env.termination_manager.time_outs[:] = torch.tensor([False, True, True, False])
    env.termination_manager._terms["success"][:] = torch.tensor([False, False, True, False])

    failure = rewards.terminal_failure(env, include_time_out=False) * env.step_dt

    assert failure.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_held_delivery_tracker_records_only_first_entries_and_is_step_idempotent():
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv

    env = FakeDeliveryEnv()
    env._target_mask[:] = torch.tensor([[True, False, False, False], [True, False, False, False]])

    FrankaPourEnv.update_held_delivery_tracker(env, torch.tensor([True, False]))
    assert FrankaPourEnv.held_delivered_mask(env).tolist() == [
        [True, False, False, False],
        [False, False, False, False],
    ]

    # A second consumer in the same manager step cannot reinterpret a changed view as a new entry.
    env._target_mask[:, 1] = True
    FrankaPourEnv.update_held_delivery_tracker(env, torch.ones(2, dtype=torch.bool))
    assert FrankaPourEnv.held_delivered_mask(env).tolist() == [
        [True, False, False, False],
        [False, False, False, False],
    ]

    env.common_step_counter += 1
    FrankaPourEnv.update_held_delivery_tracker(env, torch.ones(2, dtype=torch.bool))
    assert FrankaPourEnv.held_delivered_mask(env).tolist() == [
        [True, True, False, False],
        [False, True, False, False],
    ]

    # An earlier unheld entry can qualify after the particle leaves and validly re-enters.
    env._target_mask[1, 0] = False
    env.common_step_counter += 1
    FrankaPourEnv.update_held_delivery_tracker(env, torch.ones(2, dtype=torch.bool))
    env._target_mask[1, 0] = True
    env.common_step_counter += 1
    FrankaPourEnv.update_held_delivery_tracker(env, torch.ones(2, dtype=torch.bool))
    assert bool(FrankaPourEnv.held_delivered_mask(env)[1, 0])

    env._target_mask[0, 0] = False
    assert bool(FrankaPourEnv.held_delivered_mask(env)[0, 0])
    assert not bool(FrankaPourEnv.current_held_delivered_mask(env)[0, 0])


def test_held_delivery_progress_is_signed_capped_and_resets_selectively():
    env = FakeDeliveryEnv()
    _set_simple_env_held_state(env)
    env.pour_target_frac[:] = torch.tensor([0.5, 0.25])
    env._target_mask[:] = torch.tensor([[True, True, True, False], [True, True, False, False]])
    term = rewards.HeldDeliveryProgress(SimpleNamespace(), env)

    # Credit stops at each environment's active success threshold.
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(2))

    env.common_step_counter += 1
    env._target_mask[:] = torch.tensor([[False, True, False, False], [True, True, True, True]])
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([-0.25, 0.0]))

    env.common_step_counter += 1
    env._target_mask[:] = torch.tensor([[True, True, True, False], [False, False, False, False]])
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.25, -0.25]))

    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.5, 0.0]))


def test_held_delivery_progress_claws_back_timeout_credit_and_preserves_success_credit():
    env = FakeDeliveryEnv()
    _set_simple_env_held_state(env)
    env._target_mask[:] = torch.tensor([[True, True, False, False], [True, True, False, False]])
    term = rewards.HeldDeliveryProgress(SimpleNamespace(), env)

    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.5, 0.5]))

    env.common_step_counter += 1
    env.termination_manager.time_outs[0] = True
    env.termination_manager.terminated[1] = True
    env.termination_manager._terms["success"][1] = True

    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([-0.5, 0.0]))
    torch.testing.assert_close(term._previous_credit, torch.tensor([0.0, 0.5]))


def test_held_delivery_progress_does_not_pay_new_credit_on_failed_terminal_step():
    env = FakeDeliveryEnv()
    _set_simple_env_held_state(env)
    env._target_mask[:] = torch.tensor([[True, True, False, False], [True, True, False, False]])
    env.termination_manager.terminated[0] = True
    env.termination_manager._terms["success"][1] = True
    env.termination_manager.terminated[1] = True
    term = rewards.HeldDeliveryProgress(SimpleNamespace(), env)

    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.0, 0.5]))


def test_new_delivery_never_rewards_unheld_first_entry_after_regrasp():
    env = FakeDeliveryEnv()
    _set_simple_env_held_state(env, torch.tensor([True, False]))
    env._target_mask[:] = torch.tensor([[True, True, False, False]] * 2)
    term = rewards.NewlyDeliveredParticles(SimpleNamespace(), env)

    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.5, 0.0]))
    # First entry is consumed even when invalid, so grabbing the source afterward cannot recover it.
    env.action_manager.get_term("gripper_action").commanded_position[1] = 0.0
    env.common_step_counter += 1
    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(2))


def test_new_spill_penalizes_each_particle_at_most_once_and_resets_selectively():
    env = SimpleNamespace(
        num_envs=2,
        num_particles=4,
        device="cpu",
        step_dt=0.02,
    )
    spill_mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    env.particles_spilled_mask = lambda: spill_mask
    term = rewards.NewlySpilledParticles(SimpleNamespace(), env)

    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(2))

    spill_mask = torch.tensor([[False, True, False, False], [False, True, False, False]])
    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(2))

    spill_mask = torch.tensor([[True, True, False, False], [False, False, True, False]])
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.0, 0.25]))

    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0.5, 0.0]))


def test_spill_mask_requires_table_contact_outside_both_cups():
    points = torch.zeros((2, 4, 3))
    points[:, :, 2] = torch.tensor([0.003, 0.0031, 0.0, -0.01])
    in_source = torch.tensor([[False, False, True, False], [False, False, False, False]])
    in_target = torch.tensor([[False, False, False, True], [False, False, False, False]])

    spilled = terminations._spilled_particle_mask(points, in_source, in_target, max_height=0.003)

    assert spilled.tolist() == [[True, False, False, False], [True, False, True, True]]


def test_delivered_particle_mask_excludes_particles_still_inside_source_cup():
    in_source = torch.tensor([[True, True, False, False], [True, False, False, True]])
    in_target = torch.tensor([[True, False, True, False], [True, True, False, True]])

    delivered = terminations._delivered_particle_mask(in_source, in_target)

    assert delivered.tolist() == [[False, False, True, False], [False, True, False, False]]
    assert not bool(torch.any(delivered & in_source))


def test_particle_region_masks_use_exclusive_target_membership():
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv

    in_source = torch.tensor([[True, True, False, False]])
    in_target_region = torch.tensor([[True, False, True, False]])
    region_results = iter((in_source, in_target_region))
    points = torch.zeros((1, 4, 3))
    env = SimpleNamespace(
        _particle_region_cache=None,
        _particle_region_cache_step=-1,
        common_step_counter=4,
        particle_pos_e=lambda: points,
        cup_pose_e=lambda: torch.empty((1, 7)),
        target_pose_e=lambda: torch.empty((1, 7)),
        _source_inner_lo_t=torch.empty(3),
        _source_inner_hi_t=torch.empty(3),
        _target_inner_lo_t=torch.empty(3),
        _target_inner_hi_t=torch.empty(3),
        _points_inside_cup=lambda *_: next(region_results),
        cfg=SimpleNamespace(spill_table_height=0.0, particle_count_margin=0.003),
    )

    source, target, spilled = FrankaPourEnv._particle_region_masks(env)

    assert source.tolist() == [[True, True, False, False]]
    assert target.tolist() == [[False, False, True, False]]
    assert spilled.tolist() == [[False, False, False, True]]


def test_excessive_spill_is_strictly_greater_than_ten_percent():
    env = FakeEnv()
    env.num_particles = 10
    env._spill = torch.tensor([0.0, 1.0, 2.0, 1.0])

    excessive = terminations.excessive_spill(env)

    assert excessive.tolist() == [False, False, True, False]


def test_reset_dataset_spill_threshold_triggers_on_particle_74_of_245():
    env = FakeEnv()
    env.num_particles = 245
    env.cfg.max_spill_fraction = 0.30
    env._spill = torch.tensor([0.0, 73.0, 74.0, 245.0])

    excessive = terminations.excessive_spill(env)

    assert excessive.tolist() == [False, False, True, True]


def test_excessive_spill_can_monitor_without_terminating():
    env = FakeEnv()
    env._spill = torch.tensor([0.0, 100.0, 101.0, 1000.0])

    reported = terminations.excessive_spill(env, terminate=False)

    assert not bool(torch.any(reported))


def test_stable_success_rejects_excessive_spill_without_spill_termination():
    env = FakeEnv()
    env.pour_target_frac[:] = 0.9
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    env._tgt[:] = 950.0
    env._spill[:] = torch.tensor([0.0, 100.0, 101.0, 1000.0])

    success = terminations.stable_pour_success(env, dwell_time_s=env.step_dt)

    assert success.tolist() == [True, True, False, False]


def test_lost_grasp_requires_consecutive_loss_after_a_demonstrated_lift():
    env = FakeEnv()
    env.cfg.success_min_lift_height = 0.05
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    env._lifted_grasp_seen[:] = torch.tensor([True, True, False, False])
    env._tcp[1, 0] += 0.04

    first = terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)
    second = terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)
    env._tcp[1] = env._grasp[1]
    recovered = terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)
    env._tcp[1, 0] += 0.04
    terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)
    terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)
    third = terminations.lost_lifted_grasp(env, dwell_time_s=3.0 * env.step_dt)

    assert first.tolist() == [False, False, False, False]
    assert second.tolist() == [False, False, False, False]
    assert recovered.tolist() == [False, False, False, False]
    assert third.tolist() == [False, True, False, False]


def test_lost_grasp_can_monitor_without_terminating():
    env = FakeEnv()
    env.cfg.success_min_lift_height = 0.05
    env._cup[:, 2] = 0.06
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    terminations.lost_lifted_grasp(env, dwell_time_s=env.step_dt, terminate=False)

    env._tcp[:, 0] += 0.1
    reported = terminations.lost_lifted_grasp(env, dwell_time_s=env.step_dt, terminate=False)

    assert not bool(torch.any(reported))
    assert env._lifted_grasp_seen.tolist() == [True, True, True, True]
    assert env._lost_grasp_dwell_count.tolist() == [1, 1, 1, 1]


def test_cached_grasp_drop_terminates_and_produces_failure_pulse():
    env = FakeEnv()
    env.cfg.success_min_lift_height = 0.05
    env._cup[:, 2] = 0.06
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    # Model a grasping cache row followed by non-grasping rows. All cups are geometrically lost,
    # but only the demonstrated grasp is eligible for dropped-cup termination.
    env._lifted_grasp_seen[:] = torch.tensor([True, False, False, False])
    env._tcp[:] = env._grasp + torch.tensor((0.1, 0.0, 0.0))

    dropped = terminations.lost_lifted_grasp(env, dwell_time_s=env.step_dt)
    env.termination_manager.terminated[:] = dropped
    failure_pulse = rewards.terminal_failure(env, include_time_out=False) * env.step_dt

    assert dropped.tolist() == [True, False, False, False]
    torch.testing.assert_close(failure_pulse, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_trajectory_phase_and_applied_reference_error_are_observable():
    arm_q = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    reference_target = arm_q + 0.25
    terms = {
        "arm_action": SimpleNamespace(
            reference_phase=torch.tensor([0.2, 0.7]),
            reference_target=reference_target,
            processed_actions=reference_target + 0.5,
            reference_error=reference_target + 0.5 - arm_q,
        )
    }
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        action_manager=SimpleNamespace(get_term=lambda name: terms[name]),
        arm_joint_pos=lambda: arm_q,
    )

    torch.testing.assert_close(observations.arm_reference_phase_obs(env), torch.tensor([[0.2], [0.7]]))
    torch.testing.assert_close(observations.arm_reference_error_obs(env), torch.full((2, 7), 0.75))


def test_trajectory_status_exposes_every_stateful_gate_and_dwell():
    arm_status = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.5, 0.25], [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
    capture_status = torch.tensor([[0.0, 0.4], [1.0, 0.0]])
    terms = {
        "arm_action": SimpleNamespace(milestone_status=arm_status),
        "gripper_action": SimpleNamespace(capture_status=capture_status),
    }
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        action_manager=SimpleNamespace(get_term=lambda name: terms[name]),
    )

    torch.testing.assert_close(
        observations.trajectory_status_obs(env),
        torch.cat((arm_status, capture_status), dim=-1),
    )


def test_time_and_failure_dwell_observations_make_finite_horizon_state_observable():
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([0, 5, 10]),
        max_episode_length=10,
        step_dt=0.02,
        _lost_grasp_dwell_count=torch.tensor([0, 1, 3]),
        pour_target_frac=torch.tensor([0.1, 0.2, 0.35]),
        cfg=SimpleNamespace(lost_grasp_dwell_time_s=0.05),
    )

    torch.testing.assert_close(
        observations.time_remaining_obs(env),
        torch.tensor([[1.0], [0.5], [0.0]]),
    )
    torch.testing.assert_close(
        observations.lost_grasp_dwell_obs(env),
        torch.tensor([[0.0], [1.0 / 3.0], [1.0]]),
    )
    torch.testing.assert_close(
        observations.pour_target_fraction_obs(env),
        torch.tensor([[0.1], [0.2], [0.35]]),
    )


def test_particle_transfer_observation_reports_airborne_flow_and_handles_empty_stream():
    source = torch.tensor([[True, True, False, False], [True, True, True, True]])
    target = torch.tensor([[False, False, False, True], [False, False, False, False]])
    spilled = torch.zeros_like(source)
    positions = torch.zeros((2, 4, 3))
    positions[0, 2] = torch.tensor([0.8, -0.18, 0.0])
    velocities = torch.zeros_like(positions)
    velocities[0, 2] = torch.tensor([2.0, 0.0, 0.0])
    target_pose = torch.tensor([[0.5, -0.18, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env = SimpleNamespace(
        num_particles=4,
        particle_region_masks=lambda: (source, target, spilled),
        particle_pos_e=lambda: positions,
        particle_vel_e=lambda: velocities,
        target_pose_e=lambda: target_pose,
    )

    summary = observations.particle_transfer_obs(env)

    torch.testing.assert_close(summary[0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]))
    torch.testing.assert_close(summary[1], torch.zeros(7))


def test_task_progress_is_signed_hold_neutral_and_cycle_neutral():
    env = FakeEnv()
    env.curriculum_stage.zero_()
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp + torch.tensor([0.0, 0.0, 0.05])
    env._width[:] = env.gripper_open_width
    params = {
        "target_height": 0.12,
        "reach_std": 0.07,
        "grasp_reach_std": 0.015,
        "grasp_preload_position": 0.025,
        "lift_height": 0.06,
        "align_std": 0.12,
        "source_offset_xy": (0.0, 0.05),
        "target_tilt": math.radians(150.0),
        "pour_direction_xy": (0.0, -1.0),
        "source_mouth_height": 0.036,
        "alignment_radius": 0.15,
        "active_through_stage": 1,
        "min_lift_height": 0.05,
        "max_tcp_distance": 0.015,
        "max_gripper_width_error": 0.012,
        "max_gripper_command": 0.025,
        "discount_factor": 0.99,
    }
    term = rewards.PourTaskProgress(RewardTermCfg(func=rewards.PourTaskProgress, weight=5.0, params=params), env)
    term.reset()

    env._tcp[0] = env._grasp[0] + torch.tensor([0.0, 0.0, 0.02])
    first = term(env, **params) * env.step_dt
    env._tcp[0] = env._grasp[0] + torch.tensor([0.0, 0.0, 0.05])
    second = term(env, **params) * env.step_dt

    assert first[0] > first[1]
    cycle_return = first[0] + params["discount_factor"] * second[0]
    hold_return = first[1] + params["discount_factor"] * second[1]
    torch.testing.assert_close(cycle_return, hold_return, atol=1.0e-6, rtol=0.0)


def test_task_progress_closes_the_potential_on_timeout():
    params = {
        "target_height": 0.12,
        "reach_std": 0.07,
        "grasp_reach_std": 0.015,
        "grasp_preload_position": 0.025,
        "lift_height": 0.06,
        "align_std": 0.12,
        "source_offset_xy": (0.0, 0.05),
        "target_tilt": math.radians(150.0),
        "pour_direction_xy": (0.0, -1.0),
        "source_mouth_height": 0.036,
        "alignment_radius": 0.15,
        "active_through_stage": 1,
        "min_lift_height": 0.05,
        "max_tcp_distance": 0.015,
        "max_gripper_width_error": 0.012,
        "max_gripper_command": 0.025,
        "discount_factor": 0.99,
    }
    timeout_env = FakeEnv()
    terminated_env = FakeEnv()
    timeout_term = rewards.PourTaskProgress(
        RewardTermCfg(func=rewards.PourTaskProgress, weight=5.0, params=params), timeout_env
    )
    terminated_term = rewards.PourTaskProgress(
        RewardTermCfg(func=rewards.PourTaskProgress, weight=5.0, params=params), terminated_env
    )
    timeout_term.reset()
    terminated_term.reset()
    timeout_env.termination_manager.time_outs[:] = True
    terminated_env.termination_manager.terminated[:] = True

    timeout_progress = timeout_term(timeout_env, **params)
    terminated_progress = terminated_term(terminated_env, **params)

    torch.testing.assert_close(timeout_progress, terminated_progress)


def test_approach_progress_preserves_reach_gradient_after_premature_closure():
    env = FakeEnv()
    env.curriculum_stage[:] = 3
    env._cup[:] = torch.tensor([0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0])
    env._grasp[:] = torch.tensor([0.50, 0.00, 0.032])
    env._tcp[:] = env._grasp + torch.tensor([0.20, 0.0, 0.0])
    env._width[:] = env.gripper_open_width
    env._gripper_command[:] = 0.04
    params = {
        "position_std": 0.20,
        "orientation_std": 0.75,
        "open_hand_fraction": 0.35,
        "active_from_stage": 3,
        "discount_factor": 0.99,
    }
    term = rewards.ApproachProgress(
        RewardTermCfg(func=rewards.ApproachProgress, weight=8.0, params=params),
        env,
    )
    term.reset()

    # Closing at stand-off must lose the coordination bonus, but must not erase the independent
    # Cartesian reach gradient that lets the policy recover from this mistake.
    env._width[0] = env.gripper_grasp_width
    env._gripper_command[0] = 0.02
    premature_close = term(env, **params) * env.step_dt
    env._tcp[0] = env._grasp[0] + torch.tensor([0.10, 0.0, 0.0])
    recover_reach = term(env, **params) * env.step_dt

    assert premature_close[0] < 0.0
    assert recover_reach[0] > 0.0
    torch.testing.assert_close(premature_close[1:], recover_reach[1:])


def test_approach_progress_rewards_pose_alignment_without_a_reference_trajectory():
    env = FakeEnv()
    env.curriculum_stage[:] = 3
    env._cup[:] = torch.tensor([0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0])
    env._grasp[:] = torch.tensor([0.50, 0.00, 0.032])
    env._tcp[:] = env._grasp + torch.tensor([0.08, 0.0, 0.0])
    env._tcp_quat[0] = torch.tensor([0.0, 0.0, math.sin(0.5), math.cos(0.5)])
    params = {
        "position_std": 0.20,
        "orientation_std": 0.75,
        "open_hand_fraction": 0.35,
        "active_from_stage": 3,
        "discount_factor": 0.99,
    }
    term = rewards.ApproachProgress(
        RewardTermCfg(func=rewards.ApproachProgress, weight=8.0, params=params),
        env,
    )
    term.reset()

    env._tcp_quat[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    align = term(env, **params) * env.step_dt
    env._tcp[0] = env._grasp[0]
    approach = term(env, **params) * env.step_dt

    assert align[0] > 0.0
    assert approach[0] > 0.0


def test_grasp_lift_progress_requires_near_contact_then_rewards_lift():
    env = FakeEnv()
    env.curriculum_stage[:] = 2
    env._cup[:] = torch.tensor([0.50, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0])
    env._grasp[:] = torch.tensor([0.50, 0.00, 0.032])
    env._tcp[:] = env._grasp + torch.tensor([0.10, 0.0, 0.0])
    env._width[:] = env.gripper_open_width
    env._gripper_command[:] = 0.04
    params = {
        "target_height": 0.10,
        "grasp_reach_std": 0.025,
        "grasp_preload_position": 0.025,
        "grasp_fraction": 0.40,
        "active_from_stage": 2,
        "discount_factor": 0.99,
    }
    term = rewards.GraspLiftProgress(
        RewardTermCfg(func=rewards.GraspLiftProgress, weight=10.0, params=params),
        env,
    )
    term.reset()

    env._width[0] = env.gripper_grasp_width
    env._gripper_command[0] = 0.025
    far_close = term(env, **params) * env.step_dt
    env._tcp[0] = env._grasp[0]
    contact = term(env, **params) * env.step_dt
    env._cup[0, 2] = 0.10
    env._grasp[0, 2] += 0.10
    env._tcp[0, 2] += 0.10
    lift = term(env, **params) * env.step_dt

    assert far_close[0] <= 0.0
    assert contact[0] > 0.0
    assert lift[0] > 0.0


def test_lift_progress_is_signed_bounded_and_cycle_neutral():
    env = FakeEnv()
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.025
    env._tcp[:] = env._grasp
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    cfg = RewardTermCfg(
        func=rewards.LiftProgress,
        weight=5.0,
        params={"target_height": 0.12, "reach_std": 0.10},
    )
    term = rewards.LiftProgress(cfg, env)
    term.reset()

    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(env.num_envs))
    env._cup[0, 2] = 0.06
    forward = term(env) * env.step_dt
    env._cup[0, 2] = 0.0
    reverse = term(env) * env.step_dt

    assert 0.0 < forward[0] <= 1.0
    assert -1.0 <= reverse[0] < 0.0
    torch.testing.assert_close(forward + reverse, torch.zeros(env.num_envs))


def test_lift_progress_rewards_ordered_open_approach_grasp_and_lift():
    env = FakeEnv()
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp + torch.tensor([0.0, 0.0, 0.05])
    env._width[:] = env.gripper_open_width
    cfg = RewardTermCfg(
        func=rewards.LiftProgress,
        weight=10.0,
        params={
            "target_height": 0.12,
            "reach_std": 0.07,
            "grasp_reach_std": 0.015,
            "approach_fraction": 0.2,
            "grasp_fraction": 0.3,
        },
    )
    term = rewards.LiftProgress(cfg, env)
    term.reset()

    env._tcp[0] = env._grasp[0] + torch.tensor([0.0, 0.0, 0.025])
    approach = term(env) * env.step_dt
    env._tcp[0] = env._grasp[0]
    env._width[0] = env.gripper_grasp_width
    env._gripper_command[0] = 0.025
    grasp = term(env) * env.step_dt
    env._cup[0, 2] = 0.06
    lift = term(env) * env.step_dt

    env._cup[0, 2] = 0.0
    reverse_lift = term(env) * env.step_dt
    env._width[0] = env.gripper_open_width
    env._tcp[0] = env._grasp[0] + torch.tensor([0.0, 0.0, 0.025])
    reverse_grasp = term(env) * env.step_dt
    env._tcp[0] = env._grasp[0] + torch.tensor([0.0, 0.0, 0.05])
    reverse_approach = term(env) * env.step_dt

    assert approach[0] > 0.0
    assert grasp[0] > 0.0
    assert lift[0] > 0.0
    torch.testing.assert_close(
        approach + grasp + lift + reverse_lift + reverse_grasp + reverse_approach,
        torch.zeros(env.num_envs),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_lift_progress_penalizes_closing_empty_gripper_far_from_cup():
    env = FakeEnv()
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp + torch.tensor([0.0, 0.0, 0.08])
    env._width[:] = env.gripper_open_width
    cfg = RewardTermCfg(
        func=rewards.LiftProgress,
        weight=10.0,
        params={
            "target_height": 0.12,
            "reach_std": 0.07,
            "grasp_reach_std": 0.015,
            "approach_fraction": 0.2,
            "grasp_fraction": 0.3,
        },
    )
    term = rewards.LiftProgress(cfg, env)
    term.reset()

    # An empty hand can exactly match the nominal cup width without touching the cup.
    env._width[0] = env.gripper_grasp_width
    env._gripper_command[0] = 0.025
    premature_close = term(env) * env.step_dt

    assert premature_close[0] < 0.0


def test_lift_progress_monotonically_rewards_near_contact_closure_and_rejects_overclose():
    env = FakeEnv()
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_open_width
    cfg = RewardTermCfg(
        func=rewards.LiftProgress,
        weight=10.0,
        params={
            "target_height": 0.12,
            "reach_std": 0.07,
            "grasp_reach_std": 0.015,
            "approach_fraction": 0.2,
            "grasp_fraction": 0.3,
        },
    )
    term = rewards.LiftProgress(cfg, env)
    term.reset()

    for width, command in ((0.075, 0.035), (0.070, 0.030), (0.065, 0.025), (env.gripper_grasp_width, 0.020)):
        env._width[0] = width
        env._gripper_command[0] = command
        assert (term(env) * env.step_dt)[0] > 0.0

    env._width[0] = 0.0
    assert (term(env) * env.step_dt)[0] < 0.0


def test_lift_progress_does_not_treat_contact_compressed_open_fingers_as_a_grasp():
    env = FakeEnv()
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.04
    cfg = RewardTermCfg(
        func=rewards.LiftProgress,
        weight=10.0,
        params={
            "target_height": 0.12,
            "reach_std": 0.07,
            "grasp_reach_std": 0.015,
            "grasp_preload_position": 0.025,
            "approach_fraction": 0.2,
            "grasp_fraction": 0.3,
        },
    )
    term = rewards.LiftProgress(cfg, env)
    term.reset()

    env._gripper_command[0] = 0.025
    preload = term(env) * env.step_dt

    assert preload[0] > 0.0


def test_alignment_progress_is_signed_bounded_and_release_repays_progress():
    env = FakeEnv()
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.025
    cfg = RewardTermCfg(
        func=rewards.AlignProgress,
        weight=5.0,
        params={
            "lift_height": 0.06,
            "std": 0.12,
            "grasp_reach_std": 0.015,
            "grasp_preload_position": 0.025,
        },
    )
    term = rewards.AlignProgress(cfg, env)
    term.reset()

    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(env.num_envs))
    env._cup[0, :3] = torch.tensor([0.50, -0.18, 0.12])
    env._grasp[0, :2] = torch.tensor([0.50, -0.18])
    env._tcp[0] = env._grasp[0]
    forward = term(env) * env.step_dt
    env._gripper_command[0] = 0.04
    release = term(env) * env.step_dt

    assert 0.0 < forward[0] <= 1.0
    assert -1.0 <= release[0] < 0.0
    torch.testing.assert_close(forward + release, torch.zeros(env.num_envs))


def test_prerequisite_progress_is_gated_by_curriculum_stage():
    env = FakeEnv()
    env.curriculum_stage[:] = torch.arange(env.num_envs)
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.025
    lift = rewards.LiftProgress(RewardTermCfg(func=rewards.LiftProgress, weight=5.0), env)
    align = rewards.AlignProgress(RewardTermCfg(func=rewards.AlignProgress, weight=5.0), env)
    lift.reset()
    align.reset()

    env._cup[:, :3] = torch.tensor([0.50, -0.18, 0.12])
    env._grasp[:, :2] = torch.tensor([0.50, -0.18])
    env._tcp[:] = env._grasp
    lift_progress = lift(env) * env.step_dt
    align_progress = align(env) * env.step_dt

    torch.testing.assert_close(lift_progress[:2], torch.zeros(2))
    assert bool(torch.all(lift_progress[2:] > 0.0))
    assert align_progress[0] == 0.0
    assert bool(torch.all(align_progress[1:] > 0.0))


def _prepare_tilt_progress_env() -> FakeEnv:
    env = FakeEnv()
    env.curriculum_stage[:] = torch.arange(env.num_envs)
    env._cup[:, :3] = torch.tensor([0.50, -0.18, 0.06])
    env._cup[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    env._target[:, :3] = torch.tensor([0.50, -0.18, 0.00])
    env._tcp[:] = env._grasp
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.0
    return env


def _tilt_progress_term(env: FakeEnv) -> rewards.PourTiltProgress:
    cfg = RewardTermCfg(
        func=rewards.PourTiltProgress,
        weight=5.0,
        params={
            "target_tilt": math.radians(150.0),
            "pour_direction_xy": (0.0, -1.0),
            "source_mouth_height": 0.0,
            "alignment_radius": 0.10,
            "active_through_stage": 1,
        },
    )
    return rewards.PourTiltProgress(cfg, env)


def test_tilt_progress_is_directional_stage_gated_and_cycle_neutral():
    env = _prepare_tilt_progress_env()
    term = _tilt_progress_term(env)
    term.reset()

    target_tilt = math.radians(150.0)
    partial_tilt = 0.55
    # The former 31.5-degree target must remain partial progress toward the physical drain angle.
    env._cup[:, 3:7] = torch.tensor([math.sin(0.5 * partial_tilt), 0.0, 0.0, math.cos(0.5 * partial_tilt)])
    partial = term(env) * env.step_dt
    env._cup[:, 3:7] = torch.tensor([math.sin(0.5 * target_tilt), 0.0, 0.0, math.cos(0.5 * target_tilt)])
    completion = term(env) * env.step_dt
    env._cup[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    reverse = term(env) * env.step_dt

    partial_fraction = partial_tilt / target_tilt
    torch.testing.assert_close(
        partial,
        torch.tensor([partial_fraction, partial_fraction, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        partial + completion,
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(partial + completion + reverse, torch.zeros(env.num_envs), atol=1.0e-6, rtol=0.0)


def test_tilt_progress_rejects_wrong_direction_unaligned_and_unheld_motion():
    env = _prepare_tilt_progress_env()
    env.curriculum_stage.zero_()
    term = _tilt_progress_term(env)
    term.reset()

    half_angle = 0.5 * 0.55
    env._cup[0, 3:7] = torch.tensor([-math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)])
    env._cup[1, 3:7] = torch.tensor([0.0, math.sin(half_angle), 0.0, math.cos(half_angle)])
    env._cup[2, 3:7] = torch.tensor([math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)])
    env._cup[2, 0] += 0.10
    env._cup[3, 3:7] = torch.tensor([math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)])
    env._gripper_command[3] = 0.04

    torch.testing.assert_close(term(env) * env.step_dt, torch.zeros(env.num_envs), atol=1.0e-6, rtol=0.0)


def test_tilt_progress_selective_reset_baselines_only_selected_worlds_and_release_repays():
    env = _prepare_tilt_progress_env()
    env.curriculum_stage.zero_()
    term = _tilt_progress_term(env)
    term.reset()

    half_angle = 0.5 * 0.55
    env._cup[:2, 3:7] = torch.tensor([math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)])
    term.reset(torch.tensor([0]))
    progress = term(env) * env.step_dt
    env._gripper_command[1] = 0.04
    release = term(env) * env.step_dt

    assert progress[0] == 0.0
    assert progress[1] > 0.0
    torch.testing.assert_close(progress[1] + release[1], torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(progress[2:], torch.zeros(2))


def test_pour_reference_progress_tracks_validated_path_and_is_stage_gated():
    env = _prepare_tilt_progress_env()
    env.curriculum_stage[:] = torch.tensor([0, 0, 0, 1])
    start_q = (0.0,) * 7
    target_q = (1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75)
    cfg = RewardTermCfg(
        func=rewards.PourReferenceProgress,
        weight=10.0,
        params={"start_q": start_q, "target_q": target_q, "active_stage": 0},
    )
    term = rewards.PourReferenceProgress(cfg, env)
    term.reset()

    env._arm_q[:] = 0.5 * torch.tensor(target_q)
    halfway = term(env, start_q=start_q, target_q=target_q) * env.step_dt
    env._arm_q[:] = torch.tensor(target_q)
    completion = term(env, start_q=start_q, target_q=target_q) * env.step_dt
    env._arm_q.zero_()
    reverse = term(env, start_q=start_q, target_q=target_q) * env.step_dt

    torch.testing.assert_close(halfway[:3], torch.full((3,), 0.5), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(completion[:3], torch.full((3,), 0.5), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close((halfway + completion + reverse)[:3], torch.zeros(3), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(halfway[3:], torch.zeros(1))


def test_progress_reset_baselines_selected_worlds_without_cross_world_history():
    env = FakeEnv()
    env._width[:] = env.gripper_grasp_width
    env._gripper_command[:] = 0.025
    env._tcp[:] = env._grasp
    env._cup[:, :3] = torch.tensor([0.50, 0.00, 0.00])
    cfg = RewardTermCfg(
        func=rewards.LiftProgress,
        weight=5.0,
        params={"target_height": 0.12, "reach_std": 0.10},
    )
    term = rewards.LiftProgress(cfg, env)
    term.reset()

    env._cup[:2, 2] = 0.06
    term.reset(torch.tensor([0]))
    progress = term(env) * env.step_dt

    assert progress[0] == 0.0
    assert progress[1] > 0.0
    torch.testing.assert_close(progress[2:], torch.zeros(2))


def test_state_finite_rejects_raw_nonfinite_cup_and_robot_state():
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

    finite = terminations._state_finite(
        robot_joint_pos,
        robot_joint_vel,
        tcp_body_q,
        cup_body_q,
        cup_lin_vel,
        cup_ang_vel,
        particle_pos,
    )

    assert finite.tolist() == [True, False, False, False, False, False]


def test_rigid_state_bounds_reject_each_extreme_finite_observation_source():
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

    in_bounds = terminations._rigid_state_in_bounds(
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


def test_pose_conversion_returns_identity_for_nonfinite_or_degenerate_quaternions():
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv

    env = SimpleNamespace(env_origins=torch.zeros((4, 3)))
    pose_w = torch.tensor(
        [
            [0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 2.0],
            [0.5, 0.0, 0.1, float("nan"), 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.1, 0.0, float("inf"), 0.0, 1.0],
            [0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    pose_e = FrankaPourEnv._pose_w_to_e(env, pose_w)

    assert torch.isfinite(pose_e).all()
    torch.testing.assert_close(pose_e[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(
        pose_e[1:, 3:7],
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1),
    )


def test_gripper_width_uses_open_width_for_nonfinite_joint_positions():
    from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv

    joint_pos = torch.tensor(
        [
            [0.02, 0.03],
            [float("nan"), 0.03],
            [0.02, float("inf")],
        ]
    )
    env = SimpleNamespace(
        _robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joint_pos))),
        _finger_joint_ids=[0, 1],
        gripper_open_width=0.08,
    )

    width = FrankaPourEnv.gripper_width(env)

    assert torch.isfinite(width).all()
    torch.testing.assert_close(width, torch.tensor([0.05, 0.08, 0.08]))


def test_particle_workspace_rejects_finite_outliers_per_environment():
    particle_pos_e = torch.zeros((3, 4, 3))
    particle_pos_e[0] = torch.tensor([0.5, 0.0, 0.2])
    particle_pos_e[1, 0] = torch.tensor([1.51, 0.0, 0.2])
    particle_pos_e[2, 0] = torch.tensor([0.5, 0.0, -0.51])

    inside = terminations._particles_in_workspace(
        particle_pos_e,
        lower_bound=(-0.5, -1.0, -0.5),
        upper_bound=(1.5, 1.0, 1.5),
    )

    assert inside.tolist() == [True, False, False]
