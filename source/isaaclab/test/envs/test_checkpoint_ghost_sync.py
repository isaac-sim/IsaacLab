# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint Ghosts keep every slice of one environment on the reference slice's spawn state and commands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.utils.checkpoint_ghost_sync import CheckpointGhostSync

SLICE_SIZE = 3
NUM_SLICES = 3
NUM_ENVS = SLICE_SIZE * NUM_SLICES


class _Scene:
    """Scene whose only state is one root position per environment."""

    terrain = None

    def __init__(self, num_envs: int = NUM_ENVS) -> None:
        self.env_origins = torch.zeros(num_envs, 3)
        self.root_pos = torch.zeros(num_envs, 3)
        self.articulations = {
            "robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=self.root_pos)))
        }

    def get_state(self, is_relative: bool = False) -> dict:
        return {"articulation": {"robot": {"root_pose": self.root_pos.clone()}}}

    def reset_to(self, state: dict, env_ids: torch.Tensor, is_relative: bool = False) -> None:
        self.root_pos[env_ids] = state["articulation"]["robot"]["root_pose"]


class _CommandManager:
    """Command manager with one term that draws an independent random command per environment."""

    def __init__(self, num_envs: int = NUM_ENVS) -> None:
        self.num_envs = num_envs
        self.term = SimpleNamespace(command=torch.zeros(num_envs, 3), label="velocity")
        self.active_terms = ["velocity"]

    def get_term(self, name: str) -> SimpleNamespace:
        return self.term

    def compute(self, dt: float) -> None:
        self.term.command[:] = torch.randn(self.num_envs, 3)


def _make_env(num_envs: int = NUM_ENVS) -> SimpleNamespace:
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_Scene(num_envs),
        command_manager=_CommandManager(num_envs),
        event_manager=object(),
        reset_calls=[],
    )

    def reset_idx(env_ids: torch.Tensor) -> None:
        # every reset environment spawns at its own random position, like a reset event would
        env.reset_calls.append(env_ids.tolist())
        env.scene.root_pos[env_ids] = torch.randn(len(env_ids), 3)

    env._reset_idx = reset_idx
    return env


def test_reference_reset_mirrors_spawn_state_to_ghost_counterparts() -> None:
    env = _make_env()
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env._reset_idx(torch.tensor([1, 2]))

    assert sorted(env.reset_calls[0]) == [1, 2, 4, 5, 7, 8]
    root_pos = env.scene.root_pos
    assert torch.equal(root_pos[[4, 5]], root_pos[[1, 2]])
    assert torch.equal(root_pos[[7, 8]], root_pos[[1, 2]])


def test_ghost_only_termination_waits_for_the_reference_reset() -> None:
    env = _make_env()
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env._reset_idx(torch.tensor([4, 8]))

    assert env.reset_calls == []


def test_commands_are_copied_from_the_reference_after_every_update() -> None:
    env = _make_env()
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env.command_manager.compute(0.02)

    command = env.command_manager.term.command
    assert torch.equal(command[SLICE_SIZE : 2 * SLICE_SIZE], command[:SLICE_SIZE])
    assert torch.equal(command[2 * SLICE_SIZE :], command[:SLICE_SIZE])


def test_world_frame_command_positions_are_moved_to_the_ghost_s_own_origin() -> None:
    env = _make_env()
    env.scene.env_origins[:] = torch.arange(NUM_ENVS, dtype=torch.float32)[:, None] * torch.tensor([10.0, 0.0, 0.0])
    term = env.command_manager.term
    term.pos_command_w = torch.zeros(NUM_ENVS, 3)
    term.heading_command_w = torch.zeros(NUM_ENVS)
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    # Environment 0 is told to walk to a spot 2 m in front of its own origin.
    term.pos_command_w[0] = env.scene.env_origins[0] + torch.tensor([2.0, 0.0, 0.0])
    term.heading_command_w[0] = 1.5
    env.command_manager.compute(0.02)

    ghost = SLICE_SIZE  # The ghost counterpart of environment 0 in the next slice.
    assert term.pos_command_w[ghost].tolist() == (env.scene.env_origins[ghost] + torch.tensor([2.0, 0.0, 0.0])).tolist()
    assert term.heading_command_w[ghost] == 1.5  # A heading is the same in every world.


def test_relative_targets_are_recomputed_from_each_ghost_s_own_state_not_copied() -> None:
    env = _make_env()
    term = env.command_manager.term
    term.pos_command_b = torch.zeros(NUM_ENVS, 3)
    own_offsets = torch.arange(NUM_ENVS, dtype=torch.float32)

    def update_command() -> None:  # the real term derives these from the target and the robot's own pose
        term.pos_command_b[:, 0] = 5.0 - own_offsets

    term._update_command = update_command
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env.command_manager.compute(0.02)

    assert term.pos_command_b[SLICE_SIZE, 0] == 5.0 - SLICE_SIZE  # the ghost's own distance, not the reference's
    assert term.pos_command_b[0, 0] == 5.0


def test_a_heading_derived_turn_command_is_recomputed_from_each_ghost_s_own_heading() -> None:
    env = _make_env()
    term = env.command_manager.term
    term.cfg = SimpleNamespace(heading_command=True)
    term.vel_command_b = torch.zeros(NUM_ENVS, 3)
    term.heading_target = torch.zeros(NUM_ENVS)
    own_heading = torch.arange(NUM_ENVS, dtype=torch.float32) * 0.1

    def update_command() -> None:  # like UniformVelocityCommand: yaw rate steers toward the heading target
        term.vel_command_b[:, 2] = term.heading_target - own_heading

    term._update_command = update_command
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)
    term.vel_command_b[0, 0], term.heading_target[0] = 1.0, 2.0

    env.command_manager.compute(0.02)

    ghost = SLICE_SIZE
    assert term.vel_command_b[ghost, 0] == 1.0  # the sampled forward speed is the reference's
    assert term.heading_target[ghost] == 2.0  # and so is the heading target
    assert term.vel_command_b[ghost, 2] == pytest.approx(2.0 - own_heading[ghost])  # but the turn is the ghost's own
    assert term.vel_command_b[0, 2] == pytest.approx(2.0)


def test_ghosts_hold_the_reference_s_joint_targets_after_a_reset() -> None:
    env = _make_env()
    targets = torch.zeros(NUM_ENVS, 2)
    env.scene.articulations["robot"].data.joint_pos_target = targets
    reset_idx = env._reset_idx

    def reset_with_targets(env_ids: torch.Tensor) -> None:
        reset_idx(env_ids)
        targets[env_ids] = 0.0  # the reset leaves the reference's targets at zero...

    def reset_to(state: dict, env_ids: torch.Tensor, is_relative: bool = False) -> None:
        env.scene.root_pos[env_ids] = state["articulation"]["robot"]["root_pose"]
        targets[env_ids] = 0.04  # ...while restoring a state sets the ghost's from its joint positions

    env._reset_idx = reset_with_targets
    env.scene.reset_to = reset_to
    sync = CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env._reset_idx(torch.arange(SLICE_SIZE))

    assert (targets == 0.0).all(), targets  # every slice ends up with the reference's targets
    sync.close()


def test_close_restores_the_environment_hooks() -> None:
    env = _make_env()
    reset_idx, compute = env._reset_idx, env.command_manager.compute
    sync = CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    sync.close()

    assert env._reset_idx == reset_idx
    assert env.command_manager.compute == compute


def test_environment_size_must_match_the_slices() -> None:
    with pytest.raises(ValueError, match="slices"):
        CheckpointGhostSync(_make_env(), SLICE_SIZE, NUM_SLICES + 1)


def test_configure_env_cfg_keeps_only_randomization_the_reset_can_mirror() -> None:
    def term(mode: str) -> SimpleNamespace:
        return SimpleNamespace(mode=mode, func=lambda: None)

    cfg = SimpleNamespace(
        events=SimpleNamespace(mass=term("startup"), push=term("interval"), reset_base=term("reset"), removed=None),
        curriculum=SimpleNamespace(terrain_levels=object()),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
    )

    removed = CheckpointGhostSync.configure_env_cfg(cfg)

    assert sorted(removed) == [
        "curriculum.terrain_levels",
        "events.mass",
        "events.push",
        "observations.policy.enable_corruption",
    ]
    assert cfg.events.reset_base.mode == "reset"
    assert cfg.events.mass is None and cfg.events.push is None
    assert cfg.curriculum.terrain_levels is None
    assert cfg.observations.policy.enable_corruption is False


def test_configure_env_cfg_accepts_a_task_without_a_curriculum() -> None:
    cfg = SimpleNamespace(
        events=SimpleNamespace(push=SimpleNamespace(mode="interval", func=lambda: None)),
        curriculum=None,
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=False)),
    )

    assert CheckpointGhostSync.configure_env_cfg(cfg) == ["events.push"]
    none = SimpleNamespace(events=None, curriculum=None, observations=None)
    assert CheckpointGhostSync.configure_env_cfg(none) == []  # A task with no events, curriculum, or observations.


def test_a_larger_pool_resets_the_environments_beyond_the_slices_on_their_own() -> None:
    env = _make_env(NUM_ENVS + 4)
    CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env._reset_idx(torch.tensor([NUM_ENVS + 1]))  # A parked environment resets on its own, without mirroring.
    assert env.reset_calls == [[NUM_ENVS + 1]]

    env._reset_idx(torch.tensor([0]))
    assert sorted(env.reset_calls[1]) == [0, 3, 6]
    env.command_manager.compute(0.02)
    command = env.command_manager.term.command
    assert torch.equal(command[SLICE_SIZE : 2 * SLICE_SIZE], command[:SLICE_SIZE])


def test_close_puts_the_ghost_environments_back_on_their_own_terrain() -> None:
    env = _make_env()
    env.scene.terrain = SimpleNamespace(
        terrain_origins=object(),
        terrain_levels=torch.arange(NUM_ENVS),
        terrain_types=torch.arange(NUM_ENVS),
        env_origins=torch.arange(NUM_ENVS * 3, dtype=torch.float32).reshape(NUM_ENVS, 3),
    )
    original = env.scene.terrain.env_origins.clone()
    sync = CheckpointGhostSync(env, SLICE_SIZE, NUM_SLICES)

    assert torch.equal(env.scene.terrain.env_origins[SLICE_SIZE : 2 * SLICE_SIZE], original[:SLICE_SIZE])
    sync.close()

    assert torch.equal(env.scene.terrain.env_origins, original)
    assert torch.equal(env.scene.terrain.terrain_levels, torch.arange(NUM_ENVS))


def test_configure_env_cfg_also_disables_reset_events_that_randomize_physics_but_keeps_a_task_s_own() -> None:
    from isaaclab.envs.mdp import events as mdp_events

    def randomize_gear_type() -> None:  # a task's own event: it creates state that its later reset events read
        pass

    def reset_root_state_uniform() -> None:
        pass

    def term(mode: str, func) -> SimpleNamespace:
        return SimpleNamespace(mode=mode, func=func)

    cfg = SimpleNamespace(
        events=SimpleNamespace(
            mass=term("reset", mdp_events.randomize_rigid_body_mass),
            pose=term("reset", reset_root_state_uniform),
            gear=term("reset", randomize_gear_type),
            startup_mass=term("startup", mdp_events.randomize_rigid_body_mass),
        ),
        curriculum=None,
        observations=None,
    )

    removed = CheckpointGhostSync.configure_env_cfg(cfg)

    assert sorted(removed) == ["events.mass", "events.startup_mass"]
    assert cfg.events.pose is not None and cfg.events.gear is not None  # these are mirrored or the task's own
