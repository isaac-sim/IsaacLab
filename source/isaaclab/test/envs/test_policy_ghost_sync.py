# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy Ghosts keep every slice of one environment on the reference slice's spawn state and commands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.utils.policy_ghost_sync import PolicyGhostSync

SLICE_SIZE = 3
NUM_SLICES = 3
NUM_ENVS = SLICE_SIZE * NUM_SLICES


class _Scene:
    """Scene whose only state is one root position per environment."""

    terrain = None

    def __init__(self) -> None:
        self.env_origins = torch.zeros(NUM_ENVS, 3)
        self.root_pos = torch.zeros(NUM_ENVS, 3)

    def get_state(self, is_relative: bool = False) -> dict:
        return {"articulation": {"robot": {"root_pose": self.root_pos.clone()}}}

    def reset_to(self, state: dict, env_ids: torch.Tensor, is_relative: bool = False) -> None:
        self.root_pos[env_ids] = state["articulation"]["robot"]["root_pose"]


class _CommandManager:
    """Command manager with one term that draws an independent random command per environment."""

    def __init__(self) -> None:
        self.term = SimpleNamespace(command=torch.zeros(NUM_ENVS, 3), label="velocity")
        self.active_terms = ["velocity"]

    def get_term(self, name: str) -> SimpleNamespace:
        return self.term

    def compute(self, dt: float) -> None:
        self.term.command[:] = torch.randn(NUM_ENVS, 3)


def _make_env() -> SimpleNamespace:
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device="cpu",
        scene=_Scene(),
        command_manager=_CommandManager(),
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
    PolicyGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env._reset_idx(torch.tensor([1, 2]))

    assert sorted(env.reset_calls[0]) == [1, 2, 4, 5, 7, 8]
    root_pos = env.scene.root_pos
    assert torch.equal(root_pos[[4, 5]], root_pos[[1, 2]])
    assert torch.equal(root_pos[[7, 8]], root_pos[[1, 2]])


def test_ghost_only_termination_waits_for_the_reference_reset() -> None:
    env = _make_env()
    PolicyGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env._reset_idx(torch.tensor([4, 8]))

    assert env.reset_calls == []


def test_commands_are_copied_from_the_reference_after_every_update() -> None:
    env = _make_env()
    PolicyGhostSync(env, SLICE_SIZE, NUM_SLICES)

    env.command_manager.compute(0.02)

    command = env.command_manager.term.command
    assert torch.equal(command[SLICE_SIZE : 2 * SLICE_SIZE], command[:SLICE_SIZE])
    assert torch.equal(command[2 * SLICE_SIZE :], command[:SLICE_SIZE])


def test_close_restores_the_environment_hooks() -> None:
    env = _make_env()
    reset_idx, compute = env._reset_idx, env.command_manager.compute
    sync = PolicyGhostSync(env, SLICE_SIZE, NUM_SLICES)

    sync.close()

    assert env._reset_idx == reset_idx
    assert env.command_manager.compute == compute


def test_environment_size_must_match_the_slices() -> None:
    with pytest.raises(ValueError, match="slices"):
        PolicyGhostSync(_make_env(), SLICE_SIZE, NUM_SLICES + 1)


def test_configure_env_cfg_keeps_only_randomization_the_reset_can_mirror() -> None:
    def term(mode: str) -> SimpleNamespace:
        return SimpleNamespace(mode=mode)

    cfg = SimpleNamespace(
        events=SimpleNamespace(mass=term("startup"), push=term("interval"), reset_base=term("reset"), removed=None),
        curriculum=SimpleNamespace(terrain_levels=object()),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
    )

    removed = PolicyGhostSync.configure_env_cfg(cfg)

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
        events=SimpleNamespace(push=SimpleNamespace(mode="interval")),
        curriculum=None,
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=False)),
    )

    assert PolicyGhostSync.configure_env_cfg(cfg) == ["events.push"]
    assert PolicyGhostSync.configure_env_cfg(SimpleNamespace()) == []  # No events, curriculum, or observations.
