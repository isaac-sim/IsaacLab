# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the experimental Warp manager-based RL environment."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.envs.direct_rl_env_warp import DirectRLEnvWarp
from isaaclab_experimental.envs.manager_based_env_warp import ManagerBasedEnvWarp
from isaaclab_experimental.envs.manager_based_rl_env_warp import ManagerBasedRLEnvWarp
from isaaclab_experimental.managers.command_manager import CommandManager as ExperimentalCommandManager
from isaaclab_experimental.managers.command_manager import _CommandViewCache
from isaaclab_experimental.managers.reward_manager import RewardManager

from isaaclab.managers import CommandManager
from isaaclab.utils.warp import WarpLaunchCache


@pytest.mark.parametrize("env_type", [ManagerBasedEnvWarp, DirectRLEnvWarp])
def test_physics_ready_destroys_graphs_before_resetting_launch_cache(env_type, monkeypatch):
    """A full rebind should release graphs before delegating the cache drain."""
    calls = []
    env = object.__new__(env_type)
    env.sim = SimpleNamespace(device="cuda:0")
    env._warp_launch = SimpleNamespace(reset=lambda: calls.append("launch_reset"))
    if env_type is ManagerBasedEnvWarp:
        env._manager_call_switch = SimpleNamespace(invalidate_graphs=lambda: calls.append("graph_invalidate"))
    else:
        env._graph_cache = SimpleNamespace(invalidate=lambda: calls.append("graph_invalidate"))
    monkeypatch.setattr(wp, "synchronize_device", lambda device: pytest.fail(f"unexpected direct sync on {device}"))

    env._reset_warp_caches_after_physics_ready(None)

    assert calls == ["graph_invalidate", "launch_reset"]


def test_public_graph_invalidation_also_resets_recorded_launches(monkeypatch):
    """Topology changes should invalidate graphs before delegating the cache drain."""
    calls = []
    env = object.__new__(ManagerBasedRLEnvWarp)
    env.sim = SimpleNamespace(device="cuda:0")
    env._manager_call_switch = SimpleNamespace(invalidate_graphs=lambda: calls.append("graph_invalidate"))
    env._warp_launch = SimpleNamespace(reset=lambda: calls.append("launch_reset"))
    monkeypatch.setattr(wp, "synchronize_device", lambda device: pytest.fail(f"unexpected direct sync on {device}"))

    env.invalidate_wp_graphs()

    assert calls == ["graph_invalidate", "launch_reset"]


def test_environment_uses_warp_command_manager():
    """The Warp environment should instantiate the manager that exposes persistent Warp commands."""
    manager_type = ManagerBasedRLEnvWarp._CommandManagerWarpView
    assert issubclass(manager_type, CommandManager)
    assert hasattr(manager_type, "get_command_wp")
    assert "_CommandManagerWarpView" in ManagerBasedRLEnvWarp.load_managers.__code__.co_names


def test_warp_command_manager_stages_computed_commands(monkeypatch):
    """Command properties that allocate should be copied into a pointer-stable staging buffer."""

    class AllocatingCommandTerm:
        def __init__(self):
            self.position = torch.zeros((2, 2))
            self.heading = torch.zeros((2, 1))

        @property
        def command(self) -> torch.Tensor:
            return torch.cat((self.position, self.heading), dim=1)

        def compute(self, dt: float):
            self.position.add_(dt)
            self.heading.add_(2.0 * dt)

        def reset(self, env_ids=None) -> dict:
            self.position.fill_(3.0)
            self.heading.fill_(5.0)
            return {}

    term = AllocatingCommandTerm()

    def initialize_command_manager(manager, _cfg, _env):
        manager._terms = {"pose": term}
        manager._resolve_terms_handle = None

    monkeypatch.setattr(CommandManager, "__init__", initialize_command_manager)
    manager = ManagerBasedRLEnvWarp._CommandManagerWarpView(None, None)
    command_wp = manager.get_command_wp("pose")

    manager.compute(0.5)

    assert manager.get_command_wp("pose") is command_wp
    torch.testing.assert_close(torch.asarray(command_wp), term.command)

    manager.reset()

    assert manager.get_command_wp("pose") is command_wp
    torch.testing.assert_close(torch.asarray(command_wp), term.command)


def test_experimental_command_manager_refreshes_allocating_commands():
    """The exported manager should refresh the same persistent view after compute and reset."""

    class AllocatingCommandTerm:
        def __init__(self):
            self.position = torch.zeros((2, 2))
            self.heading = torch.zeros((2, 1))

        @property
        def command(self) -> torch.Tensor:
            return torch.cat((self.position, self.heading), dim=1)

        def compute(self, dt: float):
            self.position.add_(dt)
            self.heading.add_(2.0 * dt)

        def reset(self, env_mask: wp.array):
            del env_mask
            self.position.fill_(4.0)
            self.heading.fill_(6.0)

    term = AllocatingCommandTerm()
    manager = object.__new__(ExperimentalCommandManager)
    manager._terms = {"pose": term}
    manager._command_views = _CommandViewCache(["pose"], lambda name: manager._terms[name].command)
    manager._commands_wp = manager._command_views.views
    manager._reset_extras = {}
    manager._env = SimpleNamespace(num_envs=2, device="cpu")
    command_wp = manager.get_command_wp("pose")

    manager.compute(0.5)
    assert manager.get_command_wp("pose") is command_wp
    torch.testing.assert_close(torch.asarray(command_wp), term.command)

    manager.reset(env_mask=wp.ones(2, dtype=wp.bool, device="cpu"))
    assert manager.get_command_wp("pose") is command_wp
    torch.testing.assert_close(torch.asarray(command_wp), term.command)


def test_command_view_cache_refreshes_late_pointer_replacements():
    """A command that starts pointer-stable should remain current after replacing its tensor."""

    class ReplacingCommandTerm:
        def __init__(self):
            self.command = torch.zeros((2, 3))

    term = ReplacingCommandTerm()
    views = _CommandViewCache(["pose"], lambda name: term.command)
    command_wp = views.get("pose")

    term.command = torch.full_like(term.command, 7.0)
    views.refresh()

    assert views.get("pose") is command_wp
    torch.testing.assert_close(torch.asarray(command_wp), term.command)


def test_command_view_cache_refreshes_in_place_storage_replacements():
    """A tensor that repoints itself should not leave its persistent Warp view stale."""
    command = torch.zeros((2, 3))
    views = _CommandViewCache(["pose"], lambda _name: command)
    command_wp = views.get("pose")

    command.set_(torch.full_like(command, 7.0))
    views.refresh()

    assert views.get("pose") is command_wp
    torch.testing.assert_close(torch.asarray(command_wp), command)


def test_command_view_cache_rejects_layout_replacements():
    """Rebinding to the same storage with a new layout should fail clearly."""
    command = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    views = _CommandViewCache(["pose"], lambda _name: command)
    views.get("pose")

    command.set_(command.T)

    with pytest.raises(RuntimeError, match="changed tensor specialization"):
        views.refresh()


def test_command_view_cache_defers_unused_command_properties():
    """Constructing a manager should not evaluate a command until its Warp view is requested."""
    calls = 0

    def get_null_command(name: str):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"{name} does not generate a command")

    views = _CommandViewCache(["null"], get_null_command)

    assert calls == 0
    with pytest.raises(RuntimeError, match="does not generate a command"):
        views.get("null")
    assert calls == 1


@pytest.mark.parametrize(
    "command",
    [
        torch.zeros((2, 3), dtype=torch.float64),
        wp.zeros((2, 3), dtype=wp.int32, device="cpu"),
    ],
)
def test_command_view_cache_rejects_non_float32_commands(command):
    """The public float32 Warp-view contract should reject incompatible commands."""
    views = _CommandViewCache(["command"], lambda _name: command)

    with pytest.raises(TypeError, match="must use .*float32"):
        views.get("command")


def test_reward_manager_replay_updates_dt():
    """Reward finalization should use the current time step on every replay."""

    def fill_reward(_env, out: wp.array(dtype=wp.float32)):
        out.fill_(2.0)

    num_envs = 4
    env = SimpleNamespace(num_envs=num_envs, device="cpu", _warp_launch=WarpLaunchCache(device="cpu"))
    manager = object.__new__(RewardManager)
    manager._env = env
    manager._num_terms = 1
    manager._term_outs_wp = wp.zeros((1, num_envs), dtype=wp.float32, device="cpu")
    term_out = wp.array(
        ptr=manager._term_outs_wp.ptr,
        dtype=wp.float32,
        shape=(num_envs,),
        strides=(manager._term_outs_wp.strides[1],),
        device="cpu",
    )
    manager._term_cfgs = [SimpleNamespace(weight=3.0, func=fill_reward, out=term_out, params={})]
    manager._reward_wp = wp.zeros(num_envs, dtype=wp.float32, device="cpu")
    manager._step_reward_wp = wp.zeros((num_envs, 1), dtype=wp.float32, device="cpu")
    manager._term_weights_wp = wp.array([3.0], dtype=wp.float32, device="cpu")
    manager._episode_sums_wp = wp.zeros((1, num_envs), dtype=wp.float32, device="cpu")
    manager._reward_tensor_view = wp.to_torch(manager._reward_wp)

    reward_half_second = manager.compute(0.5).clone()
    reward_quarter_second = manager.compute(0.25).clone()

    torch.testing.assert_close(reward_half_second, torch.full((num_envs,), 3.0))
    torch.testing.assert_close(reward_quarter_second, torch.full((num_envs,), 1.5))
