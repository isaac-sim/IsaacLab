# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mask-first manager-based RL reset orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental.envs.manager_based_rl_env_warp import ManagerBasedRLEnvWarp
from isaaclab_experimental.utils.manager_call_switch import ManagerCallSwitch


def test_warp_environment_deprecates_and_preserves_stable_manager_profile() -> None:
    """The legacy stable-manager profile should warn while retaining its config bridge."""
    cfg = SimpleNamespace(rewards="warp_rewards", actions="warp_actions")
    switch = ManagerCallSwitch(cfg_source={"default": 1, "RewardManager": 0})
    switch._resolve_stable_cfg_counterpart = Mock(
        return_value=SimpleNamespace(rewards="stable_rewards", actions="stable_actions")
    )

    with pytest.warns(DeprecationWarning, match="STABLE managers"):
        switch.apply_term_cfg_profile(cfg)

    assert cfg.rewards == "stable_rewards"
    assert cfg.actions == "warp_actions"


def test_reset_without_host_consumers_does_not_compact_ids() -> None:
    """The normal Warp reset path should not call :meth:`torch.Tensor.nonzero`."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = Mock()
    env.reset_buf.nonzero.side_effect = AssertionError("reset IDs should not be materialized")
    env._reset_requires_host_selection = Mock(return_value=False)
    env._reset_idx = Mock()
    env.has_rtx_sensors = False
    env.cfg = SimpleNamespace(num_rerenders_on_reset=0)
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    env.reset_buf.nonzero.assert_not_called()
    env._reset_idx.assert_called_once_with(env_mask=reset_mask)


def test_public_reset_selection_stays_as_mask_without_host_consumers() -> None:
    """Mask-native reset selection should not be compacted preemptively."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    env._reset_requires_host_selection = Mock(return_value=False)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")

    result = env._resolve_reset_host_ids(env_ids=None, env_mask=reset_mask)

    assert result is None


def test_public_reset_compacts_mask_only_for_host_consumer() -> None:
    """A mask should cross to IDs only at an enabled host reset boundary."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    env._reset_requires_host_selection = Mock(return_value=True)
    reset_mask = wp.array([False, True, False, True], dtype=wp.bool, device="cpu")

    result = env._resolve_reset_host_ids(env_ids=None, env_mask=reset_mask)

    torch.testing.assert_close(result, torch.tensor([1, 3]))


def test_curriculum_uses_mask_before_scene_reset() -> None:
    """Curriculum updates should stay mask-native and precede scene reset."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    stages: list[tuple[str, dict]] = []

    def call(stage, fn, /, *args, **kwargs):
        del fn, args
        stages.append((stage, kwargs))
        return None if stage.endswith("_compute") or stage == "Scene_reset" else {}

    env._manager_call_switch = SimpleNamespace(call=call)
    env.sim = SimpleNamespace(device="cpu")
    env.reset_mask_wp = wp.zeros(3, dtype=wp.bool, device="cpu")
    env.scene = SimpleNamespace(num_envs=3, reset=Mock(), surface_grippers={})
    env.curriculum_manager = SimpleNamespace(compute=Mock(), reset=Mock())
    env.event_manager = SimpleNamespace(available_modes=[], reset=Mock())
    env.observation_manager = SimpleNamespace(reset=Mock())
    env.action_manager = SimpleNamespace(reset=Mock())
    env.reward_manager = SimpleNamespace(reset=Mock())
    env.command_manager = SimpleNamespace(reset=Mock())
    env.termination_manager = SimpleNamespace(reset=Mock())
    env._episode_length_buf_wp = wp.ones(3, dtype=wp.int64, device="cpu")
    env.extras = {}
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env._reset_idx(env_mask=reset_mask)

    assert [stage for stage, _ in stages[:2]] == ["CurriculumManager_compute", "Scene_reset"]
    assert stages[0][1]["env_mask"] is env.reset_mask_wp
    np.testing.assert_array_equal(stages[0][1]["env_mask"].numpy(), reset_mask.numpy())
    assert "env_ids" not in stages[0][1]
    assert "CurriculumManager_reset" in [stage for stage, _ in stages]


def test_reset_stages_reuse_owner_mask_for_sparse_and_empty_selections() -> None:
    """Every reset dispatch should receive the stable owner-held mask pointer."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    seen: list[tuple[wp.array, np.ndarray]] = []

    def call(stage, fn, /, *args, **kwargs):
        del fn, args
        if stage == "CurriculumManager_compute":
            mask = kwargs["env_mask"]
            seen.append((mask, mask.numpy().copy()))
        return None if stage.endswith("_compute") or stage == "Scene_reset" else {}

    env._manager_call_switch = SimpleNamespace(call=call)
    env.sim = SimpleNamespace(device="cpu")
    env.reset_mask_wp = wp.zeros(3, dtype=wp.bool, device="cpu")
    env.scene = SimpleNamespace(num_envs=3, reset=Mock(), surface_grippers={})
    env.curriculum_manager = SimpleNamespace(compute=Mock(), reset=Mock())
    env.event_manager = SimpleNamespace(available_modes=[], reset=Mock())
    env.observation_manager = SimpleNamespace(reset=Mock())
    env.action_manager = SimpleNamespace(reset=Mock())
    env.reward_manager = SimpleNamespace(reset=Mock())
    env.command_manager = SimpleNamespace(reset=Mock())
    env.termination_manager = SimpleNamespace(reset=Mock())
    env._episode_length_buf_wp = wp.ones(3, dtype=wp.int64, device="cpu")
    env.extras = {}

    env._reset_idx(env_mask=wp.array([False, True, False], dtype=wp.bool, device="cpu"))
    env._reset_idx(env_mask=wp.zeros(3, dtype=wp.bool, device="cpu"))

    assert seen[0][0] is env.reset_mask_wp
    assert seen[1][0] is env.reset_mask_wp
    np.testing.assert_array_equal(seen[0][1], [False, True, False])
    np.testing.assert_array_equal(seen[1][1], [False, False, False])


def test_reset_compacts_ids_for_enabled_host_consumer() -> None:
    """Host-only reset hooks should receive IDs while the Warp stage keeps its mask."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.tensor([False, True, False])
    env._reset_requires_host_selection = Mock(return_value=True)
    env._reset_idx = Mock()
    env._reset_host_pre = Mock()
    env._reset_host_post = Mock(return_value={"host": 1.0})
    env.recorder_manager = Mock()
    env._has_recorders = False
    env.has_rtx_sensors = False
    env.cfg = SimpleNamespace(num_rerenders_on_reset=0)
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    reset_ids = env._reset_host_pre.call_args.args[0]
    torch.testing.assert_close(reset_ids, torch.tensor([1]))
    env._reset_idx.assert_called_once_with(env_mask=reset_mask)
    assert env.extras["log"]["host"] == 1.0


def test_rtx_rerender_does_not_require_compact_reset_ids() -> None:
    """RTX reset rendering should use the nonempty predicate without materializing IDs."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.tensor([False, True, False])
    env._reset_requires_host_selection = Mock(return_value=False)
    env._reset_idx = Mock()
    env._reset_host_pre = Mock()
    env._reset_host_post = Mock()
    env._has_recorders = False
    env.has_rtx_sensors = True
    env.cfg = SimpleNamespace(num_rerenders_on_reset=2)
    env.sim = SimpleNamespace(render=Mock())
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    env._reset_idx.assert_called_once_with(env_mask=reset_mask)
    env._reset_host_pre.assert_not_called()
    env._reset_host_post.assert_not_called()
    assert env.sim.render.call_count == 2


def test_empty_reset_skips_legacy_host_boundary_and_preserves_logs() -> None:
    """An empty host-visible selection should not execute reset side effects."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.zeros(3, dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.zeros(3, dtype=torch.bool)
    env._reset_requires_host_selection = Mock(return_value=True)
    env._reset_idx = Mock()
    env._reset_host_pre = Mock()
    env._reset_host_post = Mock()
    env._has_recorders = False
    env.extras = {"log": {"previous": 1.0}}

    env._reset_terminated_envs()

    env._reset_idx.assert_not_called()
    env._reset_host_pre.assert_not_called()
    env._reset_host_post.assert_not_called()
    assert env.extras["log"] == {"previous": 1.0}


def test_empty_reset_skips_warp_pipeline_without_compacting_ids() -> None:
    """An empty mask should skip eager reset stages without materializing IDs."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.zeros(3, dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.zeros(3, dtype=torch.bool)
    env._reset_requires_host_selection = Mock(return_value=False)
    env._reset_idx = Mock()
    env.extras = {"log": {"previous": 1.0}}

    env._reset_terminated_envs()

    env._reset_idx.assert_not_called()
    assert env.extras["log"] == {"previous": 1.0}
