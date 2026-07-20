# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mask-first manager-based RL reset orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import warp as wp
from isaaclab_experimental.envs.manager_based_rl_env_warp import ManagerBasedRLEnvWarp


def test_reset_without_host_consumers_does_not_compact_ids() -> None:
    """The normal Warp reset path should not call :meth:`torch.Tensor.nonzero`."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = Mock()
    env.reset_buf.any.return_value.item.return_value = True
    env.reset_buf.nonzero.side_effect = AssertionError("reset IDs should not be materialized")
    env._reset_idx = Mock()
    env._has_recorders = False
    env.has_rtx_sensors = False
    env.cfg = SimpleNamespace(compute_final_obs=False, num_rerenders_on_reset=0)
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    env.reset_buf.nonzero.assert_not_called()
    env._reset_idx.assert_called_once_with(env_mask=reset_mask)


def test_curriculum_uses_mask_before_scene_reset() -> None:
    """Curriculum updates should stay mask-native and precede scene reset."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    stages: list[tuple[str, dict]] = []

    def call(stage, fn, /, *args, **kwargs):
        del fn, args
        stages.append((stage, kwargs))
        return None if stage.endswith("_compute") else {}

    env._warp_graph_cache = SimpleNamespace(call=call)
    env.sim = SimpleNamespace(device="cpu")
    env.reset_mask_wp = wp.zeros(3, dtype=wp.bool, device="cpu")
    env.scene = SimpleNamespace(
        num_envs=3,
        reset=Mock(side_effect=lambda **kwargs: stages.append(("Scene_reset", kwargs))),
        surface_grippers={},
    )
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

    env._warp_graph_cache = SimpleNamespace(call=call)
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


def test_reset_compacts_ids_only_for_active_recorders() -> None:
    """Recorder callbacks should receive IDs while the Warp reset keeps its mask."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.tensor([False, True, False])
    env._reset_idx = Mock()
    env.recorder_manager = Mock()
    env.recorder_manager.reset.return_value = {"recorder": 1.0}
    env._has_recorders = True
    env.has_rtx_sensors = False
    env.cfg = SimpleNamespace(compute_final_obs=False, num_rerenders_on_reset=0)
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    reset_ids = env.recorder_manager.record_pre_reset.call_args.args[0]
    torch.testing.assert_close(reset_ids, torch.tensor([1]))
    env._reset_idx.assert_called_once_with(env_mask=reset_mask)
    env.recorder_manager.reset.assert_called_once_with(reset_ids)
    env.recorder_manager.record_post_reset.assert_called_once_with(reset_ids)
    assert env.extras["log"]["recorder"] == 1.0


def test_rtx_rerender_does_not_require_compact_reset_ids() -> None:
    """RTX reset rendering should use the nonempty predicate without materializing IDs."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.tensor([False, True, False])
    env._reset_idx = Mock()
    env._has_recorders = False
    env.has_rtx_sensors = True
    env.cfg = SimpleNamespace(compute_final_obs=False, num_rerenders_on_reset=2)
    env.sim = SimpleNamespace(render=Mock())
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    env._reset_idx.assert_called_once_with(env_mask=reset_mask)
    assert env.sim.render.call_count == 2


def test_empty_reset_skips_warp_pipeline_without_compacting_ids() -> None:
    """An empty mask should skip eager reset stages without materializing IDs."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.zeros(3, dtype=wp.bool, device="cpu")
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.zeros(3, dtype=torch.bool)
    env._reset_idx = Mock()
    env._has_recorders = False
    env.extras = {"log": {"previous": 1.0}}

    env._reset_terminated_envs()

    env._reset_idx.assert_not_called()
    assert env.extras["log"] == {"previous": 1.0}


def test_reset_captures_final_observation_before_reset() -> None:
    """Same-step autoreset should expose the terminal observation before reset."""
    env = ManagerBasedRLEnvWarp.__new__(ManagerBasedRLEnvWarp)
    reset_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    final_obs = {"policy": torch.tensor([[1.0], [2.0], [3.0]])}
    env.termination_manager = SimpleNamespace(dones_wp=reset_mask)
    env.reset_buf = torch.tensor([False, True, False])
    env.observation_manager = SimpleNamespace(compute=Mock(return_value=final_obs))
    env._reset_idx = Mock()
    env._has_recorders = False
    env.has_rtx_sensors = False
    env.cfg = SimpleNamespace(compute_final_obs=True, num_rerenders_on_reset=0)
    env.extras = {"log": {}}

    env._reset_terminated_envs()

    assert env.extras["final_obs"] is final_obs
    env.observation_manager.compute.assert_called_once_with()
    env._reset_idx.assert_called_once_with(env_mask=reset_mask)
