# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for recurring command and action manager launch replay."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp
from isaaclab_experimental.managers import ActionManager, CommandTerm


class _LaunchRecorder:
    """Record cached-launch sites while executing kernels eagerly."""

    def __init__(self, device: str):
        self._device = device
        self.sites: list[Hashable | None] = []

    def launch(
        self,
        kernel: wp.Kernel,
        dim: int | Sequence[int],
        inputs=(),
        outputs=(),
        *,
        stream: wp.Stream | None = None,
        site: Hashable | None = None,
    ) -> None:
        self.sites.append(site)
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=self._device, stream=stream)


class _Env:
    """Minimal environment owner for manager launch tests."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = "cpu"
        self._warp_launch = _LaunchRecorder(self.device)
        self.rng_state_wp = wp.array(range(11, 11 + num_envs), dtype=wp.uint32, device=self.device)
        self.sim = SimpleNamespace(
            vis_marker_registry=SimpleNamespace(
                add_debug_vis_callback=lambda term: None,
                clear_debug_vis_callback=lambda term: None,
            )
        )


class _CommandTerm(CommandTerm):
    """Minimal command term exposing the base recurring launches."""

    def __init__(self, env: _Env):
        cfg = SimpleNamespace(debug_vis=False, resampling_time_range=(1.0, 1.0))
        super().__init__(cfg, env)
        self._command = wp.zeros((self.num_envs, 1), dtype=wp.float32, device=self.device)
        self.metrics["error"] = wp.ones(self.num_envs, dtype=wp.float32, device=self.device)
        self._prepare_reset_extras()
        self.resample_masks: list[wp.array] = []

    @property
    def command(self) -> wp.array:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_mask: wp.array) -> None:
        self.resample_masks.append(env_mask)

    def _update_command(self) -> None:
        pass


def test_command_terms_use_instance_and_mask_specific_launch_sites() -> None:
    """Command replay sites should not collide across term instances or persistent masks."""
    env = _Env()
    first_term = _CommandTerm(env)
    second_term = _CommandTerm(env)
    reset_mask = wp.array([True, False, True, False], dtype=wp.bool, device=env.device)

    assert env._warp_launch.sites == []

    first_term.reset(env_mask=reset_mask)
    first_term.compute(dt=0.1)
    second_term.compute(dt=0.1)
    wp.synchronize()

    assert env._warp_launch.sites == [
        ("command_term", first_term, "reset_count", reset_mask),
        ("command_term", first_term, "reset_scale"),
        ("command_term", first_term, "reset_metric", "error", reset_mask),
        ("command_term", first_term, "reset_counter", reset_mask),
        ("command_term", first_term, "resample_time_left", reset_mask, 1.0, 1.0),
        ("command_term", first_term, "step_time_left", 0.1),
        ("command_term", first_term, "resample_time_left", first_term._resample_mask_wp, 1.0, 1.0),
        ("command_term", second_term, "step_time_left", 0.1),
        ("command_term", second_term, "resample_time_left", second_term._resample_mask_wp, 1.0, 1.0),
    ]
    assert first_term.resample_masks == [reset_mask, first_term._resample_mask_wp]
    assert second_term.resample_masks == [second_term._resample_mask_wp]


def test_action_manager_uses_mask_specific_launch_sites() -> None:
    """Action history reset should replay independently for each persistent mask."""
    env = _Env()
    action_term = SimpleNamespace(action_dim=2, reset=Mock())
    manager = ActionManager.__new__(ActionManager)
    manager._env = env
    manager._terms = {"action": action_term}
    manager._action = wp.full((env.num_envs, 2), 1.0, dtype=wp.float32, device=env.device)
    manager._prev_action = wp.full((env.num_envs, 2), 2.0, dtype=wp.float32, device=env.device)
    reset_mask = wp.array([True, False, True, False], dtype=wp.bool, device=env.device)

    manager.reset(env_mask=reset_mask)
    wp.synchronize()

    assert env._warp_launch.sites == [
        ("action_manager", manager, "reset_previous", reset_mask),
        ("action_manager", manager, "reset_current", reset_mask),
    ]
    action_term.reset.assert_called_once_with(env_mask=reset_mask)
    np.testing.assert_array_equal(manager._action.numpy(), [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(manager._prev_action.numpy(), [[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [2.0, 2.0]])
