# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for when a manager-based environment lets physics handle its actions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab.envs.manager_based_env import _physics_can_handle_actions

pytestmark = pytest.mark.unit


def _env(*, render_interval=2, records_physics_steps=False, lazy_sensor_update=False, sensor=None):
    """An environment stepping 2 physics steps of 0.01 s per environment step, with at most one sensor."""
    return SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(render_interval=render_interval), decimation=2),
        step_dt=0.02,
        recorder_manager=SimpleNamespace(records_physics_steps=records_physics_steps),
        scene=SimpleNamespace(
            cfg=SimpleNamespace(lazy_sensor_update=lazy_sensor_update),
            sensors={} if sensor is None else {"sensor": SimpleNamespace(cfg=SimpleNamespace(**sensor))},
        ),
    )


@pytest.mark.parametrize(
    ("env", "handled"),
    [
        pytest.param(_env(render_interval=1), False, id="renders-every-physics-step"),
        pytest.param(_env(records_physics_steps=True), False, id="records-every-physics-step"),
        pytest.param(_env(sensor={"update_period": 0.02}), True, id="sensor-every-env-step"),
        pytest.param(_env(sensor={"update_period": 0.0}), False, id="sensor-every-physics-step"),
        pytest.param(_env(sensor={"update_period": 0.01}), False, id="sensor-mid-env-step"),
        pytest.param(_env(lazy_sensor_update=True, sensor={"update_period": 0.0}), True, id="lazy-sensor"),
        pytest.param(
            _env(lazy_sensor_update=True, sensor={"update_period": 0.0, "history_length": 2}),
            False,
            id="lazy-sensor-with-history",
        ),
    ],
)
def test_physics_handles_actions_only_when_nothing_runs_between_physics_steps(env, handled):
    """Renders, per-physics-step recorders and sensors sampled within an environment step keep actions in the loop."""
    assert _physics_can_handle_actions(env) is handled
