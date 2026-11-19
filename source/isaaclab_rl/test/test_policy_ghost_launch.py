# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks and restart arguments of Rollout Ghosts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("rsl_rl")
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.entrypoints.backends import play_rsl_rl
from isaaclab_rl.entrypoints.policy_ghosts import _launch_argv


def _env_cfg(backend, scene=None):
    cfg = ManagerBasedRLEnvCfg.__new__(ManagerBasedRLEnvCfg)
    cfg.sim = SimpleNamespace(visualizer_cfgs=[SimpleNamespace(backend=backend)])
    cfg.scene = scene or SimpleNamespace()
    return cfg


def test_newton_among_several_selected_viewers_is_accepted():
    backends = [SimpleNamespace(visualizer_type="viser"), SimpleNamespace(visualizer_type="newton")]

    play_rsl_rl._validate_policy_ghosts(SimpleNamespace(), _env_cfg(backends))


def test_without_newton_the_run_is_rejected():
    with pytest.raises(SystemExit, match="requires the Newton"):
        play_rsl_rl._validate_policy_ghosts(SimpleNamespace(), _env_cfg(SimpleNamespace(visualizer_type="viser")))


def test_cameras_that_render_with_ovrtx_are_rejected():
    camera = SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="ovrtx"))
    cfg = _env_cfg(SimpleNamespace(visualizer_type="newton"), SimpleNamespace(camera=camera))

    with pytest.raises(SystemExit, match="OVRTX"):
        play_rsl_rl._validate_policy_ghosts(SimpleNamespace(), cfg)


def test_a_restart_uses_the_arguments_saved_before_parsing(monkeypatch):
    original = ["python", "scripts/play.py", "--task", "Isaac-Velocity-Rough-AnymalD", "--viz", "newton"]
    parsed_later = ["scripts/play.py", "visualizer=newton"]  # what _parse_args leaves in sys.argv
    monkeypatch.setattr(sys, "argv", parsed_later)

    assert (
        _launch_argv((original, ["scripts/play.py", "--task", "Isaac-Velocity-Rough-AnymalD", "--viz", "newton"]))
        == original
    )
    assert "--task" in " ".join(_launch_argv((original, original[1:])))
