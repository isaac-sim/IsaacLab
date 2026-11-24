# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks and restart arguments of Checkpoint Ghosts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("rsl_rl")
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.entrypoints.backends import play_rsl_rl


def _env_cfg(backend, scene=None, curriculum=None):
    cfg = ManagerBasedRLEnvCfg.__new__(ManagerBasedRLEnvCfg)
    cfg.sim = SimpleNamespace(visualizer_cfgs=[SimpleNamespace(backend=backend)])
    cfg.scene = scene or SimpleNamespace()
    cfg.curriculum = curriculum
    return cfg


def test_newton_among_several_selected_viewers_is_accepted():
    backends = [SimpleNamespace(visualizer_type="viser"), SimpleNamespace(visualizer_type="newton")]

    play_rsl_rl._validate_checkpoint_ghosts(SimpleNamespace(), _env_cfg(backends))


def test_without_newton_the_run_is_rejected():
    with pytest.raises(SystemExit, match="requires the Newton"):
        play_rsl_rl._validate_checkpoint_ghosts(SimpleNamespace(), _env_cfg(SimpleNamespace(visualizer_type="viser")))


def _slots(total: int, slots: int | None):
    args = SimpleNamespace(checkpoint_ghost_slots=slots)
    play_rsl_rl._resolve_ghost_slots(args, SimpleNamespace(scene=SimpleNamespace(num_envs=total)))
    return args.checkpoint_ghost_slots


def test_the_slots_default_to_eight_capped_at_num_envs():
    assert _slots(64, None) == 8
    assert _slots(3, None) == 3


def test_more_slots_than_environments_are_capped():
    assert _slots(4, 6) == 4
    assert _slots(32, 5) == 5


def test_a_single_environment_cannot_be_split():
    with pytest.raises(SystemExit, match="at least 2"):
        _slots(1, None)


def test_the_session_is_built_from_the_play_arguments(monkeypatch, tmp_path):
    created = {}

    class _Session:
        def __init__(self, env, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(play_rsl_rl, "CheckpointGhostSession", _Session)
    checkpoint = tmp_path / "run" / "model_5.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    args = SimpleNamespace(checkpoint_ghost_root=None, task="Isaac-Reach-Franka")

    play_rsl_rl.create_checkpoint_ghost_session(Mock(), args, Mock(), str(checkpoint), str(tmp_path), 4)

    assert created["num_slots"] == 4
    assert created["task"] == "Isaac-Reach-Franka"
    assert created["checkpoint_root"] == str(tmp_path)
    assert created["reference"].name == "model_5.pt"


def test_a_task_whose_curriculum_selects_the_reset_state_is_rejected():
    class _ResetStateCurriculum:
        supports_checkpoint_ghosts = False

    curriculum = SimpleNamespace(reset_dataset=SimpleNamespace(func=_ResetStateCurriculum))

    with pytest.raises(SystemExit, match="reset_dataset"):
        play_rsl_rl._validate_checkpoint_ghosts(
            SimpleNamespace(), _env_cfg(SimpleNamespace(visualizer_type="newton"), curriculum=curriculum)
        )


def test_an_ordinary_curriculum_term_is_accepted():
    curriculum = SimpleNamespace(terrain_levels=SimpleNamespace(func=lambda env, env_ids: None))

    play_rsl_rl._validate_checkpoint_ghosts(
        SimpleNamespace(), _env_cfg(SimpleNamespace(visualizer_type="newton"), curriculum=curriculum)
    )


def test_checkpoint_selectors_pass_the_library_and_agent_to_the_resolver(monkeypatch):
    resolver = Mock(return_value="/runs/model_5.pt")
    monkeypatch.setattr(play_rsl_rl, "resolve_checkpoint_selector", resolver)
    args = SimpleNamespace(checkpoint="latest", task="Isaac-Reach-Franka", agent="rsl_rl_cfg_entry_point")

    path = play_rsl_rl._resolve_checkpoint(args, Mock(), Mock(), "/logs/rsl_rl/reach_franka")

    assert path == "/runs/model_5.pt"
    assert resolver.call_args.kwargs["library"] == "rsl_rl"
    assert resolver.call_args.kwargs["metadata"] == {"agent": "rsl_rl_cfg_entry_point"}


def test_the_session_is_closed_when_no_viewer_can_attach(monkeypatch):
    session = Mock()
    monkeypatch.setattr(play_rsl_rl, "create_checkpoint_ghost_session", Mock(return_value=session))
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace(visualizers=[object()]), step_dt=0.02))

    with pytest.raises(RuntimeError, match="initialized Newton GL visualizer"):
        play_rsl_rl.run_checkpoint_ghosts(
            env, SimpleNamespace(checkpoint_ghost_slots=2), Mock(), "/runs/model_5.pt", "/logs"
        )

    session.close.assert_called_once()  # its reset and command hooks must not stay installed
