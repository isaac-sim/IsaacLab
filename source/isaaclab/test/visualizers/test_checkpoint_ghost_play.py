# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks and session building that play scripts share for Checkpoint Ghosts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.visualizers import checkpoint_ghost_play as play


def _env_cfg(backend, scene=None, curriculum=None):
    cfg = ManagerBasedRLEnvCfg.__new__(ManagerBasedRLEnvCfg)
    cfg.sim = SimpleNamespace(visualizer_cfgs=[SimpleNamespace(backend=backend)])
    cfg.scene = scene or SimpleNamespace()
    cfg.curriculum = curriculum
    return cfg


def _args(**kwargs):
    return SimpleNamespace(checkpoint_ghosts=True, **kwargs)


def _newton(**options):
    return SimpleNamespace(visualizer_type="newton", **options)


def test_nothing_is_checked_or_changed_without_the_flag():
    args = SimpleNamespace(checkpoint_ghosts=False)

    play.check(args, object())
    play.prepare(args, object())


def test_newton_among_several_selected_viewers_is_accepted():
    backends = [SimpleNamespace(visualizer_type="viser"), _newton()]

    play.check(_args(), _env_cfg(backends))


def test_without_newton_the_run_is_rejected():
    with pytest.raises(SystemExit, match="requires the Newton"):
        play.check(_args(), _env_cfg(SimpleNamespace(visualizer_type="viser")))


def test_a_slot_count_out_of_range_is_rejected():
    with pytest.raises(SystemExit, match="between 2 and"):
        play.check(_args(), _env_cfg(_newton(checkpoint_ghost_slots=1)))


def test_the_flag_is_the_only_command_line_option():
    import argparse

    parser = argparse.ArgumentParser()
    play.add_args(parser)

    assert vars(parser.parse_args(["--checkpoint_ghosts"])) == {"checkpoint_ghosts": True}


def test_a_task_whose_curriculum_selects_the_reset_state_is_rejected():
    class _ResetStateCurriculum:
        supports_checkpoint_ghosts = False

    curriculum = SimpleNamespace(reset_dataset=SimpleNamespace(func=_ResetStateCurriculum))

    with pytest.raises(SystemExit, match="reset_dataset"):
        play.check(_args(), _env_cfg(_newton(), curriculum=curriculum))


def test_an_ordinary_curriculum_term_is_accepted():
    curriculum = SimpleNamespace(terrain_levels=SimpleNamespace(func=lambda env, env_ids: None))

    play.check(_args(), _env_cfg(_newton(), curriculum=curriculum))


def _slots(monkeypatch, total: int, slots: int | None):
    monkeypatch.setattr(play.CheckpointGhostSync, "configure_env_cfg", Mock())
    cfg = _env_cfg(_newton(checkpoint_ghost_slots=slots), scene=SimpleNamespace(num_envs=total))
    play.prepare(_args(), cfg)
    return play._slot_count(cfg, total)


def test_the_slots_default_to_eight_capped_at_num_envs(monkeypatch):
    assert _slots(monkeypatch, 64, None) == 8
    assert _slots(monkeypatch, 3, None) == 3


def test_more_slots_than_environments_are_capped(monkeypatch):
    assert _slots(monkeypatch, 4, 6) == 4
    assert _slots(monkeypatch, 32, 5) == 5


def test_a_single_environment_cannot_be_split(monkeypatch):
    with pytest.raises(SystemExit, match="at least 2"):
        _slots(monkeypatch, 1, None)


def _run_env(viewers, **options):
    cfg = _env_cfg(_newton(**options))
    cfg.seed = 0
    return SimpleNamespace(
        unwrapped=SimpleNamespace(sim=SimpleNamespace(visualizers=viewers), step_dt=0.02, num_envs=16, cfg=cfg)
    )


def test_the_session_is_built_from_the_play_arguments_and_newton_options(monkeypatch, tmp_path):
    created = {}

    class _Session:
        def __init__(self, env, **kwargs):
            created.update(kwargs)

        def step(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(play, "CheckpointGhostSession", _Session)
    checkpoint = tmp_path / "run" / "model_5.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    viewer = Mock()
    env = _run_env([viewer], checkpoint_ghost_slots=4, checkpoint_ghost_root=str(tmp_path / "ghosts"))
    args = SimpleNamespace(task="Isaac-Reach-Franka", deterministic=False)
    run_playback = Mock()

    play.run(
        env,
        args,
        policy_factory=Mock(),
        checkpoint_pattern=r"model_.*\.pt",
        reference_path=str(checkpoint),
        log_root_path=str(tmp_path),
        run_playback=run_playback,
    )

    assert created["num_slots"] == 4
    assert created["task"] == "Isaac-Reach-Franka"
    assert created["checkpoint_root"] == str(tmp_path / "ghosts")  # the configured folder, not the log directory
    assert created["reference"].name == "model_5.pt"
    viewer.attach_checkpoint_ghosts.assert_called_once()
    run_playback.assert_called_once()


def test_the_session_is_closed_when_no_viewer_can_attach(monkeypatch, tmp_path):
    session = Mock()
    monkeypatch.setattr(play, "CheckpointGhostSession", Mock(return_value=session))
    checkpoint = tmp_path / "model_5.pt"
    checkpoint.touch()
    env = _run_env([object()])
    args = SimpleNamespace(task="t", deterministic=False)

    with pytest.raises(RuntimeError, match="initialized Newton GL visualizer"):
        play.run(
            env,
            args,
            policy_factory=Mock(),
            checkpoint_pattern="",
            reference_path=str(checkpoint),
            log_root_path=str(tmp_path),
            run_playback=Mock(),
        )

    session.close.assert_called_once()  # its reset and command hooks must not stay installed


def test_the_task_of_a_run_is_read_from_its_manifest(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text(json.dumps({"task": "Isaac-Reach-Franka-OSC"}))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "run.json").write_text("not json")

    assert play._run_task(str(tmp_path / "run")) == "Isaac-Reach-Franka-OSC"
    assert play._run_task(str(tmp_path / "bad")) is None
    assert play._run_task(str(tmp_path / "missing")) is None


def test_a_direct_environment_is_accepted():
    from isaaclab.envs import DirectRLEnvCfg

    cfg = DirectRLEnvCfg.__new__(DirectRLEnvCfg)
    cfg.sim = SimpleNamespace(visualizer_cfgs=[SimpleNamespace(backend=_newton())])
    cfg.action_space = 6

    play.check(_args(), cfg)


def test_a_policy_adapter_clips_the_actions_and_forwards_resets():
    resets = []
    policy = play.PolicyAdapter(lambda obs: obs["policy"] * 10, resets.append, clip=1.0)

    actions = policy({"policy": torch.tensor([[0.5, -0.5]])})
    policy.reset()
    policy.reset(torch.tensor([True]))

    assert actions.tolist() == [[1.0, -1.0]]
    assert resets[0] is None and resets[1].tolist() == [True]
    play.PolicyAdapter(lambda obs: obs).reset()  # a policy without recurrent state needs no reset function


def test_observation_groups_are_sliced_per_slot():
    from isaaclab.visualizers.checkpoint_ghost_session import _slice

    obs = {"policy": torch.arange(6).view(6, 1), "critic": {"a": torch.arange(6)}}

    sliced = _slice(obs, slice(2, 4))

    assert sliced["policy"].tolist() == [[2], [3]] and sliced["critic"]["a"].tolist() == [2, 3]


@pytest.mark.parametrize("action_space", [{3}, {"a": 2}])
def test_a_direct_environment_with_discrete_or_composite_actions_is_rejected(action_space):
    from isaaclab.envs import DirectRLEnvCfg

    cfg = DirectRLEnvCfg.__new__(DirectRLEnvCfg)
    cfg.sim = SimpleNamespace(visualizer_cfgs=[SimpleNamespace(backend=_newton())])
    cfg.action_space = action_space

    with pytest.raises(SystemExit, match="continuous"):
        play.check(_args(), cfg)


def test_a_non_torch_skrl_framework_is_rejected():
    with pytest.raises(SystemExit, match="torch"):
        play.check(_args(ml_framework="jax"), object())


def test_a_deprecated_combined_config_holds_its_own_newton_options():
    combined = SimpleNamespace(visualizer_type="newton", backend=_newton(), checkpoint_ghost_slots=3)
    cfg = ManagerBasedRLEnvCfg.__new__(ManagerBasedRLEnvCfg)
    cfg.sim = SimpleNamespace(visualizer_cfgs=[combined])
    cfg.curriculum = None

    play.check(_args(), cfg)

    assert play._slot_count(cfg, 16) == 3
