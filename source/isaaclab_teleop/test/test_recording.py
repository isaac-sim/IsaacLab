# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-preserving demonstration export using the real recorder and HDF5 writer."""

from types import SimpleNamespace

import gymnasium as gym
import h5py
import pytest
import torch
from isaaclab_teleop.action_cfg import TeleopActionCfg
from isaaclab_teleop.recording import DemoRecording, DemoRecordingCfg, prepare_demo_recording
from isaaclab_teleop.source import TeleopSample
from isaaclab_teleop.source_cfg import TeleopSourceCfg

from isaaclab.managers import ActionTermCfg, RecorderManager, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.kitless


class _TaskRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        return "initial_state", torch.tensor([[100.0]])

    def record_pre_reset(self, env_ids):
        # A custom reset term must not turn a zero-step attempt into a demo.
        return "reset_annotation", torch.tensor([[42.0]])

    def record_pre_step(self):
        return "actions", self._env.action


@configclass
class _TaskRecorderCfg(RecorderManagerBaseCfg):
    task_signal = RecorderTermCfg(class_type=_TaskRecorder)


@pytest.fixture
def task_registration():
    task_id = "Isaac-Teleop-Recording-Test-v0"
    gym.register(task_id, entry_point="unused:Env")
    try:
        yield task_id
    finally:
        gym.envs.registry.pop(task_id)


def _env_cfg():
    return SimpleNamespace(
        actions={
            "robot": TeleopActionCfg(
                controller=ActionTermCfg(asset_name="robot"), source=TeleopSourceCfg(class_type=object)
            )
        },
        scene=SimpleNamespace(num_envs=1),
        sim=SimpleNamespace(dt=0.01, render_interval=1),
        decimation=1,
        recorders=_TaskRecorderCfg(),
        terminations={"success": object(), "time_out": object()},
        observations=SimpleNamespace(policy=SimpleNamespace(concatenate_terms=True)),
        commands=SimpleNamespace(goal=SimpleNamespace(resampling_time_range=(0.1, 0.2))),
    )


def test_preparation_preserves_task_and_existing_recorder(tmp_path, task_registration):
    env_cfg = _env_cfg()
    original_recorders = env_cfg.recorders
    terms, observations, commands = env_cfg.terminations, env_cfg.observations, env_cfg.commands
    cfg = prepare_demo_recording(env_cfg, DemoRecordingCfg(str(tmp_path / "demos")), task_registration, [0])
    assert cfg.dataset_file == str(tmp_path / "demos.hdf5")
    assert not (tmp_path / "demos.hdf5").exists()
    assert env_cfg.env_name == task_registration
    assert env_cfg.terminations is terms and env_cfg.terminations["time_out"] is not None
    assert env_cfg.observations is observations and observations.policy.concatenate_terms
    assert env_cfg.commands is commands and commands.goal.resampling_time_range == (0.1, 0.2)
    assert env_cfg.recorders.task_signal.class_type is _TaskRecorder
    assert env_cfg.recorders is not original_recorders
    assert not hasattr(original_recorders, "record_teleop_active")


@pytest.mark.parametrize("registered", [False, True])
def test_empty_recorder_uses_task_registration_then_standard_default(tmp_path, task_registration, registered):
    env_cfg = _env_cfg()
    env_cfg.recorders = RecorderManagerBaseCfg()
    if registered:
        gym.spec(task_registration).kwargs["demo_recorder_cfg_entry_point"] = _TaskRecorderCfg
    prepare_demo_recording(env_cfg, DemoRecordingCfg(str(tmp_path / "demos")), task_registration, [0])
    if registered:
        assert isinstance(env_cfg.recorders, _TaskRecorderCfg)
    else:
        assert env_cfg.recorders.record_pre_step_actions is not None
        assert env_cfg.recorders.record_initial_state is not None


@pytest.mark.parametrize("invalid", ["existing", "batch", "success", "count", "extension", "owner"])
def test_invalid_recording_fails_before_creating_dataset(tmp_path, task_registration, invalid):
    env_cfg = _env_cfg()
    path = tmp_path / "demos.hdf5"
    cfg = DemoRecordingCfg(str(path))
    if invalid == "existing":
        path.write_bytes(b"keep existing demonstrations")
    elif invalid == "batch":
        env_cfg.scene.num_envs = 2
    elif invalid == "success":
        env_cfg.terminations = None
    elif invalid == "count":
        cfg = DemoRecordingCfg(str(path), num_demos=-1)
    elif invalid == "extension":
        cfg = DemoRecordingCfg(str(tmp_path / "demos.csv"))
    else:
        env_cfg.actions["robot"].source = None
    with pytest.raises(FileExistsError if invalid == "existing" else ValueError):
        prepare_demo_recording(env_cfg, cfg, task_registration, [0])
    if invalid == "existing":
        assert path.read_bytes() == b"keep existing demonstrations"
    else:
        assert not path.exists()


@pytest.mark.parametrize("mode", ["succeeded", "all"])
def test_reset_boundaries_export_only_physical_episodes_and_drop_partial_close(tmp_path, task_registration, mode):
    env_cfg = _env_cfg()
    cfg = prepare_demo_recording(
        env_cfg, DemoRecordingCfg(str(tmp_path / "demos.hdf5"), num_demos=2, export_mode=mode), task_registration, [0]
    )
    source_term = SimpleNamespace(sample=TeleopSample(active=True, actions=torch.tensor([1.0])))
    success = torch.tensor([False])
    env = SimpleNamespace(
        cfg=env_cfg,
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        termination_manager=SimpleNamespace(active_terms=["success"], get_term=lambda name: success),
        action_manager=SimpleNamespace(get_term=lambda name: source_term),
        action=torch.tensor([[0.0]]),
    )
    env.recorder_manager = RecorderManager(env_cfg.recorders, env)
    recording = DemoRecording(cfg)

    def reset(*, explicit):
        if explicit:
            recording.before_explicit_reset(env)
        try:
            env.recorder_manager.record_pre_reset(None)
            env.recorder_manager.reset()
            success.fill_(False)
            env.recorder_manager.record_post_reset(None)
        finally:
            if explicit:
                recording.after_explicit_reset(env)

    def step(value, *, active, done=False):
        env.action.fill_(value)
        source_term.sample = TeleopSample(active=active, actions=torch.tensor([value]))
        env.recorder_manager.record_pre_step()
        env.recorder_manager.record_post_step()
        if done:
            reset(explicit=False)
        recording.after_step(env, torch.tensor([done]))

    try:
        reset(explicit=True)
        reset(explicit=True)
        assert env.recorder_manager.exported_failed_episode_count == 0
        step(1.0, active=True)
        step(2.0, active=False)
        reset(explicit=True)  # A nonempty operator interruption is an unsuccessful attempt.
        assert not recording.complete
        success.fill_(True)
        step(3.0, active=True, done=True)
        assert recording.complete == (mode == "all")
        reset(explicit=True)  # An immediate reset after auto-reset must not export its initial state.
        success.fill_(True)
        step(4.0, active=False, done=True)
        assert recording.complete
        step(99.0, active=True)  # An unfinished final attempt is deliberately dropped.
    finally:
        env.recorder_manager.close()

    with h5py.File(cfg.dataset_file) as dataset:
        episodes = list(dataset["data"].values())
        assert len(episodes) == (3 if mode == "all" else 2)
        assert [episode["actions"][:, 0].tolist() for episode in episodes] == (
            [[1.0, 2.0], [3.0], [4.0]] if mode == "all" else [[3.0], [4.0]]
        )
        assert [episode["metadata/teleop_active"][:, 0].tolist() for episode in episodes] == (
            [[True, False], [True], [False]] if mode == "all" else [[True], [False]]
        )
