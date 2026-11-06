# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback activation of task-configured input, without a simulator or device."""

from __future__ import annotations

import argparse
import contextlib
import sys
from types import SimpleNamespace

import gymnasium as gym
import h5py
import pytest
import torch
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_teleop import CaptureTeleopSourceCfg, TeleopActionCfg, TeleopSample, TeleopSource, se3_keyboard_teleop_cfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionManager,
    ActionTerm,
    ActionTermCfg,
    RecorderManager,
    TerminationManager,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_rl.entrypoints import common, dispatch, simple_agents
from isaaclab_rl.entrypoints.common import create_isaaclab_env, launch_playback, run_playback

from isaaclab_tasks.utils import preset, resolve_task_config

pytestmark = pytest.mark.kitless


class _InputSource(TeleopSource):
    """Supply one input sample, then release control to the baseline policy."""

    def __init__(self, cfg, env, *, env_ids, xr_session=None):
        self.polls = 0
        self.close_calls = 0
        self.closed = False
        self.session_cleanup_failed = False
        env.input_source = self

    def start(self):
        pass

    def advance(self):
        self.polls += 1
        return TeleopSample(actions=torch.tensor([4.0, -0.5]), active=self.polls == 1)

    def reset(self, env_ids=None):
        pass

    def close(self):
        self.close_calls += 1
        self.closed = True


class _EffortTerm(ActionTerm):
    """Minimal controller retaining the real ActionManager arbitration path."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros(env.num_envs, 1)

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions[:] = actions

    def apply_actions(self):
        self._asset.applied.append(self._actions.clone())


@configclass
class _ActionsCfg:
    gripper = TeleopActionCfg(
        controller=ActionTermCfg(class_type=_EffortTerm, asset_name="other_robot"),
        env_ids=[1],
        action_adapter=lambda values, env, env_ids: values[1:],
    )
    controlled = TeleopActionCfg(
        controller=ActionTermCfg(class_type=_EffortTerm, asset_name="robot"),
        source=CaptureTeleopSourceCfg(class_type=_InputSource, device=se3_keyboard_teleop_cfg()),
        env_ids=[1],
        action_adapter=lambda values, env, env_ids: values[:1],
    )
    policy = ActionTermCfg(class_type=_EffortTerm, asset_name="robot")


@configclass
class _PlaybackCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(device="cpu", physics=NewtonCfg())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3, env_spacing=2.0)
    actions: _ActionsCfg = _ActionsCfg()
    observations: dict = {}
    rewards: dict = {}
    terminations: dict = {}
    decimation: int = 1
    episode_length_s: float = 1.0

    def play_mode(self):
        pass


class _SinkScene(dict):
    """Small physical-state boundary used by the real action/state recorders."""

    def __init__(self, num_envs, action_dim):
        super().__init__(robot=SimpleNamespace(applied=[]), other_robot=SimpleNamespace(applied=[]))
        self.position = torch.zeros(num_envs, action_dim)

    def get_state(self, *, is_relative=False):
        return {
            "articulation": {
                "other_robot": {"joint_position": self.position[:, :1]},
                "robot": {"joint_position": self.position[:, 1:]},
            }
        }


class _ActionSink(gym.Env):
    """Exercise real action, termination, and recording managers around simple state updates."""

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs
        self.device = "cpu"
        self.sim = SimpleNamespace(
            is_running=lambda: True,
            is_playing=lambda: True,
            vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None),
        )
        self.scene = _SinkScene(self.num_envs, 0)
        self.action_manager = ActionManager(cfg.actions, self)
        action_dim = self.action_manager.total_action_dim
        self.scene.position = torch.zeros(self.num_envs, action_dim)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.termination_manager = TerminationManager(cfg.terminations, self)
        self.recorder_manager = RecorderManager(cfg.recorders, self)
        self.step_dt = 0.01
        self.single_action_space = gym.spaces.Box(-10.0, 10.0, shape=(action_dim,))
        self.action_spaces = {"robot": self.single_action_space}
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(self.num_envs, action_dim))
        self.observation_space = self.action_space
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.applied = []
        self.obs_buf = {"policy": self.scene.position.clone()}
        self.extras = {}
        self.closed = False

    def reset(self, **kwargs):
        env_ids = kwargs.get("env_ids", slice(None))
        self.recorder_manager.record_pre_reset(env_ids)
        self.scene.position[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.action_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)
        self.recorder_manager.reset(env_ids)
        self.obs_buf = {"policy": self.scene.position.clone()}
        self.recorder_manager.record_post_reset(env_ids)
        return self.obs_buf, self.extras

    def step(self, action):
        self.action_manager.process_action(action)
        self.applied.append(self.action_manager.action.clone())
        self.recorder_manager.record_pre_step()
        self.action_manager.apply_action()
        self.scene.position += self.action_manager.action
        self.episode_length_buf += 1
        self.reset_buf = self.termination_manager.compute()
        self.obs_buf = {"policy": self.scene.position.clone()}
        self.recorder_manager.record_post_step()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids):
            self.reset(env_ids=reset_env_ids)
        return (
            self.obs_buf,
            torch.zeros(self.num_envs),
            self.termination_manager.terminated,
            self.termination_manager.time_outs,
            self.extras,
        )

    def close(self):
        if self.closed:
            return
        self.action_manager.close()
        self.recorder_manager.close()
        self.closed = True


@pytest.fixture
def playback_task(monkeypatch):
    task = "Isaac-Test-Configured-Teleop"
    created = []

    def make_env(cfg, **kwargs):
        env = _ActionSink(cfg)
        created.append(env)
        return env

    gym.register(
        task,
        entry_point=make_env,
        kwargs={"env_cfg_entry_point": _PlaybackCfg, "default_agent": "rsl_rl"},
        disable_env_checker=True,
        order_enforce=False,
    )
    # Launcher imports/runtime are the external boundary; config resolution, dispatch, Gym creation,
    # action selection, source lifecycle, and the bounded playback loop are real.
    monkeypatch.setattr(simple_agents, "add_launcher_args", lambda parser: None)
    monkeypatch.setattr(common, "launch_simulation", lambda cfg, args, **kwargs: contextlib.nullcontext())
    try:
        yield task, created
    finally:
        gym.registry.pop(task, None)


def test_play_uses_configured_input_without_loading_a_policy(playback_task, monkeypatch):
    """Implicit playback routes shared input to distinct assets, then yields both to the baseline policy."""
    task, created = playback_task
    original_argv = sys.argv
    monkeypatch.setattr(dispatch, "_run_backend", lambda *args: pytest.fail("attempted to load a policy checkpoint"))

    assert dispatch.run_play_cli(["--task", task, "--max_steps", "2", "--seed", "7", "env.scene.num_envs=2"]) == 0

    env = created[0]
    assert env.num_envs == 2
    assert env.cfg.seed == 7
    assert torch.equal(env.applied[0], torch.tensor([[0.0, 0.0, 0.0], [-0.5, 4.0, 0.0]]))
    assert torch.equal(env.applied[1], torch.zeros(2, 3))
    torch.testing.assert_close(env.scene["other_robot"].applied[0], torch.tensor([[0.0], [-0.5]]))
    torch.testing.assert_close(env.scene["robot"].applied[0], torch.tensor([[0.0], [4.0]]))
    torch.testing.assert_close(env.scene["other_robot"].applied[1], torch.zeros(2, 1))
    torch.testing.assert_close(torch.stack(env.scene["robot"].applied[1:]), torch.zeros(3, 2, 1))
    assert env.input_source.polls == 2
    assert env.input_source.close_calls == 1
    assert env.input_source.closed and env.closed
    assert sys.argv is original_argv


def _success_after_two_steps(env):
    return env.episode_length_buf >= 2


def test_play_cli_exports_requested_demos_with_applied_actions_and_provenance(playback_task, monkeypatch, tmp_path):
    """Collection flags reach the real recorder and stop after completed demos, without a policy backend."""
    task, created = playback_task
    cfg = _PlaybackCfg()
    cfg.actions.controlled.env_ids = [0]
    cfg.actions.gripper.env_ids = [0]
    cfg.terminations = {"success": TerminationTermCfg(func=_success_after_two_steps)}
    gym.spec(task).kwargs["env_cfg_entry_point"] = lambda: cfg
    monkeypatch.setattr(dispatch, "_run_backend", lambda *args: pytest.fail("collection loaded a policy backend"))
    dataset = tmp_path / "demonstrations.hdf5"

    assert (
        dispatch.run_play_cli(
            ["--task", task, "--num_envs", "1", "--dataset_file", str(dataset), "--num_demos", "2", "--max_steps", "20"]
        )
        == 0
    )

    env = created[0]
    assert len(env.applied) == 4
    assert env.input_source.polls == 4
    assert env.input_source.close_calls == 1
    assert env.closed
    with h5py.File(dataset, "r") as output:
        assert list(output["data"]) == ["demo_0", "demo_1"]
        first = output["data/demo_0"]
        second = output["data/demo_1"]
        torch.testing.assert_close(
            torch.from_numpy(first["actions"][:]), torch.tensor([[-0.5, 4.0, 0.0], [0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(torch.from_numpy(second["actions"][:]), torch.zeros(2, 3))
        assert first["metadata/teleop_active"][:].tolist() == [[True], [False]]
        assert second["metadata/teleop_active"][:].tolist() == [[False], [False]]
        assert bool(first.attrs["success"]) and bool(second.attrs["success"])
        torch.testing.assert_close(torch.from_numpy(first["obs/policy"][0]), torch.zeros(3))
        torch.testing.assert_close(
            torch.from_numpy(first["states/articulation/robot/joint_position"][:]),
            torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
        )
        torch.testing.assert_close(
            torch.from_numpy(first["states/articulation/other_robot/joint_position"][:]),
            torch.tensor([[-0.5], [-0.5]]),
        )


@pytest.mark.parametrize("activation", ["unconfigured", "disabled"])
def test_collection_rejects_missing_capture_before_launch(playback_task, monkeypatch, tmp_path, activation):
    """A dataset request cannot silently become policy-only playback with no collection session."""
    task, created = playback_task
    cfg = _PlaybackCfg()
    if activation == "unconfigured":
        cfg.actions = {"policy": _ActionsCfg().policy}
    args = argparse.Namespace(
        task=task,
        frontend="torch",
        teleop=False if activation == "disabled" else None,
        dataset_file=str(tmp_path / "demonstrations.hdf5"),
    )
    monkeypatch.setattr(
        common, "launch_simulation", lambda *args, **kwargs: pytest.fail("attempted to launch simulation")
    )

    with pytest.raises(ValueError, match="--dataset_file requires playback with task-configured Capture input enabled"):
        with launch_playback(cfg, args):
            pytest.fail("playback started without a collection source")

    assert not created


def test_zero_agent_seeds_before_config_construction_without_reseeding(playback_task, monkeypatch):
    """Legacy playback keeps config-time random draws in the same seeded stream as runtime draws."""
    task, _ = playback_task
    draws = []

    def config_factory():
        draws.append(torch.rand(()))
        return _PlaybackCfg()

    def launch(cfg, args, **kwargs):
        draws.append(torch.rand(()))
        return contextlib.nullcontext()

    gym.spec(task).kwargs["env_cfg_entry_point"] = config_factory
    monkeypatch.setattr(common, "launch_simulation", launch)
    expected = torch.rand(2, generator=torch.Generator().manual_seed(42))
    previous_state = torch.get_rng_state()
    try:
        torch.set_rng_state(torch.Generator().manual_seed(123).get_state())
        assert dispatch.run_zero_agent_cli(["--task", task, "--max_steps", "0"]) == 0
    finally:
        torch.set_rng_state(previous_state)

    assert torch.equal(torch.stack(draws), expected)


def test_play_rejects_kit_task_before_launch(playback_task, monkeypatch):
    """Kit requirements declared by the task are rejected before a runtime or device starts."""
    task, created = playback_task
    cfg = _PlaybackCfg()
    cfg.sim.physics = PhysxCfg()
    gym.spec(task).kwargs["env_cfg_entry_point"] = lambda: cfg
    monkeypatch.setattr(common, "launch_simulation", lambda *args, **kwargs: pytest.fail("attempted to launch Kit"))

    with pytest.raises(ValueError, match="requires a kitless.*--no-teleop"):
        dispatch.run_play_cli(["--task", task, "--max_steps", "1"])

    assert not created


def test_play_rejects_warp_frontend_before_launch(playback_task, monkeypatch):
    """Input action tensors require the Torch frontend; reject an unsupported conversion before launching."""
    task, created = playback_task
    monkeypatch.setattr(
        common, "launch_simulation", lambda *args, **kwargs: pytest.fail("attempted to launch simulation")
    )

    with pytest.raises(ValueError, match="--frontend torch"):
        dispatch.run_play_cli(["--task", task, "--frontend", "warp", "--max_steps", "1"])

    assert not created


@pytest.mark.parametrize("selection", [["--no-teleop"], ["--rl_library", "rsl_rl"], ["--checkpoint", "policy.pt"]])
def test_policy_selection_preserves_backend_dispatch(playback_task, monkeypatch, selection):
    """Explicit policy options retain the existing backend route even with configured input."""
    task, created = playback_task
    dispatched = []
    monkeypatch.setattr(dispatch, "_run_backend", lambda *args: dispatched.append(args))

    assert dispatch.run_play_cli(["--task", task, *selection]) == 0

    assert len(dispatched) == 1
    assert not created


@pytest.mark.parametrize(
    ("override", "teleop_enabled", "expected"),
    [
        ("presets=small", False, {"network": {"width": 32}}),
        ("+agent.extra=8", True, {"network": {"width": 64}, "extra": 8}),
    ],
)
def test_implicit_policy_playback_preserves_agent_configuration(
    playback_task, monkeypatch, override, teleop_enabled, expected
):
    """Teleop discovery must not reject agent presets or consume Hydra agent additions."""
    task, created = playback_task
    gym.spec(task).kwargs["env_cfg_entry_point"] = lambda: (
        _PlaybackCfg() if teleop_enabled else _PlaybackCfg(actions={"policy": _ActionsCfg().policy})
    )
    gym.spec(task).kwargs["rsl_rl_cfg_entry_point"] = lambda: {
        "network": preset(default={"width": 64}, small={"width": 32})
    }
    resolved = []

    def run_backend(module, argv):
        _, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=argv[-1:])
        resolved.append(agent_cfg)

    monkeypatch.setattr(dispatch, "_run_backend", run_backend)
    assert dispatch.run_play_cli(["--task", task, "--max_steps", "0", override]) == 0
    assert resolved == [expected]
    assert not created


@pytest.mark.parametrize("activation", ["training", "disabled", "policy"])
def test_env_factory_activates_input_only_for_enabled_playback(playback_task, activation):
    """Training and --no-teleop retain policy actions; interactive playback activates task arbitration."""
    task, created = playback_task
    args = argparse.Namespace(frontend="torch", teleop=False if activation == "disabled" else None)
    cfg = _PlaybackCfg()
    with launch_playback(cfg, args, enable_teleop=activation != "training") as session:
        env = create_isaaclab_env(task, cfg, args, convert_marl_to_single_agent=False, playback_session=session)
        try:
            env.reset()
            policy_action = torch.ones(3, 3)
            env.step(policy_action)
        finally:
            env.close()

    sink = created[0]
    expected = torch.ones(3, 3)
    if activation == "policy":
        expected[1, 0] = -0.5
        expected[1, 1] = 4.0
        assert sink.input_source.closed
    else:
        assert not hasattr(sink, "input_source")
    assert torch.equal(sink.applied[0], expected)
    assert torch.equal(policy_action, torch.ones(3, 3))
    assert sink.closed


@pytest.mark.parametrize("max_steps", [0, 2])
def test_policy_playback_respects_step_budget(max_steps):
    """--max_steps stops the shared policy loop even without a video recorder."""
    steps = []
    args = argparse.Namespace(max_steps=max_steps, video=False, real_time=False)
    run_playback(lambda: steps.append(1), dt=0.01, args_cli=args, env_cfg=_PlaybackCfg())
    assert len(steps) == max_steps
