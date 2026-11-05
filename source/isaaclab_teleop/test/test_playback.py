# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback ownership across task actions, episode resets, and the external XR renderer."""

import sys
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_teleop.action_cfg import TeleopActionCfg
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.observation_cfg import TeleopObservationCfg
from isaaclab_teleop.ovxr_cfg import OvLibrariesXrCameraCfg, OvLibrariesXrCfg
from isaaclab_teleop.play_source_cfg import CaptureTeleopSourceCfg
from isaaclab_teleop.playback import CapturePlaySession, PlaybackStopped, configured_teleop_actions
from isaaclab_teleop.source import TeleopSample, TeleopSource

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.managers import (
    ActionManager,
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.renderers import RendererCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.visualizers import VisualizerBackendCfg, VisualizerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


class _Controller(ActionTerm):
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
        self._actions.copy_(actions)

    def apply_actions(self):
        pass


class _Source(TeleopSource):
    def __init__(self, cfg, env, *, env_ids, xr_session):
        self.env = env
        env.source = self
        self.sample = TeleopSample(torch.tensor([0.5]))
        self.fail_close = False
        self.closed = False
        self.session_cleanup_failed = False
        self.device = SimpleNamespace(
            xr_render_anchor_path="/anchor", required_openxr_extensions=(), external_anchor_state=None
        )

    def start(self):
        self.env.events.append("source_open")

    def advance(self):
        self.env.events.append("input")
        return self.sample

    def reset(self, env_ids=None):
        self.env.events.append("input_reset")

    def close(self):
        if self.closed:
            return
        if self.fail_close:
            raise RuntimeError("input still in flight")
        self.closed = True
        self.env.events.append("source_close")


class _Env(gym.Env):
    def __init__(self, cfg, events):
        self.cfg, self.events = cfg, events
        self.num_envs, self.device = cfg.scene.num_envs, "cpu"
        self.scene = {"robot": object()}
        self.steps = 0
        self.step_dt = 0.01
        self.auto_reset = False
        self.closed = False
        self.sim = SimpleNamespace(
            is_playing=lambda: True,
            get_physics_step_count=lambda: self.steps,
            get_physics_dt=lambda: 0.01,
            vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda _: None),
        )
        self.action_manager = ActionManager(cfg.actions, self)
        self.obs_buf = {"policy": torch.ones(self.num_envs, 1)}
        self.action_space = gym.spaces.Box(-1, 1, (self.num_envs, 1))
        self.observation_space = gym.spaces.Dict(policy=self.action_space)
        events.append("env_open")

    def reset(self, env_ids=None, **kwargs):
        self.events.append("env_reset")
        self.action_manager.reset(env_ids)
        self.obs_buf["policy"][slice(None) if env_ids is None else env_ids] = 0
        return self.obs_buf, {}

    def step(self, actions):
        self.events.append("step")
        self.action_manager.process_action(actions)
        self.steps += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.auto_reset:
            self.reset(torch.tensor([0]))
            done[0] = True
        return self.obs_buf, torch.ones(self.num_envs), done, torch.zeros_like(done), {}

    def close(self):
        if not self.closed:
            self.action_manager.close()
            self.events.append("env_close")
            self.closed = True


def _config(*, xr=False, num_envs=1):
    cfg = ManagerBasedRLEnvCfg(
        scene=InteractiveSceneCfg(num_envs=num_envs, env_spacing=2),
        actions=SimpleNamespace(
            robot=TeleopActionCfg(
                controller=ActionTermCfg(class_type=_Controller, asset_name="robot"),
                source=CaptureTeleopSourceCfg(class_type=_Source, device=IsaacTeleopCfg(pipeline_builder=lambda: None)),
            )
        ),
        observations=SimpleNamespace(policy=ObservationGroupCfg()),
    )
    if xr:
        cfg.observations.operator = TeleopObservationCfg()
        cfg.sim.physics = object()
    return cfg


@pytest.fixture
def xr_runtime(monkeypatch):
    events, sessions = [], []
    context = SimpleNamespace(clear_instance=lambda: events.append("context_close"))
    monkeypatch.setattr(SimulationContext, "instance", lambda: context)

    class Xr:
        async_worker_owns_environment = False
        shutdown_requested = False

        def __init__(self):
            self.env = None
            self.admissions = []

        @classmethod
        def prepare(cls, *args, **kwargs):
            events.append("xr_prepare")
            result = cls()
            sessions.append(result)
            return result

        def start_render_worker(self, env, **kwargs):
            self.env = env
            self.async_worker_owns_environment = True
            events.append("xr_start")
            return True

        def request_render(self, generation, *, force):
            events.append(("publish", generation, force))
            return self.admissions.pop(0) if self.admissions else 1

        def report_host_timing(self, *args):
            pass

        def wait_for_shutdown(self, timeout):
            events.append("xr_wait")

        def request_shutdown(self):
            events.append("shutdown")
            self.shutdown_requested = True

        def close(self):
            events.append("xr_close")
            if self.async_worker_owns_environment:
                self.env.close()

    monkeypatch.setitem(sys.modules, "isaaclab_teleop.ov_libraries_xr", SimpleNamespace(OvLibrariesXrSession=Xr))
    return events, sessions


def test_operator_reset_is_an_interruption_without_a_physics_or_recorder_step():
    """A reset marks the selected episode boundary and preserves its pre-reset observation."""
    cfg, events = _config(num_envs=2), []
    assert list(configured_teleop_actions(cfg)) == ["robot"]
    session = CapturePlaySession(cfg)
    session.prepare()
    env = _Env(cfg, events)
    wrapped = session.attach(env)
    env.source.sample = TeleopSample(reset=True)
    obs, reward, terminated, truncated, extras = wrapped.step(torch.ones(2, 1))
    assert env.steps == 0 and "step" not in events
    assert events.count("input") == 1 and events.count("input_reset") == 1
    torch.testing.assert_close(obs["policy"], torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(extras["final_obs"]["policy"], torch.ones(2, 1))
    assert not terminated.any() and truncated.tolist() == [True, False]
    assert reward.tolist() == [0.0, 0.0] and extras["teleop_reset"].tolist() == [0]
    wrapped.close()
    session.close()
    assert events[-2:] == ["source_close", "env_close"]


def test_xr_prepares_before_environment_and_publishes_resets_before_next_input(xr_runtime):
    """The renderer and input remain separate owners, including deferred reset publication."""
    events, instances = xr_runtime
    cfg = _config(xr=True)
    session = CapturePlaySession(cfg)
    session.prepare()
    env = _Env(cfg, events)
    wrapped = session.attach(env)
    assert events[:4] == ["xr_prepare", "env_open", "source_open", "xr_start"]
    xr = instances[0]
    xr.admissions = [None, None, 12]
    wrapped.reset()
    wrapped.step(torch.zeros(1, 1))
    retry_index = events.index("xr_wait")
    assert events[retry_index + 1] == ("publish", (2, 0), True)
    assert events[retry_index + 2] == "input"
    assert events.count("input") == 1 and env.steps == 1
    env.auto_reset = True
    wrapped.step(torch.zeros(1, 1))
    assert events[-1] == ("publish", (3, 2), True)
    wrapped.close()
    session.close()
    assert events[-3:] == ["source_close", "xr_close", "env_close"]
    assert events.count("env_close") == 1


def test_failed_input_drain_keeps_xr_and_environment_alive_for_retry(xr_runtime):
    events, instances = xr_runtime
    cfg = _config(xr=True)
    session = CapturePlaySession(cfg)
    session.prepare()
    env = _Env(cfg, events)
    wrapped = session.attach(env)
    env.source.fail_close = True
    with pytest.raises(RuntimeError, match="input still in flight"):
        wrapped.close()
    assert not env.closed and "xr_close" not in events
    assert instances[0].shutdown_requested
    env.source.fail_close = False
    session.close()
    assert events[-3:] == ["source_close", "xr_close", "env_close"]


def test_prepared_xr_context_is_released_when_environment_construction_fails(xr_runtime):
    events, _ = xr_runtime
    session = CapturePlaySession(_config(xr=True))
    session.prepare()
    session.close()
    session.close()
    assert events == ["xr_prepare", "xr_close", "context_close"]


def test_desktop_failed_construction_closes_only_a_new_context(monkeypatch):
    """The playback owner cleans partial startup without destroying an embedding host's context."""
    closed = []
    existing = SimpleNamespace(clear_instance=lambda: closed.append("existing"))
    current = existing
    monkeypatch.setattr(SimulationContext, "instance", lambda: current)
    session = CapturePlaySession(_config())
    session.prepare()
    session.close()
    assert not closed
    session = CapturePlaySession(_config())
    session.prepare()
    current = SimpleNamespace(clear_instance=lambda: closed.append("new"))
    session.close()
    assert closed == ["new"]


def test_failed_input_start_retains_its_handle_until_cleanup_succeeds(xr_runtime, monkeypatch):
    """Partially started input may borrow the scene even before the XR worker starts."""
    events, _ = xr_runtime
    cfg = _config(xr=True)
    session = CapturePlaySession(cfg)
    session.prepare()
    env = _Env(cfg, events)

    def fail_start(source):
        source.fail_close = True
        raise RuntimeError("input startup failed")

    monkeypatch.setattr(_Source, "start", fail_start)
    with pytest.raises(RuntimeError, match="input startup failed"):
        session.attach(env)
    with pytest.raises(RuntimeError, match="input still in flight"):
        session.close()
    assert not env.closed and "xr_close" not in events
    env.source.fail_close = False
    session.close()
    assert events[-3:] == ["source_close", "xr_close", "env_close"]


def test_xr_shutdown_during_pending_reset_stops_without_polling_input(xr_runtime):
    events, instances = xr_runtime
    cfg = _config(xr=True)
    session = CapturePlaySession(cfg)
    session.prepare()
    env = _Env(cfg, events)
    wrapped = session.attach(env)
    instances[0].admissions = [None]
    wrapped.reset()
    instances[0].request_shutdown()
    with pytest.raises(PlaybackStopped):
        wrapped.step(torch.zeros(1, 1))
    assert "input" not in events and env.steps == 0
    session.close()


def test_xr_preserves_learning_cameras_and_rejects_shared_renderer_consumers():
    """No visual policy is silently changed to accommodate an operator view."""
    cfg = _config(xr=True)
    view = cfg.observations.operator
    cfg.scene.learning_camera = CameraCfg(
        prim_path="/World/learning", width=8, height=8, spawn=None, renderer_cfg=RendererCfg(class_type=object)
    )
    original = cfg.scene.learning_camera
    session = CapturePlaySession(cfg)
    session.validate()
    assert cfg.scene.learning_camera is original
    original.renderer_cfg = OVRTXRendererCfg()
    with pytest.raises(ValueError, match="Learning camera.*shares"):
        session.validate()
    original.renderer_cfg = RendererCfg(class_type=object)
    panel = CameraCfg(prim_path="/World/panel", width=8, height=8, spawn=None, renderer_cfg=view.renderer)
    cfg.scene.panel = panel
    view.presentation = OvLibrariesXrCfg(auxiliary_cameras=(OvLibrariesXrCameraCfg(name="panel", camera_cfg=panel),))
    cfg.observations.critic = ObservationGroupCfg()
    cfg.observations.critic.image = ObservationTermCfg(
        func=lambda env, nested: torch.zeros(1, 1), params={"nested": {"camera": [SceneEntityCfg("panel")]}}
    )
    with pytest.raises(ValueError, match="critic/image"):
        session.validate()
    cfg.observations.critic.image = None
    cfg.video_recorders = [VideoRecorderCfg(source="sensor:panel")]
    with pytest.raises(ValueError, match="Video recording"):
        session.validate()
    cfg.video_recorders = []
    session.validate()
    assert cfg.scene.learning_camera is original and cfg.scene.panel is panel
    cfg.sim.visualizer_cfgs = VisualizerCfg(backend=VisualizerBackendCfg(visualizer_type="newton_gl"))
    with pytest.raises(ValueError, match="local visualizers"):
        session.validate()
