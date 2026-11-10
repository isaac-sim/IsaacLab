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
from isaaclab_teleop.input_cfg import TeleopInputCfg
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.observation_cfg import TeleopObservationCfg
from isaaclab_teleop.ovxr_cfg import OvLibrariesXrCameraCfg, OvLibrariesXrCfg
from isaaclab_teleop.play_source_cfg import CaptureTeleopSourceCfg
from isaaclab_teleop.playback import (
    activate_capture,
    configured_teleop_inputs,
    prepare_playback_config,
    validate_playback,
)
from isaaclab_teleop.source import TeleopSample, TeleopSource
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg

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
from isaaclab.visualizers import ImageViewCfg, SceneCameraCfg, VisualizerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.ov]


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
    def __init__(self, cfg, env, *, env_ids, presentation, startup_timeout_s):
        self.env = env
        env.source = self
        self.presentation = presentation
        self.startup_timeout_s = startup_timeout_s
        self.sample = TeleopSample(torch.tensor([0.5]))
        self.fail_close = False
        self.closed = False

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
        self.scene = {"robot": object(), "other_robot": object()}
        self.closed = False
        self.sim = SimpleNamespace(
            is_playing=lambda: True,
            vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda _: None),
        )
        self.action_manager = ActionManager(cfg.actions, self)
        events.append("env_open")

    def close(self):
        if not self.closed:
            self.action_manager.close()
            self.events.append("env_close")
            self.closed = True


def _config(*, xr=False, num_envs=1):
    cfg = ManagerBasedRLEnvCfg(
        scene=InteractiveSceneCfg(num_envs=num_envs, env_spacing=2),
        actions=SimpleNamespace(
            robot=ActionTermCfg(
                class_type=_Controller,
                asset_name="robot",
                input=TeleopInputCfg(
                    source=CaptureTeleopSourceCfg(
                        class_type=_Source, device=IsaacTeleopCfg(pipeline_builder=lambda: None)
                    )
                ),
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
    """Isolate XR construction while exercising the real setup scope and its ownership transfer."""
    events, sessions = [], []
    current_context = None

    def clear_context():
        nonlocal current_context
        events.append("context_close")
        current_context = None

    monkeypatch.setattr(SimulationContext, "instance", lambda: current_context)

    class Xr:
        def __init__(self, environment_owner):
            self.environment_owner = environment_owner
            self.closed = False

        @classmethod
        def prepare(cls, *args, environment_owner, **kwargs):
            nonlocal current_context
            events.append("xr_prepare")
            current_context = SimpleNamespace(clear_instance=clear_context)
            result = cls(environment_owner)
            sessions.append(result)
            return result

        def close(self):
            if not self.closed:
                events.append("xr_close")
                self.closed = True

    monkeypatch.setitem(sys.modules, "isaaclab_teleop.ov_libraries_xr", SimpleNamespace(OvLibrariesXrSession=Xr))
    return events, sessions


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("different_row", "same environment row"),
        ("second_source", "exactly one.*owning input"),
    ],
)
def test_shared_input_rejects_multiple_sources_or_rows_before_resource_start(violation, message, monkeypatch):
    """A Capture session must have one source and one environment row across its controllers."""
    cfg = _config(num_envs=2)
    cfg.actions.gripper = ActionTermCfg(class_type=_Controller, asset_name="robot", input=TeleopInputCfg())
    if violation == "different_row":
        cfg.actions.gripper.input.env_ids = [1]
    else:
        cfg.actions.gripper.input.source = cfg.actions.robot.input.source.copy()
    monkeypatch.setattr(_Source, "start", lambda *args: pytest.fail("invalid configuration started Capture"))
    with pytest.raises(ValueError, match=message):
        validate_playback(cfg)


def test_activation_shares_one_sample_without_replacing_controllers_or_environment():
    """Task inputs use the ordinary ActionManager even when the source-owning controller comes last."""
    cfg, events = _config(num_envs=2), []
    owner_cfg = cfg.actions.robot
    cfg.actions = SimpleNamespace(
        follower=ActionTermCfg(class_type=_Controller, asset_name="other_robot", input=TeleopInputCfg()),
        robot=owner_cfg,
    )
    assert list(configured_teleop_inputs(cfg)) == ["follower", "robot"]
    validate_playback(cfg)
    with activate_capture(cfg) as activate:
        env = _Env(cfg, events)
        original_controllers = [env.action_manager.get_term(name) for name in ("follower", "robot")]
        assert activate(env) is None
        policy = torch.ones(2, 2)
        env.action_manager.process_action(policy)
        assert events.count("input") == 1
        torch.testing.assert_close(env.action_manager.action, torch.tensor([[0.5, 0.5], [1.0, 1.0]]))
        torch.testing.assert_close(policy, torch.ones(2, 2))
        assert [env.action_manager.get_term(name) for name in ("follower", "robot")] == original_controllers
        for controller in original_controllers:
            torch.testing.assert_close(controller.processed_actions, torch.tensor([[0.5], [1.0]]))
        env.source.sample = TeleopSample(active=False)
        env.action_manager.process_action(policy)
        torch.testing.assert_close(env.action_manager.action, policy)
        with pytest.raises(RuntimeError, match="only one environment"):
            activate(env)
    assert events[-2:] == ["source_close", "env_close"]


def test_xr_preparation_is_transferred_to_source_before_environment_teardown(xr_runtime):
    """Setup reserves XR before scene construction and gives the source the presentation owner."""
    events, sessions = xr_runtime
    cfg = _config(xr=True)
    cfg.observations.operator.startup_timeout_s = 123.0
    with activate_capture(cfg) as activate:
        assert events == ["xr_prepare"]
        env = _Env(cfg, events)
        activate(env)
        assert env.source.presentation.session is sessions[0]
        assert env.source.startup_timeout_s == 123.0
        assert sessions[0].environment_owner == "host"
        assert events == ["xr_prepare", "env_open", "source_open"]
    assert events[-3:] == ["source_close", "xr_close", "env_close"]


def test_failed_input_drain_retains_environment_for_retry():
    """Action input cleanup fails before controllers or the owning environment retire."""
    cfg, events = _config(), []
    with activate_capture(cfg) as activate:
        env = _Env(cfg, events)
        activate(env)
        env.source.fail_close = True
        with pytest.raises(RuntimeError, match="Failed to close 1 action input"):
            env.close()
        assert not env.closed and not env.source.closed
        env.source.fail_close = False
    assert events[-2:] == ["source_close", "env_close"]


def test_failed_controller_close_can_retry_after_other_action_terms_close(monkeypatch):
    """Partially retired action terms must not block a later environment cleanup attempt."""
    cfg, events = _config(), []
    cfg.actions.gripper = ActionTermCfg(class_type=_Controller, asset_name="robot", input=TeleopInputCfg())
    with activate_capture(cfg) as activate:
        env = _Env(cfg, events)
        activate(env)
        controller = env.action_manager.get_term("robot")
        close_controller = controller.close
        fail_close = True

        def close():
            if fail_close:
                raise RuntimeError("controller cleanup failed")
            close_controller()

        monkeypatch.setattr(controller, "close", close)
        with pytest.raises(RuntimeError, match="Failed to close 1 action term"):
            env.close()
        assert env.source.closed and not env.closed
        fail_close = False
    assert env.closed
    assert events.count("source_close") == events.count("env_close") == 1


def test_prepared_xr_context_is_released_when_environment_construction_fails(xr_runtime):
    events, _ = xr_runtime
    with pytest.raises(RuntimeError, match="construction failed"):
        with activate_capture(_config(xr=True)):
            raise RuntimeError("environment construction failed")
    assert events == ["xr_prepare", "xr_close", "context_close"]


def test_desktop_failed_construction_closes_only_a_new_context(monkeypatch):
    """The preparation scope cleans partial startup without destroying an embedding host's context."""
    closed = []
    existing = SimpleNamespace(clear_instance=lambda: closed.append("existing"))
    current = existing
    monkeypatch.setattr(SimulationContext, "instance", lambda: current)
    with activate_capture(_config()):
        pass
    assert not closed
    with activate_capture(_config()):
        current = SimpleNamespace(clear_instance=lambda: closed.append("new"))
    assert closed == ["new"]


def test_failed_input_start_preserves_error_and_live_dependencies(xr_runtime, monkeypatch):
    """Failed startup keeps borrowed scene/XR alive if its cleanup also fails."""
    events, sessions = xr_runtime
    cfg = _config(xr=True)

    def fail_start(source):
        source.fail_close = True
        raise RuntimeError("input startup failed")

    monkeypatch.setattr(_Source, "start", fail_start)
    with pytest.raises(RuntimeError, match="input startup failed") as caught:
        with activate_capture(cfg) as activate:
            env = _Env(cfg, events)
            activate(env)
    assert any("Failed to close 1 action input" in note for note in caught.value.__notes__)
    assert not env.closed and not sessions[0].closed
    env.source.fail_close = False
    env.action_manager.close_inputs()
    sessions[0].close()
    env.close()


def test_task_preparation_runs_before_renderer_selection():
    """The device's task preparation hook can install the operator view that launch must resolve."""
    cfg = _config()
    view = TeleopObservationCfg()
    cfg.actions.robot.input.source.device.prepare_environment = lambda env_cfg: setattr(
        env_cfg.observations, "operator", view
    )
    assert prepare_playback_config(cfg) == (cfg, view.renderer)


def test_xr_preserves_learning_cameras_and_rejects_shared_renderer_consumers():
    """No visual policy is silently changed to accommodate an operator view."""
    cfg = _config(xr=True)
    view = cfg.observations.operator
    cfg.scene.learning_camera = CameraCfg(
        prim_path="/World/learning", width=8, height=8, spawn=None, renderer_cfg=RendererCfg(class_type=object)
    )
    original = cfg.scene.learning_camera
    validate_playback(cfg)
    assert cfg.scene.learning_camera is original
    original.renderer_cfg = OVRTXRendererCfg()
    with pytest.raises(ValueError, match="Learning camera.*shares"):
        validate_playback(cfg)
    original.renderer_cfg = RendererCfg(class_type=object)
    panel = CameraCfg(prim_path="/World/panel", width=8, height=8, spawn=None, renderer_cfg=view.renderer)
    cfg.scene.panel = panel
    view.presentation = OvLibrariesXrCfg(auxiliary_cameras=(OvLibrariesXrCameraCfg(name="panel", camera_cfg=panel),))
    cfg.observations.critic = ObservationGroupCfg()
    cfg.observations.critic.image = ObservationTermCfg(
        func=lambda env, nested: torch.zeros(1, 1), params={"nested": {"camera": [SceneEntityCfg("panel")]}}
    )
    with pytest.raises(ValueError, match="critic/image"):
        validate_playback(cfg)
    cfg.observations.critic.image = None
    cfg.video_recorders = [VideoRecorderCfg(source="sensor:panel")]
    with pytest.raises(ValueError, match="Video recording"):
        validate_playback(cfg)
    cfg.video_recorders = []
    validate_playback(cfg)
    assert cfg.scene.learning_camera is original and cfg.scene.panel is panel
    cfg.sim.visualizer_cfgs = VisualizerCfg(backend=NewtonGLBackendCfg())
    validate_playback(cfg)


def test_xr_allows_independent_lab_views_and_rejects_consumers_of_leased_cameras():
    """A Lab viewer may use its own view while Capture alone renders XR-owned cameras."""
    cfg = _config(xr=True)
    view = cfg.observations.operator
    cfg.scene.learning_camera = CameraCfg(
        prim_path="/World/learning", width=8, height=8, spawn=None, renderer_cfg=RendererCfg(class_type=object)
    )
    cfg.sim.visualizer_cfgs = VisualizerCfg(backend=NewtonGLBackendCfg(), eye=(1.0, 2.0, 3.0))
    validate_playback(cfg)
    assert not view.presentation.desktop_mirror_enabled
    assert cfg.sim.visualizer_cfgs.eye == (1.0, 2.0, 3.0)

    cfg.sim.visualizer_cfgs = VisualizerCfg(backend=NewtonRTXBackendCfg())
    with pytest.raises(ValueError, match="perspective view shares"):
        validate_playback(cfg)
    cfg.sim.visualizer_cfgs.backend.renderer_cfg = OVRTXRendererCfg(log_file_path="/tmp/independent-viewer.log")
    validate_playback(cfg)

    # An RTX viewer borrowing an independent sensor never uses its default
    # perspective renderer, so that unused default must not reject the view.
    cfg.sim.visualizer_cfgs = VisualizerCfg(backend=NewtonRTXBackendCfg(), view=ImageViewCfg(source="learning_camera"))
    validate_playback(cfg)
    panel = CameraCfg(prim_path="{ENV_REGEX_NS}/HeadCamera", width=8, height=8, spawn=None, renderer_cfg=view.renderer)
    cfg.scene.panel = panel
    view.presentation = OvLibrariesXrCfg(auxiliary_cameras=(OvLibrariesXrCameraCfg(name="panel", camera_cfg=panel),))
    cfg.sim.visualizer_cfgs.view.source = "panel"
    with pytest.raises(ValueError, match="XR-owned panel cameras"):
        validate_playback(cfg)
    cfg.sim.visualizer_cfgs = VisualizerCfg(
        backend=NewtonGLBackendCfg(),
        cameras=[SceneCameraCfg(prim_path="/World/envs/env_0/HeadCamera")],
        streaming_view=True,
    )
    with pytest.raises(ValueError, match="XR-owned panel cameras"):
        validate_playback(cfg)
    cfg.sim.visualizer_cfgs.cameras = [SceneCameraCfg(prim_path="/World/learning")]
    validate_playback(cfg)
    cfg.sim.visualizer_cfgs.cameras = None
    with pytest.raises(ValueError, match="XR-owned panel cameras"):
        validate_playback(cfg)
    # Automatic selection only binds cameras that supply all requested channels.
    cfg.sim.visualizer_cfgs.streaming_gt_types = ("depth",)
    validate_playback(cfg)
