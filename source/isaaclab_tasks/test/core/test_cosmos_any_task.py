# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The single Cosmos path: ``--cosmos`` on train/play puts Cosmos on an unmodified camera task."""

import argparse
import logging

import pytest
from isaaclab_experimental.cosmos import apply_cosmos
from isaaclab_experimental.cosmos.client import camera as cosmos_camera_module

from isaaclab_rl.entrypoints.common import (
    add_common_play_args,
    add_common_train_args,
    apply_env_overrides,
    show_run_summary,
)

from isaaclab_tasks.utils import parse_env_cfg

KUKA_TASK = "Isaac-Reorient-KukaAllegro-Camera"
KUKA_PRESETS = "presets=cube,single_camera,newton_mjwarp,newton_renderer,rgb64"


def _kuka(**overrides):
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_PRESETS,))
    for name, value in overrides.items():
        setattr(env_cfg, name, value)
    return env_cfg


def _budget(env_cfg):
    return env_cfg.scene.base_camera.modifiers["distance_to_image_plane"][1].backend.max_episode_frames


def _cli(add_args, argv):
    parser = argparse.ArgumentParser()
    add_args(parser, agent_default=None, agent_help="")
    return parser.parse_args(argv)


@pytest.mark.parametrize("add_args", [add_common_train_args, add_common_play_args])
def test_cosmos_options_put_cosmos_on_the_policy_camera_of_an_unmodified_task(add_args, monkeypatch):
    """The Kuka Allegro camera task gets Cosmos from command-line options, with its 64 x 64 observation kept."""
    _serve(monkeypatch)
    args = _cli(add_args, ["--cosmos", "--cosmos_prompt", "A lab.", "--cosmos_prompt", "A kitchen."])
    env_cfg = _kuka()
    apply_env_overrides(args, env_cfg)
    env_cfg.validate()
    camera = env_cfg.scene.base_camera
    chain = camera.modifiers["distance_to_image_plane"]

    assert env_cfg.scene.num_envs == 1
    assert (camera.width, camera.height) == (640, 640)
    assert chain[1].backend.prompt == ["A lab.", "A kitchen."]
    assert chain[-1].params == {"width": 64, "height": 64}


def _serve(monkeypatch, partial_resets=False):
    """Stand in for the running service's status."""
    capabilities = {
        "max_episode_frames": None,
        "partial_resets": partial_resets,
    }
    monkeypatch.setattr(cosmos_camera_module, "service_capabilities", lambda endpoint, **kwargs: capabilities)


@pytest.mark.parametrize("add_args,enable", [(add_common_train_args, []), (add_common_play_args, ["--cosmos"])])
def test_cosmos_preset_accepts_cli_overrides_without_replacing_its_chain(add_args, enable, monkeypatch, caplog):
    """Train overrides the preset directly; play also accepts --cosmos without applying the modifier twice."""
    _serve(monkeypatch)
    env_cfg = parse_env_cfg("Isaac-Reorient-Cube-Shadow-Camera-Direct", overrides=("presets=cosmos",))
    args = _cli(
        add_args,
        [
            *enable,
            "--cosmos_prompt",
            "A kitchen.",
            "--cosmos_endpoint",
            "tcp://127.0.0.1:5556",
            "--cosmos_near",
            "0.2",
            "--cosmos_transport",
            "socket",
        ],
    )
    with caplog.at_level(logging.INFO):
        apply_env_overrides(args, env_cfg)
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    chain = camera.modifiers["distance_to_image_plane"]
    assert len(chain) == 2
    assert chain[0].params == {"near": 0.2, "far": 1.5}
    assert chain[1].backend.prompt == "A kitchen."
    assert chain[1].backend.endpoint == "tcp://127.0.0.1:5556"
    assert chain[1].backend.transport == "socket"
    assert chain[1].backend.max_episode_frames == 601
    assert camera.update_period == 0.0
    assert env_cfg.feature_extractor.image_update_frames == 4
    assert "60.000 captures/s, 15.000 generated updates/s" in caplog.text


def test_cosmos_preset_keeps_defaults_and_rejects_changing_its_control(monkeypatch):
    _serve(monkeypatch)
    env_cfg = parse_env_cfg("Isaac-Reorient-Cube-Shadow-Camera-Direct", overrides=("presets=cosmos",))
    original = env_cfg.scene.tiled_camera.modifiers["distance_to_image_plane"][-1].backend
    apply_env_overrides(_cli(add_common_train_args, []), env_cfg)
    backend = env_cfg.scene.tiled_camera.modifiers["distance_to_image_plane"][-1].backend
    assert backend.prompt == original.prompt and backend.endpoint == original.endpoint
    assert backend.transport == original.transport
    with pytest.raises(ValueError, match="preset prepares depth"):
        apply_env_overrides(_cli(add_common_train_args, ["--cosmos_control", "edge"]), env_cfg)


def test_cosmos_options_without_a_cosmos_camera_fail_instead_of_being_ignored():
    with pytest.raises(ValueError, match="require --cosmos"):
        apply_env_overrides(_cli(add_common_train_args, ["--cosmos_prompt", "A lab."]), _kuka())


def test_several_environments_need_a_compiled_service_that_batches_them(monkeypatch):
    """--num_envs selects the batch without a server view limit; independent resets still need compiled inference."""
    args = _cli(add_common_train_args, ["--cosmos", "--num_envs", "4"])
    _serve(monkeypatch, partial_resets=False)
    with pytest.raises(ValueError, match="without --no-compile"):
        apply_env_overrides(args, _kuka())

    _serve(monkeypatch, partial_resets=True)
    env_cfg = _kuka()
    apply_env_overrides(args, env_cfg)
    assert env_cfg.scene.num_envs == 4
    assert env_cfg.scene.base_camera.modifiers["distance_to_image_plane"][1].backend.max_episode_frames > 1


def test_the_shadow_hand_preset_batches_several_environments_at_its_camera_rate(monkeypatch):
    """--num_envs on the preset keeps its camera rate; each environment's budget counts its own captures."""
    _serve(monkeypatch, partial_resets=True)
    args = _cli(add_common_train_args, ["--num_envs", "2"])
    env_cfg = parse_env_cfg(
        "Isaac-Reorient-Cube-Shadow-Camera-Direct",
        overrides=("presets=cosmos", "env.scene.tiled_camera.update_period=0.1"),
    )
    apply_env_overrides(args, env_cfg)
    env_cfg.validate()

    assert env_cfg.scene.num_envs == 2
    assert env_cfg.scene.tiled_camera.update_period == 0.1
    assert env_cfg.scene.tiled_camera.modifiers["distance_to_image_plane"][-1].backend.max_episode_frames == 101


def test_an_explicit_cap_rejects_the_task_without_slowing_its_camera():
    env_cfg = _kuka()
    with pytest.raises(ValueError, match="needs a Cosmos budget of 361 frames.*cap is 201"):
        apply_cosmos(env_cfg, prompt="A lab.", max_episode_frames=201)
    assert env_cfg.scene.base_camera.update_period == 0.0
    assert not env_cfg.scene.base_camera.modifiers


def test_the_capture_count_uses_the_sensor_update_tolerance():
    """A sensor updates 1e-6 s early, so a 0.0200001 s period at 0.01 s steps captures every 2 steps, not 3."""
    env_cfg = _kuka(episode_length_s=4.0, decimation=2)
    env_cfg.sim.dt = 0.005
    env_cfg.scene.base_camera.update_period = 0.0200001
    apply_cosmos(env_cfg, prompt="A lab.", max_episode_frames=None)

    assert env_cfg.scene.base_camera.update_period == 0.0200001
    assert _budget(env_cfg) == 201  # 400 steps: 201 captures


@pytest.mark.parametrize("cap", [601, None])
def test_a_higher_or_removed_episode_cap_keeps_the_camera_at_every_step(cap):
    env_cfg = _kuka()
    apply_cosmos(env_cfg, prompt="A lab.", max_episode_frames=cap)

    assert env_cfg.scene.base_camera.update_period == 0.0
    assert _budget(env_cfg) == 361  # 12 s at 1/30 s steps: 360 steps and the initial frame


def test_cosmos_rejects_an_invalid_cap_and_needs_one_rgb_camera_or_an_explicit_choice():
    for cap in (1, 200):
        with pytest.raises(ValueError, match="1 \\+ 4\\*k"):
            apply_cosmos(_kuka(), prompt="A lab.", max_episode_frames=cap)
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=("presets=cube,duo_camera,newton_mjwarp,newton_renderer,rgb64",))
    with pytest.raises(ValueError, match="--cosmos_camera"):
        apply_cosmos(env_cfg, prompt="A lab.", max_episode_frames=None)
    apply_cosmos(env_cfg, prompt="A lab.", camera="wrist_camera", max_episode_frames=None)
    assert env_cfg.scene.wrist_camera.modifiers and not env_cfg.scene.base_camera.modifiers


def test_direct_tasks_that_size_observations_from_the_camera_are_warned(caplog):
    """Cartpole camera derives its observation space from the camera size, which Cosmos changes to the canvas."""
    env_cfg = parse_env_cfg("Isaac-Cartpole-Camera-Direct")
    with caplog.at_level(logging.WARNING):
        apply_cosmos(env_cfg, prompt="A cart.", max_episode_frames=None)
    assert "Direct tasks" in caplog.text


def test_the_run_summary_shows_the_single_cosmos_environment():
    """The summary appears before the overrides; with --cosmos it reports the one environment the run uses."""

    class Screen:
        def summary(self, title, fields):
            self.fields = fields

    screen = Screen()
    show_run_summary(screen, _cli(add_common_train_args, ["--cosmos"]), _kuka(), library="rsl_rl", action="train")
    assert screen.fields["Environments"] == "1"
    args = _cli(add_common_train_args, ["--cosmos", "--num_envs", "4"])
    show_run_summary(screen, args, _kuka(), library="rsl_rl", action="train")
    assert screen.fields["Environments"] == "4"
    show_run_summary(screen, _cli(add_common_train_args, []), _kuka(), library="rsl_rl", action="train")
    assert screen.fields["Environments"] != "1"
