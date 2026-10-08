# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The single Cosmos path: ``--cosmos`` on train/play puts Cosmos on an unmodified camera task."""

import argparse
import logging
import math

import pytest
from isaaclab_experimental.cosmos import DEFAULT_MAX_EPISODE_FRAMES, apply_cosmos
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
    monkeypatch.setattr(cosmos_camera_module, "service_max_episode_frames", lambda endpoint: DEFAULT_MAX_EPISODE_FRAMES)
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

    with pytest.raises(ValueError, match="--num_envs 1"):
        apply_env_overrides(_cli(add_args, ["--cosmos", "--num_envs", "4"]), _kuka())


@pytest.mark.parametrize(
    "episode_length_s,dt,decimation",
    [(12.0, 1 / 120, 4), (8.8, 0.002, 2), (20.0, 0.005, 2)],
)
def test_a_fast_camera_captures_every_few_steps_and_never_exceeds_the_cap(episode_length_s, dt, decimation, caplog):
    """Whole environment steps between captures keep every episode within the cap, including rounding cases."""
    env_cfg = _kuka(episode_length_s=episode_length_s, decimation=decimation)
    env_cfg.sim.dt = dt
    step_dt = dt * decimation
    with caplog.at_level(logging.WARNING):
        apply_cosmos(env_cfg, prompt="A lab.", max_episode_frames=DEFAULT_MAX_EPISODE_FRAMES)
    period = env_cfg.scene.base_camera.update_period
    episode_steps = math.ceil(episode_length_s / step_dt - 1e-9)
    captures = math.ceil(episode_steps / round(period / step_dt)) + 1

    assert period / step_dt == pytest.approx(round(period / step_dt))
    assert captures <= _budget(env_cfg) <= DEFAULT_MAX_EPISODE_FRAMES and (_budget(env_cfg) - 1) % 4 == 0
    assert "captures every" in caplog.text


@pytest.mark.parametrize("cap", [601, None])
def test_a_higher_or_removed_episode_cap_keeps_the_camera_at_every_step(cap):
    env_cfg = _kuka()
    episode_steps = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation) - 1e-9)
    apply_cosmos(env_cfg, prompt="A lab.", max_episode_frames=cap)

    assert env_cfg.scene.base_camera.update_period == 0.0
    assert _budget(env_cfg) == 1 + 4 * math.ceil(episode_steps / 4)


def test_cosmos_rejects_an_invalid_cap_and_needs_one_rgb_camera_or_an_explicit_choice():
    with pytest.raises(ValueError, match="1 \\+ 4\\*k"):
        apply_cosmos(_kuka(), prompt="A lab.", max_episode_frames=200)
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
    show_run_summary(screen, _cli(add_common_train_args, []), _kuka(), library="rsl_rl", action="train")
    assert screen.fields["Environments"] != "1"
