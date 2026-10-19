# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos on camera tasks: ``apply_cosmos`` for an unmodified task, and the Shadow Hand preset with its overrides."""

import logging

import pytest
from isaaclab_experimental.cosmos import apply_cosmos
from isaaclab_experimental.cosmos.client import camera as cosmos_camera_module

from isaaclab_tasks.utils import parse_env_cfg

KUKA_TASK = "Isaac-Reorient-KukaAllegro-Camera"
KUKA_PRESETS = "presets=cube,single_camera,newton_mjwarp,newton_renderer,rgb64"
SHADOW_HAND_TASK = "Isaac-Reorient-Cube-Shadow-Camera-Direct"
SHADOW_HAND_CHAIN = "env.scene.tiled_camera.modifiers.distance_to_image_plane"


def _kuka(**overrides):
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_PRESETS,))
    for name, value in overrides.items():
        setattr(env_cfg, name, value)
    return env_cfg


def _budget(env_cfg):
    return env_cfg.scene.base_camera.modifiers["distance_to_image_plane"][1].backend.max_episode_frames


def _serve(monkeypatch, partial_resets=False):
    """Stand in for the running service's status."""
    capabilities = {"partial_resets": partial_resets}
    monkeypatch.setattr(cosmos_camera_module, "service_capabilities", lambda endpoint, **kwargs: capabilities)


@pytest.fixture(autouse=True)
def _idle_service(monkeypatch):
    """Every test sees a running, idle service; tests that batch environments call ``_serve`` again."""
    _serve(monkeypatch)


def test_apply_cosmos_puts_cosmos_on_the_policy_camera_of_an_unmodified_task():
    """The Kuka Allegro camera task gets Cosmos with its 64 x 64 observation kept."""
    env_cfg = _kuka()
    apply_cosmos(env_cfg, prompt=["A lab.", "A kitchen."])
    env_cfg.validate()
    camera = env_cfg.scene.base_camera
    chain = camera.modifiers["distance_to_image_plane"]

    assert env_cfg.scene.num_envs == 1
    assert (camera.width, camera.height) == (640, 640)
    assert chain[1].backend.prompt == ["A lab.", "A kitchen."]
    assert chain[-1].params == {"width": 64, "height": 64}


def test_the_shadow_hand_preset_options_are_set_with_overrides():
    """``env.cosmos`` overrides set the camera's own Cosmos settings; the depth range is set on its chain."""
    prompts = "['A kitchen.', 'A lab.']"
    env_cfg = parse_env_cfg(
        SHADOW_HAND_TASK,
        overrides=(
            "presets=cosmos",
            "env.scene.num_envs=2",
            f"env.cosmos.prompt={prompts}",
            "env.cosmos.endpoint=tcp://127.0.0.1:5556",
            "env.cosmos.transport=socket",
            f"{SHADOW_HAND_CHAIN}.0.params.near=0.2",
        ),
    )
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    chain = camera.modifiers["distance_to_image_plane"]
    assert len(chain) == 2
    assert chain[0].params == {"near": 0.2, "far": 1.5}
    assert chain[1].backend.prompt == ["A kitchen.", "A lab."]
    assert chain[1].backend.endpoint == "tcp://127.0.0.1:5556"
    assert chain[1].backend.transport == "socket"
    assert chain[1].backend.max_episode_frames == 601
    assert env_cfg.cosmos is chain[1].backend
    assert camera.update_period == 0.0
    assert env_cfg.feature_extractor.image_update_frames == 4


def test_several_environments_need_a_compiled_service_that_batches_them(monkeypatch):
    """The batch size is the environment count; independent resets still need compiled inference."""
    _serve(monkeypatch, partial_resets=False)
    with pytest.raises(ValueError, match="without --no-compile"):
        apply_cosmos(_kuka(), num_envs=4)

    _serve(monkeypatch, partial_resets=True)
    env_cfg = _kuka()
    apply_cosmos(env_cfg, num_envs=4)
    assert env_cfg.scene.num_envs == 4
    assert env_cfg.scene.base_camera.modifiers["distance_to_image_plane"][1].backend.max_episode_frames > 1


def test_the_shadow_hand_preset_batches_several_environments_at_its_camera_rate():
    """Several environments on the preset keep its camera rate; each environment's budget counts its own captures."""
    env_cfg = parse_env_cfg(
        SHADOW_HAND_TASK,
        overrides=("presets=cosmos", "env.scene.num_envs=2", "env.scene.tiled_camera.update_period=0.1"),
    )
    env_cfg.validate()

    assert env_cfg.scene.num_envs == 2
    assert env_cfg.scene.tiled_camera.update_period == 0.1
    assert env_cfg.scene.tiled_camera.modifiers["distance_to_image_plane"][-1].backend.max_episode_frames == 101


def test_the_budget_counts_every_capture_and_keeps_the_camera_at_every_step():
    env_cfg = _kuka()
    apply_cosmos(env_cfg, prompt="A lab.")

    assert env_cfg.scene.base_camera.update_period == 0.0
    assert _budget(env_cfg) == 361  # 12 s at 1/30 s steps: 360 steps and the initial frame


def test_the_capture_count_uses_the_sensor_update_tolerance():
    """A sensor updates 1e-6 s early, so a 0.0200001 s period at 0.01 s steps captures every 2 steps, not 3."""
    env_cfg = _kuka(episode_length_s=3.0, decimation=2)
    env_cfg.sim.dt = 0.005
    env_cfg.scene.base_camera.update_period = 0.0200001
    apply_cosmos(env_cfg, prompt="A lab.")

    assert env_cfg.scene.base_camera.update_period == 0.0200001
    assert _budget(env_cfg) == 153  # 300 steps: 150 captures and the initial one, rounded up to 1 + 4*k


def test_cosmos_needs_one_rgb_camera_or_an_explicit_choice():
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=("presets=cube,duo_camera,newton_mjwarp,newton_renderer,rgb64",))
    with pytest.raises(ValueError, match="`camera` argument"):
        apply_cosmos(env_cfg, prompt="A lab.")
    apply_cosmos(env_cfg, prompt="A lab.", camera="wrist_camera")
    assert env_cfg.scene.wrist_camera.modifiers and not env_cfg.scene.base_camera.modifiers


def test_direct_tasks_that_size_observations_from_the_camera_are_warned(caplog):
    """Cartpole camera derives its observation space from the camera size, which Cosmos changes to the canvas."""
    env_cfg = parse_env_cfg("Isaac-Cartpole-Camera-Direct")
    with caplog.at_level(logging.WARNING):
        apply_cosmos(env_cfg, prompt="A cart.")
    assert "Direct tasks" in caplog.text


def test_separate_shadow_hand_configs_do_not_share_cosmos_settings():
    first = parse_env_cfg(SHADOW_HAND_TASK, overrides=("presets=cosmos", "env.cosmos.prompt=A kitchen."))
    second = parse_env_cfg(SHADOW_HAND_TASK, overrides=("presets=cosmos",))
    assert first.cosmos.prompt == "A kitchen." and second.cosmos.prompt != "A kitchen."


def test_a_subclass_of_the_shadow_hand_preset_keeps_env_cosmos_on_its_camera():
    """Deriving a configuration from the preset copies its fields; ``env.cosmos`` still names the camera's settings."""
    from isaaclab.utils import configclass

    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import (
        ShadowHandCameraCosmosEnvCfg,
    )

    @configclass
    class ShorterEpisodeCfg(ShadowHandCameraCosmosEnvCfg):
        episode_length_s: float = 5.0

    env_cfg = ShorterEpisodeCfg()
    assert env_cfg.cosmos is env_cfg.scene.tiled_camera.modifiers["distance_to_image_plane"][1].backend
