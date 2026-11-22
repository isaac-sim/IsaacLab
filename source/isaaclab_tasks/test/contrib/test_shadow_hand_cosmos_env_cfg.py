# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos on camera tasks: ``cosmos_camera`` on an unmodified task, and the registered Cosmos tasks."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from isaaclab_contrib.sensors.cosmos import CosmosModelCfg, cosmos_camera, episode_frame_budget

from isaaclab_tasks.utils import parse_env_cfg, resolve_task_config

pytestmark = pytest.mark.unit

KUKA_TASK = "Isaac-Reorient-KukaAllegro-Camera"
KUKA_PRESETS = "presets=cube,single_camera,newton_mjwarp,newton_warp,rgb64"
SHADOW_HAND_TASK = "IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos-Direct"
SHADOW_HAND_MANAGER_TASK = "IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos"
KUKA_DUO_PRESETS = "presets=cube,duo_camera,newton_mjwarp,newton_warp,rgb64"


def test_registering_the_cosmos_tasks_imports_no_cosmos_code():
    """Importing the tasks registers the Cosmos tasks by name only; selecting one imports the Cosmos client.

    Run in a new process, because this test session has already imported the client.
    """
    script = textwrap.dedent("""
        import sys

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        for task in ("Isaac-Reorient-Cube-Shadow-Camera-Direct", "Isaac-Reorient-Cube-Shadow-Camera"):
            load_cfg_from_registry(task, "env_cfg_entry_point")
        cosmos = ("isaaclab_contrib.sensors.cosmos", "isaaclab_cosmos_server")
        assert not [name for name in sys.modules if name.startswith(cosmos)]

        load_cfg_from_registry("IsaacContrib-Reorient-Cube-Shadow-Camera-Cosmos-Direct", "env_cfg_entry_point")
        assert "isaaclab_contrib.sensors.cosmos" in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr[-2000:]


def test_cosmos_camera_puts_cosmos_on_the_policy_camera_of_an_unmodified_task():
    """The Kuka Allegro camera task gets Cosmos with its 64 x 64 observation kept, as the guide shows."""
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_PRESETS, "env.scene.num_envs=1"))
    camera = env_cfg.scene.base_camera
    model = CosmosModelCfg(
        prompt=["A lab.", "A kitchen."],
        max_episode_frames=episode_frame_budget(env_cfg, camera.update_period)[2],
    )
    env_cfg.scene.base_camera = cosmos_camera(camera, model, near=0.6, far=1.9)
    env_cfg.validate()
    camera = env_cfg.scene.base_camera
    chain = camera.tensor_processors

    assert (camera.width, camera.height) == (640, 640)
    assert chain[1].backend.prompt == ["A lab.", "A kitchen."]
    assert chain[1].backend.max_episode_frames == 361  # 12 s at 1/30 s steps: 360 steps and the initial frame
    assert len(chain) == 2 and chain[1].image_size == (64, 64)


@pytest.mark.parametrize(
    "dt,decimation,episode_length_s,update_period,expected",
    [
        # A capture every step: 5 steps and the initial frame, rounded up to 1 + 4*k.
        (0.01, 1, 0.05, 0.0, (1, 6, 9)),
        # A capture every 10 steps of a 100-step episode.
        (0.01, 1, 1.0, 0.1, (10, 11, 13)),
        # A sensor updates 1e-6 s early, so a 0.0200001 s period at 0.01 s steps captures every 2 steps, not 3.
        (0.005, 2, 3.0, 0.0200001, (2, 151, 153)),
    ],
)
def test_the_frame_budget_holds_the_captures_of_an_episode(dt, decimation, episode_length_s, update_period, expected):
    """Steps per capture, captures per episode including the initial one, and the budget ``1 + 4*k`` holding them."""
    env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=dt), decimation=decimation, episode_length_s=episode_length_s)

    assert episode_frame_budget(env_cfg, update_period) == expected


def test_named_cameras_share_one_session_at_one_canvas_and_keep_their_sizes():
    """Cameras named in the model share one session; a wider wrist camera renders at the shared canvas."""
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_DUO_PRESETS, "env.scene.num_envs=1"))
    env_cfg.scene.wrist_camera = env_cfg.scene.wrist_camera.replace(width=128)
    names = ("base_camera", "wrist_camera")
    model = CosmosModelCfg(
        cameras=names,
        max_episode_frames=episode_frame_budget(env_cfg, env_cfg.scene.base_camera.update_period)[2],
    )
    for name in names:
        setattr(env_cfg.scene, name, cosmos_camera(getattr(env_cfg.scene, name), model, canvas=(640, 640), name=name))
    env_cfg.validate()

    for name, size in (("base_camera", (64, 64)), ("wrist_camera", (64, 128))):
        camera = getattr(env_cfg.scene, name)
        chain = camera.tensor_processors
        assert (camera.width, camera.height) == (640, 640)
        assert chain[1].backend.cameras == names and chain[1].camera == name
        # The Cosmos processor publishes last, at the camera's (height, width).
        assert chain[-1] is chain[1] and chain[1].image_size == size


@pytest.mark.parametrize("task", [SHADOW_HAND_TASK, SHADOW_HAND_MANAGER_TASK])
def test_the_shadow_hand_task_options_are_set_with_overrides(task):
    """``env.cosmos`` overrides set the camera's own Cosmos settings, with several environments."""
    prompts = "['A kitchen.', 'A lab.']"
    env_cfg = parse_env_cfg(
        task,
        overrides=(
            "env.scene.num_envs=2",
            f"env.cosmos.prompt={prompts}",
            "env.cosmos.endpoint=tcp://127.0.0.1:5556",
        ),
    )
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    chain = camera.tensor_processors
    assert env_cfg.scene.num_envs == 2 and len(chain) == 2
    assert chain[1].backend.prompt == ["A kitchen.", "A lab."]
    assert chain[1].backend.endpoint == "tcp://127.0.0.1:5556"
    assert chain[1].backend.max_episode_frames == 601
    assert env_cfg.cosmos is chain[1].backend
    assert camera.update_period == 0.0
    assert env_cfg.feature_extractor.hold_image_targets is True


def test_the_shadow_hand_task_batches_several_environments_at_its_camera_rate():
    """Several environments on the task keep its camera rate; each environment's budget counts its own captures.

    Without training, as in playback: training with several environments needs a capture every step.
    """
    env_cfg = parse_env_cfg(
        SHADOW_HAND_TASK,
        overrides=(
            "env.scene.num_envs=2",
            "env.scene.tiled_camera.update_period=0.1",
            "env.feature_extractor.train=False",
        ),
    )
    env_cfg.validate()

    assert env_cfg.scene.num_envs == 2
    assert env_cfg.scene.tiled_camera.update_period == 0.1
    assert env_cfg.scene.tiled_camera.tensor_processors[-1].backend.max_episode_frames == 101


def test_separate_shadow_hand_configs_do_not_share_cosmos_settings():
    first = parse_env_cfg(SHADOW_HAND_TASK, overrides=("env.cosmos.prompt=A kitchen.",))
    second = parse_env_cfg(SHADOW_HAND_TASK)
    assert first.cosmos.prompt == "A kitchen." and second.cosmos.prompt != "A kitchen."


def test_a_subclass_of_the_shadow_hand_task_keeps_env_cosmos_on_its_camera():
    """Deriving a configuration from the task copies its fields; ``env.cosmos`` still names the camera's settings."""
    from isaaclab.utils import configclass

    from isaaclab_tasks.contrib.reorient.config.shadow_hand.shadow_hand_cosmos_env_cfg import (
        ShadowHandCameraCosmosEnvCfg,
    )

    @configclass
    class ShorterEpisodeCfg(ShadowHandCameraCosmosEnvCfg):
        episode_length_s: float = 5.0

    env_cfg = ShorterEpisodeCfg()
    assert env_cfg.cosmos is env_cfg.scene.tiled_camera.tensor_processors[1].backend


@pytest.mark.parametrize("task", [SHADOW_HAND_TASK, SHADOW_HAND_MANAGER_TASK])
def test_cosmos_tasks_train_the_feature_extractor_on_generated_rgb_and_play_their_own_checkpoint(task):
    """Both tasks run the same Cosmos camera, train the CNN on its rgb, and play back that run's CNN."""
    env_cfg = parse_env_cfg(task, overrides=("env.episode_length_s=30.0",))
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    processors = camera.tensor_processors

    assert env_cfg.scene.num_envs == 1 and (camera.width, camera.height) == (640, 640)
    assert camera.data_types == ["rgb"] and camera.update_period == 0.0
    assert processors[0].inputs == ["distance_to_image_plane"] and processors[-1].outputs == ["rgb"]
    assert processors[-1].backend.max_episode_frames == 1801
    assert env_cfg.feature_extractor.enabled and env_cfg.feature_extractor.train
    assert env_cfg.feature_extractor.hold_image_targets is True
    assert env_cfg.feature_extractor.pretrained_checkpoint is None
    if task == SHADOW_HAND_MANAGER_TASK:
        # The Manager task's observation term reads the generated rgb with the task's feature extractor.
        term = env_cfg.observations.policy.camera_features
        assert term.params["feature_extractor_cfg"] == env_cfg.feature_extractor

    play_cfg, _ = resolve_task_config(task, None, play_mode=True, overrides=())
    play_cfg.validate()

    assert play_cfg.scene.num_envs == 1
    assert not play_cfg.feature_extractor.train
    assert play_cfg.feature_extractor.load_checkpoint
    assert play_cfg.feature_extractor.pretrained_checkpoint is None


@pytest.mark.parametrize(
    "task,overrides", [(SHADOW_HAND_TASK, ("env.max_consecutive_success=1",)), (SHADOW_HAND_MANAGER_TASK, ())]
)
def test_single_frame_cosmos_tasks_need_no_episode_frame_budget(task, overrides):
    """Each single-frame image is a one-frame episode on the service, so no frame budget limits the episode."""
    single_frame = ("env.cosmos.generation_mode=single_frame", "env.cosmos.max_episode_frames=1")
    env_cfg = parse_env_cfg(task, overrides=(*single_frame, *overrides))
    env_cfg.validate()

    assert env_cfg.cosmos.max_episode_frames == 1


@pytest.mark.parametrize(
    "overrides,error",
    [
        (("env.max_consecutive_success=1",), "max_consecutive_success=0"),
        (("env.scene.lazy_sensor_update=False",), "lazy_sensor_update"),
        (("renderer=ovrtx", "env.scene.tiled_camera.renderer_cfg.async_rendering=True"), "synchronous"),
        # Staggered captures would pair a generated image with a later step's pose.
        (("env.scene.num_envs=2", "env.scene.tiled_camera.update_period=0.1"), "capture every step"),
    ],
)
def test_cosmos_task_rejects_overrides_exceeding_service_limits_before_simulation(overrides, error):
    """CLI overrides must preserve the service limits and camera supervision timing."""
    env_cfg = parse_env_cfg(SHADOW_HAND_TASK, overrides=overrides)

    with pytest.raises(ValueError, match=error):
        env_cfg.validate()
