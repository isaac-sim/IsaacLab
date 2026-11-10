# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos on camera tasks: ``apply_cosmos`` for an unmodified task, and the registered Cosmos tasks."""

import logging
import subprocess
import sys
import textwrap

import pytest
from isaaclab_experimental.cosmos import (
    apply_cosmos,
    tasks,  # noqa: F401  (registers the Cosmos tasks)
)
from isaaclab_experimental.cosmos.client import camera as cosmos_camera_module

from isaaclab_tasks.utils import parse_env_cfg, resolve_task_config

KUKA_TASK = "Isaac-Reorient-KukaAllegro-Camera"
KUKA_PRESETS = "presets=cube,single_camera,newton_mjwarp,newton_renderer,rgb64"
SHADOW_HAND_TASK = "Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct"
SHADOW_HAND_MANAGER_TASK = "Isaac-Reorient-Cube-Shadow-Camera-Cosmos"
KUKA_DUO_PRESETS = "presets=cube,duo_camera,newton_mjwarp,newton_renderer,rgb64"


def _kuka(**overrides):
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_PRESETS, "env.scene.num_envs=1"))
    for name, value in overrides.items():
        setattr(env_cfg, name, value)
    return env_cfg


def _budget(env_cfg):
    return env_cfg.scene.base_camera.tensor_processors[1].backend.max_episode_frames


def _serve(monkeypatch, partial_resets=False):
    """Stand in for the running service's status."""
    capabilities = {"partial_resets": partial_resets}
    monkeypatch.setattr(cosmos_camera_module, "service_capabilities", lambda endpoint, **kwargs: capabilities)


@pytest.fixture(autouse=True)
def _idle_service(monkeypatch):
    """Every test sees a running, idle service; tests that batch environments call ``_serve`` again."""
    _serve(monkeypatch)


def test_core_tasks_do_not_import_cosmos_and_registration_imports_no_task_code():
    """The core tasks load without experimental code, and naming the Cosmos tasks is cheap.

    Run in a new process, because this test session has already imported both packages.
    """
    script = textwrap.dedent("""
        import sys

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        for task in (
            "Isaac-Reorient-Cube-Shadow-Camera-Direct",
            "Isaac-Reorient-Cube-Shadow-Camera",
            "IsaacContrib-PickPlace-Apple-G1-Dex3",
        ):
            load_cfg_from_registry(task, "env_cfg_entry_point")
        assert not [name for name in sys.modules if name.startswith("isaaclab_experimental")]

        before = set(sys.modules)
        import isaaclab_experimental.cosmos.tasks  # noqa: F401

        added = set(sys.modules) - before
        assert not [n for n in added if "shadow_hand" in n or "apple" in n or ".client" in n or ".server" in n], added
        load_cfg_from_registry("Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct", "env_cfg_entry_point")
        assert "isaaclab_experimental.cosmos.client" in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr[-2000:]


def test_apply_cosmos_puts_cosmos_on_the_policy_camera_of_an_unmodified_task():
    """The Kuka Allegro camera task gets Cosmos with its 64 x 64 observation kept."""
    env_cfg = _kuka()
    apply_cosmos(env_cfg, prompt=["A lab.", "A kitchen."])
    env_cfg.validate()
    camera = env_cfg.scene.base_camera
    chain = camera.tensor_processors

    assert (camera.width, camera.height) == (640, 640)
    assert chain[1].backend.prompt == ["A lab.", "A kitchen."]
    assert chain[-1].params == {"width": 64, "height": 64}


def test_the_shadow_hand_task_options_are_set_with_overrides():
    """``env.cosmos`` overrides set the camera's own Cosmos settings."""
    prompts = "['A kitchen.', 'A lab.']"
    env_cfg = parse_env_cfg(
        SHADOW_HAND_TASK,
        overrides=(
            "env.scene.num_envs=2",
            f"env.cosmos.prompt={prompts}",
            "env.cosmos.endpoint=tcp://127.0.0.1:5556",
            "env.cosmos.transport=socket",
        ),
    )
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    chain = camera.tensor_processors
    assert len(chain) == 2
    assert chain[1].backend.prompt == ["A kitchen.", "A lab."]
    assert chain[1].backend.endpoint == "tcp://127.0.0.1:5556"
    assert chain[1].backend.transport == "socket"
    assert chain[1].backend.max_episode_frames == 601
    assert env_cfg.cosmos is chain[1].backend
    assert camera.update_period == 0.0
    assert env_cfg.feature_extractor.image_update_frames == 4


def test_several_environments_need_a_compiled_service_that_batches_them(monkeypatch):
    """The batch size is the task's environment count; independent resets still need compiled inference."""
    env_cfg = _kuka()
    env_cfg.scene.num_envs = 4
    _serve(monkeypatch, partial_resets=False)
    with pytest.raises(ValueError, match="without --no-compile"):
        apply_cosmos(env_cfg)

    _serve(monkeypatch, partial_resets=True)
    apply_cosmos(env_cfg)
    assert env_cfg.scene.num_envs == 4
    assert env_cfg.scene.base_camera.tensor_processors[1].backend.max_episode_frames > 1


def test_several_services_split_the_environments_and_each_is_checked(monkeypatch):
    """A list of endpoints splits the environments; only a service with several needs compiled inference."""
    checked = []

    def capabilities(endpoint, **kwargs):
        checked.append(endpoint)
        return {"partial_resets": endpoint == "tcp://127.0.0.1:5556"}

    monkeypatch.setattr(cosmos_camera_module, "service_capabilities", capabilities)
    compiled, eager = "tcp://127.0.0.1:5556", "tcp://127.0.0.1:5557"

    env_cfg = _kuka()
    env_cfg.scene.num_envs = 3
    apply_cosmos(env_cfg, endpoint=[compiled, eager])
    backend = env_cfg.scene.base_camera.tensor_processors[1].backend
    assert backend.endpoint == ["tcp://127.0.0.1:5556", "tcp://127.0.0.1:5557"]
    assert checked == ["tcp://127.0.0.1:5556", "tcp://127.0.0.1:5557"]

    # In the other order, the eager service would get two of the three environments.
    env_cfg = _kuka()
    env_cfg.scene.num_envs = 3
    with pytest.raises(ValueError, match="at tcp://127.0.0.1:5557 without --no-compile"):
        apply_cosmos(env_cfg, endpoint=[eager, compiled])
    with pytest.raises(ValueError, match="at least one environment per service"):
        apply_cosmos(_kuka(), endpoint=[compiled, eager])


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
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_DUO_PRESETS, "env.scene.num_envs=1"))
    with pytest.raises(ValueError, match="`camera` argument"):
        apply_cosmos(env_cfg, prompt="A lab.")
    apply_cosmos(env_cfg, prompt="A lab.", camera="wrist_camera")
    assert env_cfg.scene.wrist_camera.tensor_processors and not env_cfg.scene.base_camera.tensor_processors


def test_named_cameras_share_one_session_at_one_canvas_and_keep_their_sizes():
    """A list of cameras shares one session; a wider wrist camera renders at the shared canvas."""
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_DUO_PRESETS, "env.scene.num_envs=1"))
    env_cfg.scene.wrist_camera = env_cfg.scene.wrist_camera.replace(width=128)
    apply_cosmos(env_cfg, camera=["base_camera", "wrist_camera"])
    env_cfg.validate()

    for name, size in (("base_camera", (64, 64)), ("wrist_camera", (128, 64))):
        camera = getattr(env_cfg.scene, name)
        chain = camera.tensor_processors
        assert (camera.width, camera.height) == (640, 640)
        assert chain[1].backend.cameras == ("base_camera", "wrist_camera") and chain[1].camera == name
        assert (chain[-1].params["width"], chain[-1].params["height"]) == size

    # A list of names selects the cameras in the order given.
    names = ["wrist_camera", "base_camera"]
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_DUO_PRESETS, "env.scene.num_envs=1"))
    apply_cosmos(env_cfg, camera=names)
    assert env_cfg.scene.base_camera.tensor_processors[1].backend.cameras == tuple(names)

    # Two cameras of two environments are four views, which need independent resets.
    with pytest.raises(ValueError, match="without --no-compile"):
        apply_cosmos(parse_env_cfg(KUKA_TASK, overrides=(KUKA_DUO_PRESETS, "env.scene.num_envs=2")), camera=names)
    with pytest.raises(ValueError, match="distinct scene names"):
        apply_cosmos(_kuka(), camera=["base_camera", "base_camera"])


def test_direct_tasks_that_size_observations_from_the_camera_are_warned(caplog, monkeypatch):
    """Cartpole camera derives its observation space from the camera size, which Cosmos changes to the canvas."""
    # A Hydra run earlier in the session with ``hydra/job_logging=disabled`` disables every existing logger.
    monkeypatch.setattr(cosmos_camera_module.logger, "disabled", False)
    env_cfg = parse_env_cfg("Isaac-Cartpole-Camera-Direct", overrides=("env.scene.num_envs=1",))
    with caplog.at_level(logging.WARNING):
        apply_cosmos(env_cfg, prompt="A cart.")
    assert "Direct tasks" in caplog.text


def test_every_camera_sharing_a_session_must_render_synchronously():
    env_cfg = parse_env_cfg(KUKA_TASK, overrides=(KUKA_DUO_PRESETS, "env.scene.num_envs=1"))
    env_cfg.scene.wrist_camera.renderer_cfg.async_rendering = True
    with pytest.raises(ValueError, match="wrist_camera"):
        apply_cosmos(env_cfg, camera=["base_camera", "wrist_camera"])
    assert not env_cfg.scene.base_camera.tensor_processors


def test_separate_shadow_hand_configs_do_not_share_cosmos_settings():
    first = parse_env_cfg(SHADOW_HAND_TASK, overrides=("env.cosmos.prompt=A kitchen.",))
    second = parse_env_cfg(SHADOW_HAND_TASK)
    assert first.cosmos.prompt == "A kitchen." and second.cosmos.prompt != "A kitchen."


def test_a_subclass_of_the_shadow_hand_task_keeps_env_cosmos_on_its_camera():
    """Deriving a configuration from the task copies its fields; ``env.cosmos`` still names the camera's settings."""
    from isaaclab_experimental.cosmos.tasks.shadow_hand_cosmos_env_cfg import ShadowHandCameraCosmosEnvCfg

    from isaaclab.utils import configclass

    @configclass
    class ShorterEpisodeCfg(ShadowHandCameraCosmosEnvCfg):
        episode_length_s: float = 5.0

    env_cfg = ShorterEpisodeCfg()
    assert env_cfg.cosmos is env_cfg.scene.tiled_camera.tensor_processors[1].backend


def test_registered_cosmos_task_composes_rgb_observations_and_local_checkpoint_playback():
    """The training task composes depth-guided RGB, and playback retains its own trained CNN."""
    task_name = SHADOW_HAND_TASK
    env_cfg = parse_env_cfg(task_name, overrides=("env.episode_length_s=30.0",))
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera
    transfer = camera.tensor_processors[-1]

    assert env_cfg.scene.num_envs == 1
    assert camera.data_types == ["rgb"] and transfer.outputs == ["rgb"]
    assert transfer.backend.max_episode_frames == 1801
    assert camera.update_period == 0.0
    assert env_cfg.feature_extractor.enabled and env_cfg.feature_extractor.train
    assert env_cfg.feature_extractor.pretrained_checkpoint is None

    play_cfg, _ = resolve_task_config(task_name, None, play_mode=True, overrides=())
    play_cfg.validate()

    assert play_cfg.scene.num_envs == 1
    assert not play_cfg.feature_extractor.train
    assert play_cfg.feature_extractor.load_checkpoint
    assert play_cfg.feature_extractor.pretrained_checkpoint is None


def test_manager_cosmos_task_feeds_generated_rgb_to_the_camera_observation_term():
    """The Manager task uses the Direct task's Cosmos camera; its observation term reads the generated rgb."""
    env_cfg = parse_env_cfg(SHADOW_HAND_MANAGER_TASK)
    env_cfg.validate()
    camera = env_cfg.scene.tiled_camera

    assert env_cfg.scene.num_envs == 1 and (camera.width, camera.height) == (640, 640)
    processors = camera.tensor_processors
    assert processors[0].inputs == ["distance_to_image_plane"] and processors[-1].outputs == ["rgb"]
    assert env_cfg.observations.policy.camera_features.params["feature_extractor_cfg"] == env_cfg.feature_extractor
    assert env_cfg.feature_extractor.image_update_frames == 4
    assert env_cfg.feature_extractor.pretrained_checkpoint is None


@pytest.mark.parametrize("task", [SHADOW_HAND_TASK, SHADOW_HAND_MANAGER_TASK])
def test_cosmos_tasks_accept_several_environments(task):
    """Both Shadow Hand Cosmos tasks run several environments, one Cosmos view each."""
    env_cfg = parse_env_cfg(task, overrides=("env.scene.num_envs=2",))
    env_cfg.validate()

    assert env_cfg.scene.num_envs == 2


@pytest.mark.parametrize(
    "overrides,error",
    [
        (("env.max_consecutive_success=1",), "max_consecutive_success=0"),
        (("env.scene.lazy_sensor_update=False",), "lazy_sensor_update"),
        (("renderer=ovrtx", "env.scene.tiled_camera.renderer_cfg.async_rendering=True"), "synchronous"),
        (("env.feature_extractor.image_update_frames=1",), "image_update_frames"),
        # Staggered captures would pair a generated image with a later step's pose.
        (("env.scene.num_envs=2", "env.scene.tiled_camera.update_period=0.1"), "capture every step"),
    ],
)
def test_cosmos_task_rejects_overrides_exceeding_service_limits_before_simulation(overrides, error):
    """CLI overrides must preserve the service limits and camera supervision timing."""
    env_cfg = parse_env_cfg(SHADOW_HAND_TASK, overrides=overrides)

    with pytest.raises(ValueError, match=error):
        env_cfg.validate()


APPLE_TASK = "IsaacContrib-PickPlace-Apple-G1-Dex3-Cosmos"


def test_apple_cosmos_task_generates_the_head_camera_image_for_whole_episodes():
    from isaaclab_experimental.cosmos import CosmosTransferProcessorCfg, read_prompt_files
    from isaaclab_experimental.cosmos.tasks.apple_pick_place_cosmos_env_cfg import G1ApplePickPlaceCosmosEnvCfg

    cfg = parse_env_cfg(APPLE_TASK, overrides=())
    assert isinstance(cfg, G1ApplePickPlaceCosmosEnvCfg)
    cfg.validate()
    camera = cfg.scene.head_camera
    # Cosmos renders depth on its 4:3 canvas; the policy still reads a 640 x 480 rgb image.
    assert (camera.width, camera.height) == (736, 544)
    assert camera.tensor_processors[-1].outputs == ["rgb"]
    assert cfg.observations.policy.ego_view.params["sensor_cfg"].name == "head_camera"
    model = next(p.backend for p in camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg))
    # One frame per 30 Hz step for a 20 s episode, plus the first frame, in the service's 1 + 4*k budget.
    assert model.max_episode_frames >= 601 and (model.max_episode_frames - 1) % 4 == 0
    # A longer episode gets a larger budget.
    longer = parse_env_cfg(APPLE_TASK, overrides=("env.episode_length_s=30.0",))
    longer.validate()
    longer_model = next(
        p.backend for p in longer.scene.head_camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg)
    )
    assert longer_model.max_episode_frames >= 901
    # The prompt describes the real demonstrations' appearance.
    assert "apple" in read_prompt_files([model.prompt_files])[0]
    # The core task keeps the rendered image.
    assert not parse_env_cfg("IsaacContrib-PickPlace-Apple-G1-Dex3", overrides=()).scene.head_camera.tensor_processors


@pytest.mark.parametrize(
    "overrides, physics, renderer",
    (
        (("physics=ovphysx", "renderer=ovrtx"), "OvPhysxCfg", "OVRTXRendererCfg"),
        (("physics=newton_mjwarp", "renderer=newton_renderer"), "NewtonCfg", "NewtonWarpRendererCfg"),
    ),
)
def test_apple_cosmos_task_keeps_the_backend_presets(overrides, physics, renderer):
    cfg = parse_env_cfg(APPLE_TASK, overrides=overrides)
    assert type(cfg.sim.physics).__name__ == physics
    assert type(cfg.scene.head_camera.renderer_cfg).__name__ == renderer
