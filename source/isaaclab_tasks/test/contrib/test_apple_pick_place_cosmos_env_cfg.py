# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The apple pick-and-place task with a head camera image that Cosmos generates."""

import os
import subprocess
import sys
import textwrap

import pytest

from isaaclab_contrib.sensors.cosmos_transfer import CosmosTransferProcessorCfg, read_prompt_files

from isaaclab_tasks.contrib.apple_pick_place.apple_pick_place_cosmos_env_cfg import G1ApplePickPlaceCosmosEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

pytestmark = pytest.mark.unit

APPLE_TASK = "IsaacContrib-PickPlace-Apple-G1-Dex3"
APPLE_COSMOS_TASK = "IsaacContrib-PickPlace-Apple-G1-Dex3-Cosmos"


def _transfer(cfg) -> CosmosTransferProcessorCfg:
    return next(p for p in cfg.scene.head_camera.tensor_processors if isinstance(p, CosmosTransferProcessorCfg))


def test_loading_the_apple_task_imports_no_cosmos_code():
    """The apple task loads without the Cosmos client; selecting its Cosmos version imports it.

    Run in a new process, because this test session has already imported the client.
    """
    script = textwrap.dedent(f"""
        import sys

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        load_cfg_from_registry("{APPLE_TASK}", "env_cfg_entry_point")
        cosmos = ("isaaclab_contrib.sensors.cosmos_transfer", "isaaclab_cosmos_server", "isaaclab_neural_bridge")
        assert not [name for name in sys.modules if name.startswith(cosmos)]

        load_cfg_from_registry("{APPLE_COSMOS_TASK}", "env_cfg_entry_point")
        assert "isaaclab_contrib.sensors.cosmos_transfer" in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr[-2000:]


def test_apple_cosmos_task_generates_the_head_camera_image_for_whole_episodes():
    cfg = parse_env_cfg(APPLE_COSMOS_TASK, overrides=())
    assert isinstance(cfg, G1ApplePickPlaceCosmosEnvCfg)
    cfg.validate()
    camera = cfg.scene.head_camera
    # Cosmos renders depth on its 4:3 canvas; the policy still reads a 640 x 480 rgb image.
    assert (camera.width, camera.height) == (736, 544)
    assert camera.tensor_processors[-1].outputs == ["rgb"]
    assert cfg.observations.camera_images.ego_view.params["sensor_cfg"].name == "head_camera"
    client = _transfer(cfg).backend
    # One frame per 30 Hz step for a 20 s episode, plus the first frame, in the service's 1 + 4*k budget.
    assert client.max_episode_frames >= 601 and (client.max_episode_frames - 1) % 4 == 0
    # A longer episode gets a larger budget.
    longer = parse_env_cfg(APPLE_COSMOS_TASK, overrides=("env.episode_length_s=30.0",))
    longer.validate()
    assert _transfer(longer).backend.max_episode_frames >= 901
    # The prompt describes the real demonstrations' appearance.
    assert "apple" in read_prompt_files([client.prompt_files])[0]
    # ``env.cosmos`` overrides choose other appearances, here every bundled prompt in sorted order, and
    # ``env.cosmos_seed`` gives each environment its own seed.
    pattern = os.path.join(os.path.dirname(client.prompt_files), "apple_pick_place*.json")
    overrides = (f"env.cosmos.prompt_files={pattern}", "env.cosmos_seed=[101,202]", "env.scene.num_envs=2")
    transfer = _transfer(parse_env_cfg(APPLE_COSMOS_TASK, overrides=overrides))
    assert transfer.backend.prompt_files == pattern and list(transfer.seed) == [101, 202]
    prompts = read_prompt_files([pattern])
    assert len(prompts) == 4 and prompts[0] == read_prompt_files([client.prompt_files])[0]
    # The core task keeps the rendered image.
    assert not parse_env_cfg(APPLE_TASK, overrides=()).scene.head_camera.tensor_processors


@pytest.mark.parametrize(
    "overrides, physics, renderer",
    (
        (("physics=ovphysx", "renderer=ovrtx"), "OvPhysxCfg", "OVRTXRendererCfg"),
        (("physics=newton_mjwarp", "renderer=newton_warp"), "NewtonCfg", "NewtonWarpRendererCfg"),
    ),
)
def test_apple_cosmos_task_keeps_the_backend_presets(overrides, physics, renderer):
    cfg = parse_env_cfg(APPLE_COSMOS_TASK, overrides=overrides)
    assert type(cfg.sim.physics).__name__ == physics
    assert type(cfg.scene.head_camera.renderer_cfg).__name__ == renderer
