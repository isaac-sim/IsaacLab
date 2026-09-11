# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv
from isaaclab.utils.leapp import leapp_tensor_semantics


class _BaseCameraData:
    @property
    @leapp_tensor_semantics(input_transform=lambda image: image.float())
    def rgb(self):
        raise NotImplementedError


class _CameraData(_BaseCameraData):
    def __init__(self):
        self._rgb = SimpleNamespace(torch=torch.full((1, 2, 2, 3), 127, dtype=torch.uint8))

    @property
    def rgb(self):
        return self._rgb


def test_read_inputs_applies_inherited_input_transform():
    """Deployment should replay the transform used to declare an exported input."""
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"camera": SimpleNamespace(data=_CameraData())}
    env.inference = SimpleNamespace(
        nodes={
            "model": SimpleNamespace(
                input_descriptions=[
                    {
                        "name": "base_camera_rgb",
                        "isaaclab_connection": "state:camera:rgb",
                    }
                ],
                output_descriptions=[],
            )
        }
    )
    env._leapp_desc = {
        "pipeline": {
            "inputs": {"model": ["base_camera_rgb"]},
            "outputs": {},
        }
    }
    env._input_mapping = {}
    env._output_mapping = {}

    env._resolve_io()
    inputs = env._read_inputs()

    camera_input = inputs["model/base_camera_rgb"]
    assert camera_input.dtype == torch.float32
    torch.testing.assert_close(camera_input, torch.full((1, 2, 2, 3), 127.0))
