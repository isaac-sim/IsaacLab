# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv, StateInputSpec


@pytest.mark.parametrize("property_name", ["rgb", "output.rgb"])
def test_read_inputs_resolves_data_property(property_name: str):
    """Test deployment reads both direct data properties and individual camera buffers."""
    rgb = torch.zeros(1, 4, 4, 4, dtype=torch.uint8)
    env = object.__new__(LeappDeploymentEnv)
    buffer = SimpleNamespace(torch=rgb)
    env.scene = {"camera": SimpleNamespace(data=SimpleNamespace(rgb=buffer, output={"rgb": buffer}))}
    env._input_mapping = {"policy/camera_output_rgb": StateInputSpec(entity_name="camera", property_name=property_name)}

    assert env._read_inputs() == {"policy/camera_output_rgb": rgb}
