# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Fabric visual-material writer hot path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from isaaclab_physx.renderers import visual_material
from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter


def test_writer_dispatches_only_requested_channels() -> None:
    roughness_selection = MagicMock()
    color_selection = MagicMock()
    writer = FabricVisualMaterialWriter.__new__(FabricVisualMaterialWriter)
    roughness_values = MagicMock(device="cuda:0")
    writer._writes = {
        "roughness": ((object(), roughness_values, roughness_selection, "inverse", "inputs:roughness"),),
        "color": ((object(), object(), color_selection, "inverse", "inputs:diffuseColor"),),
    }
    offsets = MagicMock(__len__=lambda _self: 1)
    env_ids = MagicMock(__len__=lambda _self: 1)

    with (
        patch.object(visual_material.wp, "fabricarray", return_value="output"),
        patch.object(visual_material.wp, "launch") as launch,
    ):
        writer({"roughness": offsets}, env_ids)

    roughness_selection.PrepareForReuse.assert_called_once_with()
    color_selection.PrepareForReuse.assert_not_called()
    launch.assert_called_once_with(
        writer._writes["roughness"][0][0],
        dim=(1, 1),
        inputs=[roughness_values, offsets, env_ids, "inverse", "output"],
        device="cuda:0",
    )
