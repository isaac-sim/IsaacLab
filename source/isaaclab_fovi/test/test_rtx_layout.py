# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reject atlas layouts that destroy spatial neighborhoods before rendering."""

import pytest
from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg


@pytest.mark.parametrize("layout", ["row_major", "unknown"])
def test_unsupported_layout_rejected_at_configuration(layout: str) -> None:
    with pytest.raises(ValueError, match="ndc_layout.*cortical_disc"):
        FoveatedCameraCfg(ndc_layout=layout)


def test_cortical_disc_remains_default() -> None:
    assert FoveatedCameraCfg().ndc_layout == "cortical_disc"


def test_native_grid_requires_matching_sensor() -> None:
    with pytest.raises(ValueError, match="native_grid requires"):
        FoveatedCameraCfg(ndc_layout="native_grid")
    with pytest.raises(ValueError, match="requires ndc_layout"):
        FoveatedCameraCfg(foveated_style="warped_cartesian_as_grid")
    cfg = FoveatedCameraCfg(ndc_layout="native_grid", foveated_style="warped_cartesian_as_grid")
    assert cfg.fov_type == "circular"
