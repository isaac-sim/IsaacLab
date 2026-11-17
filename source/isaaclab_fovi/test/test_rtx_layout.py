# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration defaults and validation of the foveated camera."""

import pytest
from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg


def test_defaults_are_spherical_grid_sampling_at_the_human_cmf_estimate() -> None:
    cfg = FoveatedCameraCfg()
    assert cfg.field_geometry == "spherical"
    assert cfg.foveated_style == "warped_cartesian_as_grid"
    assert cfg.cmf_a_deg == 0.591


def test_unknown_field_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="field_geometry"):
        FoveatedCameraCfg(field_geometry="legacy")
