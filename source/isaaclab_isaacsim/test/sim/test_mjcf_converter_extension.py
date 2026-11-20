# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that the MJCF converter enables the Isaac Sim importer extension under Kit."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import os

import newton
import pytest

import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]

_MJCF_IMPORTER_EXTENSION = "isaacsim.asset.importer.mjcf"


def test_converter_enables_importer_extension(tmp_path):
    """Constructing the converter enables the owning importer extension."""
    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.is_extension_enabled(_MJCF_IMPORTER_EXTENSION):
        pytest.skip("MJCF importer extension was already enabled before constructing MjcfConverter.")

    sim_utils.create_new_stage()
    # force the lazy importer load, which conversion skips when a matching USD already exists
    MjcfConverter(
        MjcfConverterCfg(
            asset_path=os.path.join(os.path.dirname(newton.__file__), "examples", "assets", "nv_ant.xml"),
            usd_dir=str(tmp_path),
            force_usd_conversion=True,
        )
    )

    assert manager.is_extension_enabled(_MJCF_IMPORTER_EXTENSION)
