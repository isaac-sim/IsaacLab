# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper that prepares the downloadable H1 asset before startup."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from .h1_env_cfg import H1TableclothEnvCfg


def _convert_h1_asset() -> str:
    """Download and cache the H1-with-hands MJCF as a fixed-base USD."""
    import newton.utils

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    converter_cfg = MjcfConverterCfg(
        asset_path=str(newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml"),
        usd_dir=os.path.join(tempfile.gettempdir(), "IsaacLab", "tablecloth_h1_vbd"),
        usd_file_name="h1_with_hand.usda",
        fix_base=True,
        self_collision=False,
        robot_type="Humanoid",
        run_asset_transformer=False,
        run_multi_physics_conversion=False,
    )
    return MjcfConverter(converter_cfg).usd_path


class H1TableclothEnv(ManagerBasedRLEnv):
    """Manager-based tablecloth task with lazy H1 asset conversion."""

    def __init__(self, cfg: H1TableclothEnvCfg, **kwargs):
        cfg.scene.robot.spawn.usd_path = _convert_h1_asset()
        super().__init__(cfg, **kwargs)
