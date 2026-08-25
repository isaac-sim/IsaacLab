# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Original GR1T2 wheel teleoperation with a curated Snap Circuits bench."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab_teleop.xr_cfg import XrCfg

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass

from .pickplace_gr1t2_env_cfg import ObjectTableSceneCfg
from .pickplace_gr1t2_waist_enabled_env_cfg import PickPlaceGR1T2WaistEnabledEnvCfg

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ASSET_ROOT = _REPO_ROOT / "docker" / "apple-vision-pro" / "snap-circuits" / "assets"
_ASSET_ROOT = Path(os.environ.get("ISAACLAB_SNAP_CIRCUITS_ASSET_ROOT", _DEFAULT_ASSET_ROOT)).expanduser()
_DEMO_USD = _ASSET_ROOT / "prepared" / "snap_circuits_table.usda"


@configclass
class SnapCircuitsGR1T2SceneCfg(ObjectTableSceneCfg):
    """The proven GR1T2 wheel scene plus the curated component layout."""

    # Keep the wheel at its original left-side pose. The prepared scene holds
    # 17 curated items and is shifted right so the two manipulation areas do
    # not overlap on the packing table.
    snap_circuits = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SnapCircuitsDemo",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.60, 1.0)),
        spawn=UsdFileCfg(usd_path=str(_DEMO_USD), semantic_tags=[("class", "demo_component")]),
    )


@configclass
class PickPlaceGR1T2SnapCircuitsEnvCfg(PickPlaceGR1T2WaistEnabledEnvCfg):
    """Use the original GR1T2 control pipeline with wheel and Snap Circuits."""

    # Absolute action used by the headset-free preview: left wrist pose,
    # right wrist pose, then 11 joints per hand.  The waist-enabled base task
    # retains the same 36-D action contract but did not expose its idle pose.
    idle_action = [
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        *([0.0] * 22),
    ]

    scene: SnapCircuitsGR1T2SceneCfg = SnapCircuitsGR1T2SceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Preserve the original task's proven wheel pose and scale exactly.
        # A larger wheel at a slightly altered height can initially intersect
        # the tabletop and be expelled from the live PhysX scene.
        self.scene.object.init_state.pos = (-0.45, 0.45, 0.9996)
        self.scene.object.spawn.scale = (0.75, 0.75, 0.75)
        # Present the 1.0 m world-height worktop at about 1.15 m from the
        # tracked floor, which is a comfortable standing height for this user.
        self.xr = XrCfg(anchor_pos=(0.0, 0.25, -0.15), anchor_rot=(0.0, 0.0, 0.0, 1.0))
        self.isaac_teleop.xr_cfg = self.xr
