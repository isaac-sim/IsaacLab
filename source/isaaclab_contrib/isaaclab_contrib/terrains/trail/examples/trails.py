# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Example configuration for custom trail terrains."""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from ..elements import roll_functions as roll
from ..trail_cfg import (
    DropCurvesCfg,
    JumpsCfg,
    MultipleRampsCfg,
    RampsCfg,
    RootsCfg,
    SinusoidalCurvesCfg,
    SkinnyCfg,
    SlalomCfg,
    StonesCfg,
    TerrainParameters,
    WavesCfg,
    WingCurvesCfg,
)
from ..trail_cfg import ObjectParameters as OP
from ..trail_cfg import WallParameters as WP

# The following proportions were found iteratively.
# If performance is poor on one type, the corresponding weight is increased
TERRAIN_PROPORTIONS = {
    "reactive": 6.0,
    "drops": 2.5,
    "waves": 3.0,
    "curves": 7.0,
    "skinny": 2.0,
    "slalom": 2.5,
}


@configclass
class TrailTerrainGeneratorCfg(TerrainGeneratorCfg):
    curriculum = False
    size = (50.0, 40.0)  # length, width
    border_width = 0.0
    border_height: float = 0.0
    num_rows = 2  # num terrain levels
    num_cols = 4  # num terrain types
    difficulty_range = (0.8, 1.0)  # 0 for easiest and 1.0 for hardest
    use_cache = False  # set this to True once you are happy with terrains (speed!!)


sub_terrains = {}

# ----------------------------------------------------------------------------------
if TERRAIN_PROPORTIONS["reactive"] > 0:
    sub_terrains["stones"] = StonesCfg(
        proportion=TERRAIN_PROPORTIONS["reactive"] * 0.4,
        length_between_objects=(0.0, 0.15),
        length_between_platform_and_object=(5.0, 6.0),
        cp0=OP(length=(0.2, 0.4), width=2.0, params={"height": 0.01}),
        cp1=OP(length=(0.2, 0.8), width=(0.5, 1.5), params={"height": (0.03, 0.08)}),
        skip_terrain_functions=["delta_z_noise"],
        roll_functions=[roll.sin_x],
        roll0=TerrainParameters(params={"Ax": 0.0, "Tx": 25.0}),
        roll1=TerrainParameters(params={"Ax": (0.0, 0.15), "Tx": (25.0, 30.0)}),
        wp=WP(wall_direction={"up": 1.0}),
    )
    sub_terrains["trunks"] = RootsCfg(
        proportion=TERRAIN_PROPORTIONS["reactive"] * 0.2,
        length_between_objects=(2.0, 6.0),
        length_between_platform_and_object=(0.0, 2.0),
        cp0=OP(
            length=(0.3, 0.5),
            width=2.0,
            params={
                "height": 0.01,
                "amplitude_x": 0.0,
                "amplitude_z": 0.0,
                "T_xz": (0.5, 2.0),
            },
        ),
        cp1=OP(
            length=(0.3, 0.5),
            width=(0.5, 1.5),
            params={
                "height": (0.03, 0.2),
                "amplitude_x": (0.0, 0.2),
                "amplitude_z": (0.0, 0.02),
                "T_xz": (0.5, 2.0),
            },
        ),
        skip_terrain_functions=["delta_z_noise"],
        roll_functions=[roll.sin_x],
        roll0=TerrainParameters(params={"Ax": 0.0, "Tx": 25.0}),
        roll1=TerrainParameters(params={"Ax": (0.0, 0.15), "Tx": (25.0, 30.0)}),
        wp=WP(wall_direction={"up": 1.0}),
    )
    sub_terrains["roots"] = RootsCfg(
        proportion=TERRAIN_PROPORTIONS["reactive"] * 0.4,
        length_between_objects=(0.4, 3.0),
        length_between_platform_and_object=(0.0, 2.0),
        cp0=OP(
            length=(0.18, 0.25),
            width=2.0,
            params={
                "height": 0.01,
                "amplitude_x": 0.0,
                "amplitude_z": 0.0,
                "T_xz": (0.3, 1.5),
            },
        ),
        cp1=OP(
            length=(0.18, 0.25),
            width=(0.5, 1.0),
            params={
                "height": (0.03, 0.1),
                "amplitude_x": (0.0, 0.2),
                "amplitude_z": (0.0, 0.03),
                "T_xz": (0.3, 1.5),
            },
        ),
        skip_terrain_functions=["delta_z_noise"],
        wp=WP(wall_direction={"up": 1.0}),
    )
# ----------------------------------------------------------------------------------
if TERRAIN_PROPORTIONS["drops"] > 0:
    sub_terrains["drops"] = DropCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["drops"] * 0.1,
        length_between_objects=(4.0, 7.0),
        length_between_platform_and_object=(1.0, 5.0),
        cp0=OP(length=(5.5, 6.0), width=2.0, params={"amplitude": 0.0}),
        cp1=OP(length=(4.5, 5.0), width=(0.5, 1.5), params={"amplitude": (1.0, 1.9)}),
        overhanging_clearance=10.0,
    )
    sub_terrains["multiple_ramps"] = MultipleRampsCfg(
        proportion=TERRAIN_PROPORTIONS["drops"] * 0.75,
        length_between_objects=(4.0, 7.0),
        length_between_platform_and_object=(1.0, 5.0),
        cp0=OP(length=(1.0, 1.5), width=2.0, params={"height": 0.0}),
        cp1=OP(length=(2.0, 4.0), width=(1.8, 2.5), params={"height": (0.4, 0.7)}),
    )
    sub_terrains["ramps"] = RampsCfg(
        proportion=TERRAIN_PROPORTIONS["drops"] * 0.15,
        length_between_objects=(4.0, 7.0),
        length_between_platform_and_object=(1.0, 6.0),
        cp0=OP(length=(1.0, 1.5), width=2.0, params={"height": 0.0}),
        cp1=OP(length=(2.5, 3.0), width=(1.0, 2.0), params={"height": (0.3, 0.7)}),
    )
# ----------------------------------------------------------------------------------
if TERRAIN_PROPORTIONS["waves"] > 0:
    sub_terrains["pits"] = DropCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["waves"] * 0.1,
        length_between_objects=(0.0, 4.0),
        num_curves=2,
        cp0=OP(length=(5.5, 6.0), width=2.0, params={"amplitude": 0.0}),
        cp1=OP(length=(5.5, 6.0), width=(0.5, 1.5), params={"amplitude": (-0.5, -1.0)}),
        overhanging_clearance=10.0,
    )
    sub_terrains["big_waves"] = WavesCfg(
        proportion=TERRAIN_PROPORTIONS["waves"] * 0.5,
        length_between_objects=(0.0, 0.4),
        cp0=OP(length=(3.0, 3.5), width=2.0, params={"height": 0.0}),
        cp1=OP(length=(3.0, 3.5), width=(1.2, 2.0), params={"height": (0.3, 0.6)}),
        roll_functions=[roll.sin_x],
        roll0=TerrainParameters(params={"Ax": 0.0, "Tx": 25.0}),
        roll1=TerrainParameters(params={"Ax": (0.0, 0.1), "Tx": (25.0, 30.0)}),
    )
    sub_terrains["little_waves"] = WavesCfg(
        proportion=TERRAIN_PROPORTIONS["waves"] * 0.2,
        length_between_objects=(0.0, 0.2),
        cp0=OP(length=(1.0, 1.5), width=2.0, params={"height": 0.0}),
        cp1=OP(length=(1.0, 1.5), width=(1.2, 2.0), params={"height": (0.05, 0.2)}),
        skip_terrain_functions=["delta_z_noise"],
        roll_functions=[roll.sin_x],
        roll0=TerrainParameters(params={"Ax": 0.0, "Tx": 25.0}),
        roll1=TerrainParameters(params={"Ax": (0.0, 0.15), "Tx": (25.0, 30.0)}),
    )
    sub_terrains["table_jumps"] = JumpsCfg(
        proportion=TERRAIN_PROPORTIONS["waves"] * 0.2,
        gap=False,
        length_between_objects=(1.0, 2.0),
        num_segments=30,
        length_between_platform_and_object=(0.5, 2.0),
        cp0=OP(
            length=(1.5, 2.0),
            width=2.0,
            params={"height": 0.0, "plateau_proportion": 0.0},
        ),
        cp1=OP(
            length=(4.0, 5.0),
            width=(1.5, 2.0),
            params={"height": (0.25, 0.55), "plateau_proportion": (0.35, 0.4)},
        ),
    )
# ----------------------------------------------------------------------------------
if TERRAIN_PROPORTIONS["curves"] > 0:
    sub_terrains["sine_cliff"] = SinusoidalCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["curves"] * 0.3,
        wp=WP(
            wall_functions={"gaussian_wall": 1.0},
            wall_dim={"width": (0.7, 1.1), "height": (0.4, 1.0)},
            wall_direction={"up-down": 0.5, "up": 0.5},
            num_segments=(5, 8),
        ),
        length_between_objects=(0.0, 1.0),
        cp0=OP(length=12.0, width=2.0, params={"amplitude": 0.0}),
        cp1=OP(length=(9.0, 12.0), width=(0.5, 1.0), params={"amplitude": (4.0, 6.0)}),
    )
    sub_terrains["sine_slalom"] = SinusoidalCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["curves"] * 0.2,
        length_between_objects=(0.0, 3.0),
        cp0=OP(length=15.0, width=2.0, params={"amplitude": 1.5}),
        cp1=OP(length=(5.0, 7.0), width=(0.5, 0.7), params={"amplitude": (2.0, 2.5)}),
    )
    sub_terrains["wings_double"] = WingCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["curves"] * 0.15,
        length_between_objects=(0.0, 2.0),
        length_between_platform_and_object=(1.0, 2.0),
        cp0=OP(
            length=10.0,
            width=2.0,
            params={"radius": 1.0, "rel_angle": 0.0, "slope": 0.0},
        ),
        cp1=OP(
            length=10.0,
            width=(0.5, 1.5),
            params={"radius": (0.2, 0.4), "rel_angle": (0.8, 0.9), "slope": (0.0, 0.5)},
        ),
        max_wing_length=(6.0, 7.0),
        skip_terrain_functions=["delta_z_slope_x"],
    )
    sub_terrains["wings_single"] = WingCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["curves"] * 0.1,
        length_between_objects=100.0,
        length_between_platform_and_object=(5.0, 8.0),
        cp0=OP(
            length=10.0,
            width=2.0,
            params={"radius": 1.0, "rel_angle": 0.0, "slope": 0.0},
        ),
        cp1=OP(
            length=10.0,
            width=(0.5, 2.0),
            params={
                "radius": (0.1, 0.4),
                "rel_angle": (0.9, 0.95),
                "slope": (0.0, 0.4),
            },
        ),
        max_wing_length=9.0,
    )
    sub_terrains["wings_slalom"] = WingCurvesCfg(
        proportion=TERRAIN_PROPORTIONS["curves"] * 0.25,
        length_between_objects=(0.0, 2.0),
        cp0=OP(
            length=15.0,
            width=2.0,
            params={"radius": 2.0, "rel_angle": 0.0, "slope": 0.0},
        ),
        cp1=OP(
            length=(5.0, 6.0),
            width=(0.5, 0.7),
            params={"radius": 0.2, "rel_angle": (0.6, 0.7), "slope": (-0.15, 0.15)},
        ),
        max_wing_length=8.0,
    )
# ----------------------------------------------------------------------------------
if TERRAIN_PROPORTIONS["skinny"] > 0:
    sub_terrains["skinny_grounded"] = SkinnyCfg(
        proportion=TERRAIN_PROPORTIONS["skinny"],
        length_between_objects=(3.0, 5.0),
        length_between_platform_and_object=(0.0, 1.0),
        cp0=OP(
            length=2.0,
            width=2.5,
            params={
                "beam_thickness": 0.0,
                "amplitude_y": 0.0,
                "amplitude_z": 0.0,
                "rel_knot_point": (0.25, 0.4),
                "beam_width": 0.7,
            },
        ),
        cp1=OP(
            length=(3.0, 7.0),
            width=(1.5, 2.5),
            params={
                "beam_thickness": (0.1, 0.15),
                "amplitude_y": (0.0, 0.2),
                "amplitude_z": (0.0, 0.1),
                "rel_knot_point": (0.25, 0.4),
                "beam_width": (0.4, 0.7),
            },
        ),
        trail_under_object=None,
    )
# ----------------------------------------------------------------------------------
if TERRAIN_PROPORTIONS["slalom"] > 0:
    sub_terrains["slalom"] = SlalomCfg(
        proportion=TERRAIN_PROPORTIONS["slalom"],
        length_between_objects=(1.0, 3.5),
        length_between_platform_and_object=(1.0, 3.0),
        cp0=OP(
            length=0.4,
            width=2.0,
            params={"height": 1.0, "rel_dist_from_center": (1.0, 1.0)},
        ),
        cp1=OP(
            length=(0.25, 0.4),
            width=(1.1, 2.0),
            params={"height": (0.2, 1.0), "rel_dist_from_center": (0.5, 1.0)},
        ),
        roll1=TerrainParameters(params={"A": (0.0, 0.2)}),
    )

# ----------------------------------------------------------------------------------
TRAIL_CFG = TrailTerrainGeneratorCfg(sub_terrains=sub_terrains)
