# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""mjlab's terrain, which is a different place from ours.

Cloning mjlab's plant and objective (``mc``) still fails the same way as porting the rewards alone:
terrain level pinned at 0, 84% of episodes ending on ``fell_over``. That rules out the actuator as
the explanation, and leaves the ground itself as the largest untested difference.

Read side by side, ``mjlab/terrains/config.py`` against Isaac Lab's ``ROUGH_TERRAINS_CFG`` -- same
size, border, rows and columns, and then:

=====================  ==================  ===========================
sub-terrain            mjlab               ours
=====================  ==================  ===========================
flat                   **0.2**             **absent**
pyramid_stairs         0.2, step 0.0-0.1   0.2, step **0.05-0.23**
pyramid_stairs_inv     0.2, step 0.0-0.1   0.2, step 0.05-0.23
boxes / random grid    **absent**          **0.2**, 0.05-0.2 m
random_rough           0.1                 0.2
hf_pyramid_slope       0.1, slope 0.0-1.0  0.1, slope 0.0-0.4
hf_pyramid_slope_inv   0.1, slope 0.0-1.0  0.1, slope 0.0-0.4
wave_terrain           **0.1**             **absent**
=====================  ==================  ===========================

Two of those matter more than the rest. **A fifth of mjlab's tiles are flat and none of ours are**,
so at difficulty zero their policy has somewhere to learn to stand while ours is already on 5 cm
stairs, 5 cm boxes or noise. And **their tallest step is 0.1 m against our 0.23** -- on a robot whose
pelvis-to-ankle is 0.74 m, a 23 cm step is a third of the leg.

That is exactly the kind of difference a reward set with no termination penalty and no base-height
termination would not survive: mjlab's objective is calibrated to ground where falling is rare, and
our ground makes it common. The steeper slopes and the wave terrain go the other way and are
included anyway, because the point is to reproduce their distribution rather than to cherry-pick
the easy half of it.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_mjlabclone_env_cfg import G129DofRoughMjlabCloneEnvCfg

MJLAB_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=True,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            convert_to_heightfield=True,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 1.0), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 1.0), platform_width=2.0, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=4, border_width=0.25
        ),
    },
)
"""``mjlab/terrains/config.py``'s ``ROUGH_TERRAINS_CFG``, term for term.

``flat`` and ``wave_terrain`` have no counterpart in ours and ``boxes`` has none in theirs; the
stair heights and slope ranges are theirs. Everything outside the sub-terrain table -- size, border,
rows, columns and the scales -- already matched.
"""


@configclass
class G129DofRoughMjlabTerrainEnvCfg(G129DofRoughMjlabCloneEnvCfg):
    """The full clone, standing on mjlab's ground as well."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = MJLAB_TERRAINS_CFG
