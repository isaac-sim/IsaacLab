# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain walking and whole-body recovery configurations."""

from isaaclab.utils import clone, configclass

from .groundpick.groundpick_env_cfg import CurriculumCfg as GroundPickCurriculumCfg
from .groundpick.groundpick_env_cfg import MicroDuckGroundPickFlatEnvCfg
from .groundpick.groundpick_env_cfg import TerminationsCfg as GroundPickTerminationsCfg
from .sitstand.sitstand_env_cfg import CurriculumCfg as SitStandCurriculumCfg
from .sitstand.sitstand_env_cfg import MicroDuckSitStandFlatEnvCfg
from .sitstand.sitstand_env_cfg import TerminationsCfg as SitStandTerminationsCfg
from .standup.standup_env_cfg import CurriculumCfg as StandUpCurriculumCfg
from .standup.standup_env_cfg import MicroDuckStandUpFlatEnvCfg
from .standup.standup_env_cfg import TerminationsCfg as StandUpTerminationsCfg
from .terrain import MicroDuckTerrainImporter
from .velocity.velocity_env_cfg import CurriculumCfg as WalkingCurriculumCfg
from .velocity.velocity_env_cfg import MicroDuckVelocityRoughEnvCfg
from .velocity.velocity_env_cfg import TerminationsCfg as WalkingTerminationsCfg


@configclass
class StandUpRoughCurriculumCfg(StandUpCurriculumCfg):
    terrain_levels = clone(WalkingCurriculumCfg().terrain_levels)


@configclass
class SitStandRoughCurriculumCfg(SitStandCurriculumCfg):
    terrain_levels = clone(WalkingCurriculumCfg().terrain_levels)


@configclass
class GroundPickRoughCurriculumCfg(GroundPickCurriculumCfg):
    terrain_levels = clone(WalkingCurriculumCfg().terrain_levels)


@configclass
class StandUpRoughTerminationsCfg(StandUpTerminationsCfg):
    """Retain the source finite-terrain timeout alongside the task's termination terms."""

    out_of_terrain_bounds = clone(WalkingTerminationsCfg().out_of_terrain_bounds)


@configclass
class SitStandRoughTerminationsCfg(SitStandTerminationsCfg):
    """Retain the source finite-terrain timeout alongside the task's termination terms."""

    out_of_terrain_bounds = clone(WalkingTerminationsCfg().out_of_terrain_bounds)


@configclass
class GroundPickRoughTerminationsCfg(GroundPickTerminationsCfg):
    """Retain the source finite-terrain timeout alongside the task's termination terms."""

    out_of_terrain_bounds = clone(WalkingTerminationsCfg().out_of_terrain_bounds)


def _rough_play_mode(cfg):
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.curriculum.terrain_levels = None


class _RoughTerrainMixin:
    """Keep each task's MDP while enabling independently owned rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        walking = MicroDuckVelocityRoughEnvCfg()
        self.scene.terrain = clone(walking.scene.terrain)
        self.scene.terrain.class_type = MicroDuckTerrainImporter
        self.scene.terrain.terrain_generator.curriculum = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # Retain the task's hardened solver and allocate enough rough contacts.
        for name in ("default", "newton_mjwarp"):
            solver = getattr(self.sim.physics, name).solver_cfg
            rough_solver = getattr(walking.sim.physics, name).solver_cfg
            solver.nconmax = max(solver.nconmax, rough_solver.nconmax)
            solver.njmax = max(solver.njmax, rough_solver.njmax)

    def play_mode(self):
        super().play_mode()
        _rough_play_mode(self)


@configclass
class MicroDuckStandUpRoughEnvCfg(_RoughTerrainMixin, MicroDuckStandUpFlatEnvCfg):
    """Stand-up on the rough terrain distribution."""

    curriculum: StandUpRoughCurriculumCfg = StandUpRoughCurriculumCfg()
    terminations: StandUpRoughTerminationsCfg = StandUpRoughTerminationsCfg()


@configclass
class MicroDuckSitStandRoughEnvCfg(_RoughTerrainMixin, MicroDuckSitStandFlatEnvCfg):
    """Commanded sitting and standing on rough terrain."""

    curriculum: SitStandRoughCurriculumCfg = SitStandRoughCurriculumCfg()
    terminations: SitStandRoughTerminationsCfg = SitStandRoughTerminationsCfg()


@configclass
class MicroDuckGroundPickRoughEnvCfg(_RoughTerrainMixin, MicroDuckGroundPickFlatEnvCfg):
    """Ground-mouth reaching on rough terrain."""

    curriculum: GroundPickRoughCurriculumCfg = GroundPickRoughCurriculumCfg()
    terminations: GroundPickRoughTerminationsCfg = GroundPickRoughTerminationsCfg()


@configclass
class MicroDuckVelocityRoughPresetEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """Walking with bounded, noncurricular rough-terrain playback."""

    def play_mode(self):
        super().play_mode()
        _rough_play_mode(self)
