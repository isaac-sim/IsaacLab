# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single workcell environment: a Franka and deformable berry tissue on Newton's coupled MPM."""

from isaaclab.envs import ManagerBasedRLEnv

from .physics.coupling import berry_physics_cfg, bind_tissues, configure_tissue_solver, reset_tissue
from .physics.tissue import BerryTissue, berry_layout, load_tissue, place_tissue, random_punnet_poses, tissue_object_cfg


class BerryPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("Berry teleoperation currently supports one environment")
        # Read by _reset_idx, which can run during super().__init__.
        self.berries = {}
        self.berry = None
        specs = [load_tissue(cfg, *berry) for berry in berry_layout(cfg)]
        if cfg.berry_count > 1 and cfg.randomize_layout and cfg.background == "ebc":
            # Scatter the berries in the punnet with random orientations; every berry shares the punnet frame.
            rotations, shifts = random_punnet_poses(specs[0].proxy, len(specs), cfg.layout_seed)
            specs = [
                place_tissue(spec, rotation, shift, cfg.berry_position)
                for spec, rotation, shift in zip(specs, rotations, shifts)
            ]
        for spec in specs:
            setattr(cfg.scene, spec.scene_name, tissue_object_cfg(spec))
        if cfg.tissue_solver != "explicit":
            # The configuration's default physics uses the explicit solver.
            cfg.sim.physics = berry_physics_cfg(solver=cfg.tissue_solver)
        configure_tissue_solver(cfg.sim.physics, specs, cfg.background)
        super().__init__(cfg, **kwargs)
        self.berries = {spec.name: BerryTissue(spec, self.scene[spec.scene_name]) for spec in specs}
        bind_tissues(self.berries.values())
        # Scripted modes and the close-up handle one berry: the selected species, or the first of several.
        self.berry = self.berries[cfg.target_berry] if cfg.berry == "all" else next(iter(self.berries.values()))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.berries:
            # The scene reset restores the particles; this clears the solver's stress and deformation history.
            reset_tissue()
