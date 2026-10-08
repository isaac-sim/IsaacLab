# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single workcell environment: a Franka and deformable berry tissue on Newton's coupled MPM."""

from isaaclab.envs import ManagerBasedRLEnv

from .physics.coupling import (
    check_tissue_solver,
    configure_tissue_solver,
    coupled_physics_cfg,
    register_tissues,
    reset_tissue_solver,
)
from .physics.tissue import (
    BerryTissue,
    fixed_layout,
    load_tissue,
    random_punnet_poses,
    tissue_object_cfg,
    transform_tissue,
)


class BerryPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("Berry teleoperation currently supports one environment")
        # Read by _reset_idx, which can run during super().__init__.
        self.berries = {}
        self.handled_berry = None
        specs = [load_tissue(cfg, *berry) for berry in fixed_layout(cfg)]
        if cfg.num_berries > 1 and cfg.randomize_layout:
            # Scatter the berries in the punnet with random orientations; every berry shares the punnet frame.
            rotations, shifts = random_punnet_poses(specs[0].particles, len(specs), cfg.layout_seed)
            specs = [
                transform_tissue(spec, rotation, shift, cfg.berry_position)
                for spec, rotation, shift in zip(specs, rotations, shifts)
            ]
        for spec in specs:
            setattr(cfg.scene, spec.scene_name, tissue_object_cfg(spec))
        if cfg.tissue_solver != "explicit":
            # The configuration's default physics uses the explicit solver; keep its CUDA graph choice.
            use_cuda_graph = cfg.sim.physics.use_cuda_graph
            cfg.sim.physics = coupled_physics_cfg(solver=cfg.tissue_solver)
            cfg.sim.physics.use_cuda_graph = use_cuda_graph
        configure_tissue_solver(cfg.sim.physics, specs)
        super().__init__(cfg, **kwargs)
        self.berries = {spec.name: BerryTissue(spec, self.scene[spec.scene_name]) for spec in specs}
        register_tissues(self.berries.values())
        # The berry being handled, which the close-up follows; the scripted demo moves on through the others.
        self.handled_berry = next(iter(self.berries.values()))

    def step(self, action):
        result = super().step(action)
        # Reading the solver's error counters synchronizes with the device: check once a second.
        if self.common_step_counter % 30 == 0:
            check_tissue_solver()
        return result

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.berries:
            # The scene reset restores the particles; this clears the solver's stress and deformation history.
            reset_tissue_solver()
