# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment setup for the G1 terrain traversals."""

from pathlib import Path
from tempfile import TemporaryDirectory

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnv

from .terrain import create_terrain
from .terrain_env_cfg import G1PlatformsEnvCfg


class G1TerrainEnv(ManagerBasedRLEnv):
    """A fixed course with explicit resets and absolute joint position actions."""

    def __init__(self, cfg: G1PlatformsEnvCfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("The reconstructed courses support one robot per scene.")
        is_physx = isinstance(cfg.sim.physics, PhysxCfg)
        if not is_physx and not isinstance(cfg.sim.physics, NewtonCfg):
            raise ValueError("Select physics=isaacsim_physx or physics=newton_mjwarp.")
        cfg.sim.dt = 0.0025 if is_physx else 0.005
        cfg.decimation = 8 if is_physx else 4
        cfg.sim.render_interval = cfg.decimation
        if cfg.terrain_name == "granular":
            from .granular import configure_granular  # noqa: PLC0415

            configure_granular(cfg, is_physx)
        self._terrain_directory = TemporaryDirectory(prefix="g1-terrain-")
        cfg.scene.terrain.usd_path = str(
            create_terrain(Path(self._terrain_directory.name), cfg.terrain_name, cfg.course, is_physx)
        )
        super().__init__(cfg, **kwargs)
        if cfg.terrain_name == "granular" and is_physx:
            self.sim.set_setting("/physics/updateParticlesToUsd", True)

    def close(self):
        """Release the generated terrain after the simulator closes."""
        super().close()
        if hasattr(self, "_terrain_directory"):
            self._terrain_directory.cleanup()

    def reset(self, **kwargs):
        """Initialize solver and sensor buffers with a neutral airborne step before resetting."""
        result = super().reset(**kwargs)
        if self.cfg.warmup_height is None:
            return result
        robot = self.scene["robot"]
        pose = robot.data.root_pose_w.torch.clone()
        pose[:, 2] = self.cfg.warmup_height
        robot.write_root_pose_to_sim(pose)
        self.step(robot.data.default_joint_pos.torch.clone())
        return super().reset(**kwargs)
