# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared policy interface for pure MJWarp and coupled MJWarp/MPM worlds."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import MPMObject
from isaaclab_newton.envs.mdp.events import randomize_mpm_material as RandomizeMPMMaterial
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .mdp.terminations import root_outside_workspace
from .mpm_env import G1MovingPatchEnv


@wp.kernel
def sum_rigid_foot_forces(
    contact_count: wp.array[int],
    contact_world: wp.array[int],
    contact_geom: wp.array[wp.vec2i],
    geom_to_shape: wp.array2d[int],
    shape_to_foot: wp.array[int],
    shape_is_ground: wp.array[wp.bool],
    contact_wrench: wp.array[wp.spatial_vector],
    foot_force: wp.array[wp.vec3],
):
    """Sum the world-frame forces applied to each foot across its MJWarp contacts."""
    contact = wp.tid()
    if contact < contact_count[0]:
        world = contact_world[contact]
        geoms = contact_geom[contact]
        force = wp.spatial_top(contact_wrench[contact])
        for side in range(2):
            shape = geom_to_shape[world, geoms[side]]
            other_shape = geom_to_shape[world, geoms[1 - side]]
            if shape >= 0 and other_shape >= 0 and shape_is_ground[other_shape]:
                foot = shape_to_foot[shape]
                if foot >= 0:
                    sign = float(2 * side - 1)
                    wp.atomic_add(foot_force, foot, sign * force)


class MixedMPMObject(MPMObject):
    """Particle asset present only in the first half of scene environments."""

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._scene_env_ids = torch.arange(2 * self.num_instances, device=self.device)

    def select_mpm_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        """Map scene reset selections to the particle asset's contiguous instances."""
        ids = self._scene_env_ids[slice(None) if env_ids is None else env_ids]
        return ids[ids < self.num_instances]

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None, env_mask: wp.array | None = None
    ) -> None:
        if env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero().flatten()
        super().reset(env_ids=self.select_mpm_env_ids(env_ids))


class randomize_mpm_material(RandomizeMPMMaterial):
    """Apply the existing material sampler only to environments containing sand."""

    def __init__(self, cfg: EventTermCfg, env: G1MixedTerrainEnv) -> None:
        # The generic sampler requires one particle object per scene environment.
        # Here its sampling rows correspond only to the MPM half of the scene.
        ManagerTermBase.__init__(self, cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._particle_offsets = wp.to_torch(self.asset._particle_offsets).long()
        self._particle_indices = torch.arange(self.asset._particles_per_object, device=env.device)
        self._env_ids = torch.arange(self.asset.num_instances, device=env.device)
        self.material_parameters = {}

    def __call__(
        self,
        env: G1MixedTerrainEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        parameter_ranges: dict[str, tuple[float, float]],
        distribution: str = "uniform",
    ) -> None:
        ids = self.asset.select_mpm_env_ids(env_ids)
        super().__call__(env, ids, asset_cfg, parameter_ranges, distribution)
        if "density" in parameter_ranges and ids.numel():
            env.moving_patch_particle.set_dynamic_particle_density(self.material_parameters["density"][ids], ids)


def outside_contact_region(
    env: G1MixedTerrainEnv, margin: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Keep robots in their contact region without restricting movement between difficulty rows."""
    outside = root_outside_workspace(env, margin=margin, asset_cfg=asset_cfg)
    # The two identical terrain grids meet at world Y=0 after the generator centers the combined mesh.
    split_y = 0.0
    half_patch_width = env.cfg.scene.terrain.moving_patch_terrain.total_patch_size[1] / 2
    y = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 1]
    num_mpm_envs = env.num_envs // 2
    outside[:num_mpm_envs] |= y[:num_mpm_envs] > split_y - half_patch_width - margin
    outside[num_mpm_envs:] |= y[num_mpm_envs:] < split_y + margin
    return outside


class G1MixedTerrainEnv(G1MovingPatchEnv):
    """Advance all robots with MJWarp and only the soft-ground worlds with MPM."""

    def _set_contact_margin(self, payload=None):
        """Use a fixed foot margin for each terrain group before solver construction."""
        num_mpm_envs = self.num_envs // 2
        margins = [self.cfg.mpm_contact_margin] * num_mpm_envs
        margins += [self.cfg.rigid_contact_margin] * (self.num_envs - num_mpm_envs)
        self._set_foot_contact_margins(margins)

    def _setup_contact_state(self) -> None:
        entry = NewtonManager._solver._entries["robot"]
        self._rigid_solver = entry.solver
        foot_names = self.scene["robot"].find_bodies(self.cfg.foot_body_expr, preserve_order=True)[1]
        foot_bodies = self._resolve_newton_body_ids(foot_names).cpu().numpy().flatten()
        body_to_foot = {int(body): index for index, body in enumerate(foot_bodies)}
        local_to_global = entry.body_local_to_global.numpy()
        shape_to_foot = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_foot[shape] = body_to_foot.get(int(local_to_global[body]), -1)
        self._shape_to_foot = wp.array(shape_to_foot, dtype=int, device=self.device)
        self._shape_is_ground = wp.array(
            [label.startswith("/World/ground/") for label in entry.solver.model.shape_label],
            dtype=wp.bool,
            device=self.device,
        )
        capacity = self._rigid_solver.mjw_data.contact.geom.shape[0]
        self._contact_indices = wp.array(np.arange(capacity), dtype=int, device=self.device)
        self._rigid_contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._rigid_foot_force = wp.zeros(len(foot_bodies), dtype=wp.vec3, device=self.device)
        self._rigid_contact_valid = torch.zeros((self.num_envs, 1, 1), dtype=torch.bool, device=self.device)
        self._rigid_contact_valid_wp = wp.from_torch(self._rigid_contact_valid)
        super()._setup_contact_state()

    def _restore_boundary_particles(self) -> None:
        super()._restore_boundary_particles()
        self._rigid_contact_valid_wp.fill_(True)

    def _refresh_contact_forces(self) -> None:
        super()._refresh_contact_forces()
        # Coupled solvers do not support ContactSensor yet; read the MJWarp entry.
        solver = self._rigid_solver
        mjw.contact_force(solver.mjw_model, solver.mjw_data, self._contact_indices, True, self._rigid_contact_wrench)
        self._rigid_foot_force.zero_()
        wp.launch(
            sum_rigid_foot_forces,
            dim=self._contact_indices.shape[0],
            inputs=[
                solver.mjw_data.nacon,
                solver.mjw_data.contact.worldid,
                solver.mjw_data.contact.geom,
                solver.mjc_geom_to_newton_shape,
                self._shape_to_foot,
                self._shape_is_ground,
                self._rigid_contact_wrench,
            ],
            outputs=[self._rigid_foot_force],
            device=self.device,
        )
        # Include MJWarp contacts also when an MPM foot reaches the supporting floor.
        rigid_force = wp.to_torch(self._rigid_foot_force).reshape(self.num_envs, self.foot_count, 3)
        self._foot_contact_force += torch.where(self._rigid_contact_valid, rigid_force, 0.0)
        torch.nan_to_num_(self._foot_contact_force, nan=0.0, posinf=0.0, neginf=0.0)
        self._foot_contact = self._foot_contact_force.norm(dim=-1) > self.cfg.foot_contact_force_threshold

    def reset_mpm_state(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Reset selected robots and clear particle state only for selected MPM worlds."""
        ids = self._sand.select_mpm_env_ids(env_ids)
        if ids.numel():
            particle_state = self._sand.data.default_particle_state_w.torch[ids].clone()
            if self.cfg.reset_particle_jitter > 0.0:
                particle_state[..., :3] += (
                    2.0 * torch.rand_like(particle_state[..., :3]) - 1.0
                ) * self.cfg.reset_particle_jitter
            particle_state[..., 3:] = 0.0
            self._sand.write_particle_state_to_sim_index(particle_state, env_ids=ids)
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[: self.num_envs][env_ids] = True
        mask = wp.from_torch(world_mask, dtype=wp.bool)
        NewtonMPMManager.reset_solver_state(world_mask=mask, flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)
        self._foot_air_time[env_ids] = 0.0
        self._foot_contact_time[env_ids] = 0.0
        self._foot_first_contact[env_ids] = False
        self._rigid_contact_valid[env_ids] = False
        self._contact_state_step = -1
        self.moving_patch_particle.reset(mask)
