# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-learning coffee scene with IK control and reusable physics/reset setup."""

from __future__ import annotations

import re

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import math as math_utils

from .assets import (
    BOWL_CAVITY_OFFSET,
    BOWL_HEIGHT,
    BOWL_INNER_RADIUS,
    BOWL_POSITION,
    BOWL_SCALE,
    CUP_BOTTOM_THICKNESS,
    CUP_HEIGHT,
    CUP_INNER_RADIUS,
    CUP_POSITION,
    SPOON_BOWL_DEPTH,
    SPOON_BOWL_LENGTH,
    SPOON_BOWL_WIDTH,
)
from .attachment import MountedSpoonAttachment
from .demo_env_cfg import CoffeeDemoEnvCfg, configure_scene
from .physics import CoffeeMPMSolver, configure_grasp_contacts, configure_spoon_rim_margin
from .robot import PALM_BODY_NAME


class CoffeeDemoEnv(ManagerBasedEnv):
    """Control an already-held spoon using absolute palm and finger targets.

    MJWarp advances the arm and hand; implicit MPM advances the material. Mounted
    mode makes the spoon follow the palm. Contact mode instead gives the free
    spoon to MJWarp and couples MPM reaction impulses back through a proxy.
    Both modes start with an already-held spoon; acquisition is a separate task.
    """

    cfg: CoffeeDemoEnvCfg

    def __init__(self, cfg: CoffeeDemoEnvCfg):
        configure_scene(cfg)
        self.presentation = None
        self.dissolution = None
        self._attachment = None
        self._margin = cfg.voxel_size * 0.3
        self._spoon_margin = cfg.voxel_size * cfg.spoon_margin_voxels
        with newton_builder_world_hook(self._configure_colliders):
            super().__init__(cfg)

    def load_managers(self) -> None:
        robot = self.scene["robot"]
        self._palm_id = robot.find_bodies(PALM_BODY_NAME)[0][0]
        self._tool_position = torch.tensor(self.cfg.spoon_offset_pos, device=self.device).expand(self.num_envs, -1)
        self._tool_quaternion = torch.tensor(self.cfg.spoon_offset_rot, device=self.device).expand(self.num_envs, -1)
        self._bowl_position = torch.tensor(
            tuple(position + offset for position, offset in zip(BOWL_POSITION, BOWL_CAVITY_OFFSET)), device=self.device
        )
        self._bowl_inner_radii = torch.tensor(
            (BOWL_INNER_RADIUS * BOWL_SCALE[0], BOWL_INNER_RADIUS * BOWL_SCALE[1]), device=self.device
        )
        self._cup_position = torch.tensor(CUP_POSITION, device=self.device)
        self._source_top = max(BOWL_HEIGHT + 0.002, self.cfg.coffee_fill_height + 0.010)
        self._fraction_cache = None
        self._fraction_step = -1
        self.sync_spoon()
        if self.cfg.grasp_mode == "mounted":
            self._attachment = MountedSpoonAttachment(
                robot, self.scene["spoon"], self.cfg.spoon_offset_pos, self.cfg.spoon_offset_rot
            )
        (solver,) = (
            solver for solver in NewtonMPMManager._implicit_mpm_solvers() if isinstance(solver, CoffeeMPMSolver)
        )
        configure_spoon_rim_margin(solver, self.cfg.spoon_rim_margin_scale)
        if self.cfg.water_enabled:
            from .dissolution import CoffeeDissolution

            solver.dissolution = CoffeeDissolution(
                solver,
                self.particle_ids("coffee"),
                self.particle_ids("water"),
                wetting_time=self.cfg.coffee_wetting_time,
                contact_radius=0.85 * self.cfg.voxel_size,
            )
            self.dissolution = solver.dissolution
        super().load_managers()
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def particle_ids(self, asset_name: str) -> wp.array:
        """Global Newton particle indices for an MPM asset across all scene worlds."""
        expression = re.compile(self.scene[asset_name].cfg.prim_path + r"/.*")
        ranges = [
            np.arange(start, start + count, dtype=np.int32)
            for path, (start, count) in NewtonManager.backend.particle_ranges.items()
            if expression.fullmatch(path)
        ]
        if not ranges:
            raise RuntimeError(f"No particle ranges match {asset_name}.")
        return wp.array(np.concatenate(ranges), dtype=int, device=NewtonManager.get_model().device)

    def _configure_colliders(self, builder, env_id: int, _position, _quaternion) -> None:
        """Expand the thin spoon shell separately from the container colliders."""
        # Keep advection inside the native collider query band at coarse timesteps.
        material_dt = self.cfg.sim.dt / self.cfg.sim.physics.solver_cfg.entries[1].substeps
        builder.particle_max_velocity = min(4.0, 0.5 * self.cfg.voxel_size / material_dt)
        prefix = f"/World/envs/env_{env_id}/"
        for index, label in enumerate(builder.shape_label):
            if not str(label).startswith(prefix):
                continue
            if "/Robot/" in str(label):
                builder.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            elif int(builder.shape_flags[index]) & int(
                newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES
            ):
                builder.shape_flags[index] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
                if self.cfg.grasp_mode != "contact" or "/Spoon/" not in str(label):
                    builder.shape_flags[index] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
                builder.shape_margin[index] = self._spoon_margin if "/Spoon/" in str(label) else self._margin
        if self.cfg.grasp_mode == "contact":
            configure_grasp_contacts(
                builder, (index for index, label in enumerate(builder.shape_label) if str(label).startswith(prefix))
            )

    def palm_pose(self) -> torch.Tensor:
        """Measured palm pose relative to the robot root, shape (N, 7), XYZW."""
        robot = self.scene["robot"]
        palm = robot.data.body_link_pose_w.torch[:, self._palm_id]
        root = robot.data.root_link_pose_w.torch
        pos, quat = math_utils.subtract_frame_transforms(root[:, :3], root[:, 3:], palm[:, :3], palm[:, 3:])
        return torch.cat((pos, quat), dim=-1)

    def spoon_pose(self) -> torch.Tensor:
        """Measured spoon cavity pose relative to its scene origin, shape (N, 7)."""
        pose = self.scene["spoon"].data.root_link_pose_w.torch.clone()
        pose[:, :3] -= self.scene.env_origins
        return pose

    def is_state_finite(self) -> torch.Tensor:
        """Check robot and material positions for each scene, shape (N,)."""
        finite = torch.isfinite(self.scene["robot"].data.joint_pos.torch).all(dim=1)
        finite &= torch.isfinite(self.scene["coffee"].data.particle_pos_w.torch).all(dim=(1, 2))
        if self.cfg.water_enabled:
            finite &= torch.isfinite(self.scene["water"].data.particle_pos_w.torch).all(dim=(1, 2))
        return finite

    def sync_spoon(self, env_ids=None) -> None:
        """Initialize the spoon from the palm during construction or episode reset."""
        selected = slice(None) if env_ids is None else env_ids
        data = self.scene["robot"].data
        palm = data.body_link_pose_w.torch[selected, self._palm_id]
        pos, quat = math_utils.combine_frame_transforms(
            palm[:, :3], palm[:, 3:], self._tool_position[selected], self._tool_quaternion[selected]
        )
        velocity = data.body_link_vel_w.torch[selected, self._palm_id].clone()
        offset = pos - palm[:, :3]
        velocity[:, :3] += torch.cross(velocity[:, 3:], offset, dim=-1)
        spoon = self.scene["spoon"]
        spoon.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos, quat), dim=-1), env_ids=env_ids)
        spoon.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    def particle_fractions(self) -> torch.Tensor:
        """Measure source, carried, delivered and spilled coffee once per step."""
        if self._fraction_step == self._sim_step_counter and self._fraction_cache is not None:
            return self._fraction_cache
        points = self.scene["coffee"].data.particle_pos_w.torch - self.scene.env_origins[:, None, :]
        spoon = self.scene["spoon"].data.root_link_pose_w.torch
        local = math_utils.quat_apply_inverse(
            spoon[:, None, 3:].expand(-1, points.shape[1], -1),
            points + self.scene.env_origins[:, None, :] - spoon[:, None, :3],
        )
        elliptical_radius = (local[..., 0] / (SPOON_BOWL_LENGTH / 2.0)).square() + (
            local[..., 1] / (SPOON_BOWL_WIDTH / 2.0)
        ).square()
        carried = (
            (elliptical_radius < 1.0)
            & (local[..., 2] > SPOON_BOWL_DEPTH * elliptical_radius - 0.001)
            & (local[..., 2] < SPOON_BOWL_DEPTH + 0.004)
        )
        source_local = points - self._bowl_position
        source = (
            ((source_local[..., :2] / self._bowl_inner_radii).square().sum(dim=-1) < 1.0)
            & (source_local[..., 2] > 0.002)
            & (source_local[..., 2] < self._source_top)
            & ~carried
        )
        cup_local = points - self._cup_position
        delivered = (
            (cup_local[..., :2].norm(dim=-1) < CUP_INNER_RADIUS - self._margin)
            & (cup_local[..., 2] > CUP_BOTTOM_THICKNESS)
            & (cup_local[..., 2] < CUP_HEIGHT - self._margin)
            & ~carried
        )
        spilled = ~source & ~carried & ~delivered & (points[..., 2] < 0.005)
        self._fraction_cache = torch.stack((source, carried, delivered, spilled), dim=-1).float().mean(dim=1)
        self._fraction_step = self._sim_step_counter
        return self._fraction_cache

    def reset_scene(self, env_ids: torch.Tensor | slice) -> None:
        """Reset selected robot and tool state, particle history and appearance."""
        robot = self.scene["robot"]
        q = robot.data.default_joint_pos.torch[env_ids].clone()
        robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=env_ids)
        targets = q.clone()
        if self.cfg.grasp_reset_joint_targets is not None:
            names = list(self.cfg.grasp_reset_joint_targets)
            joint_ids, _ = robot.find_joints(names, preserve_order=True)
            targets[:, joint_ids] = torch.tensor(
                [self.cfg.grasp_reset_joint_targets[name] for name in names], device=self.device
            )
        robot.actuators.target_command.set_position_index(value=targets, env_ids=env_ids)
        self.sync_spoon(env_ids)
        mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        mask[: self.num_envs][env_ids] = True
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(mask, dtype=wp.bool), flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE
        )
        self._fraction_cache = None
        self._fraction_step = -1
        if self.presentation is not None:
            self.presentation.reset(env_ids)

    def close(self) -> None:
        if self._attachment is not None:
            self._attachment.close()
            self._attachment = None
        if self.presentation is not None:
            self.presentation.close()
            self.presentation = None
        super().close()
