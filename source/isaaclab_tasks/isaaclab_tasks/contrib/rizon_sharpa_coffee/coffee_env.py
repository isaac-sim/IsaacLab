# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based coffee scooping with a mounted or contact-held teaspoon."""

from __future__ import annotations

import re

import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.cloner import newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.envs import ManagerBasedRLEnv
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
from .coffee_env_cfg import CoffeeEnvCfg, configure_scene
from .physics import CoffeeMPMSolver, configure_grasp_contacts, configure_spoon_rim_margin
from .robot import PALM_BODY_NAME


class CoffeeEnv(ManagerBasedRLEnv):
    """Lean tool-use environment with a 28-dimensional policy action interface.

    MJWarp advances the arm and hand; implicit MPM advances the material. Mounted
    mode makes the spoon follow the palm. Contact mode instead gives the free
    spoon to MJWarp and couples MPM reaction impulses back through a proxy.
    Both modes start with an already-held spoon; acquisition is a separate task.
    """

    cfg: CoffeeEnvCfg

    def __init__(self, cfg: CoffeeEnvCfg, render_mode: str | None = None, **kwargs):
        configure_scene(cfg)
        self.presentation = None
        self.dissolution = None
        self._attachment = None
        self._margin = cfg.voxel_size * 0.3
        self._spoon_margin = cfg.voxel_size * cfg.spoon_margin_voxels
        with newton_builder_world_hook(self._configure_colliders):
            super().__init__(cfg, render_mode=render_mode, **kwargs)

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
        self.tool_targets = torch.stack((self._bowl_position, self._cup_position)).clone()
        self.tool_targets[0, 2] += 0.7 * self.cfg.coffee_fill_height
        self.tool_targets[1, 2] += CUP_HEIGHT + 0.020
        self._source_top = max(BOWL_HEIGHT + 0.002, self.cfg.coffee_fill_height + 0.010)
        self.delivery_dwell = torch.zeros(self.num_envs, device=self.device)
        self._fraction_cache = None
        self._fraction_step = -1
        self._dwell_step = 0
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

    def particle_ids(self, asset_name: str) -> wp.array:
        """Global Newton particle indices for an MPM asset across all task worlds."""
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

    def actions_from_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Encode absolute palm XYZ/XYZW and 22 finger angles into task actions.

        Args:
            targets: Robot-base palm pose followed by named hand positions [rad],
                shape (num_envs, 29). Recompute the encoding before every step.

        Returns:
            Scaled relative palm control and normalized hand positions, shape
            (num_envs, 28). The policy wrapper clips exploration to [-1, 1];
            scripted and teleoperated targets retain their full pose errors.
        """
        if targets.shape != (self.num_envs, 29):
            raise ValueError(f"Expected targets with shape ({self.num_envs}, 29), received {tuple(targets.shape)}.")
        arm = self.action_manager.get_term("arm").actions_from_pose(targets[:, :7])
        hand = self.action_manager.get_term("hand").actions_from_positions(targets[:, 7:])
        return torch.cat((arm, hand), dim=-1)

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
        if self._fraction_step == self.common_step_counter and self._fraction_cache is not None:
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
        qualified = self._fraction_cache[:, 2] >= self.cfg.success_fraction
        if self._dwell_step != self.common_step_counter:
            self.delivery_dwell.copy_(torch.where(qualified, self.delivery_dwell + self.step_dt, 0.0))
            self._dwell_step = self.common_step_counter
        self._fraction_step = self.common_step_counter
        return self._fraction_cache

    def reset_task(self, env_ids: torch.Tensor | slice) -> None:
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
        self.delivery_dwell[env_ids] = 0.0
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
