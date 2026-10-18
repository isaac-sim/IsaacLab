# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 29-DoF locomotion environment on a coupled MPM granular bed."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .env_cfg.physics_cfg import MPM_ENTRY, RIGID_ENTRY, configure_sparse_mpm_capacities
from .g1_mpm_env_cfg import G1MPMEnvCfg


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


class G1MPMEnv(ManagerBasedRLEnv):
    """G1 walking on a granular bed simulated by Newton's implicit MPM solver.

    The MDP terms of ``g1_29dof_soft`` read foot contact from a contact sensor, which a coupled
    solver cannot provide because contact forces live in per-entry buffers. Here the feet are
    virtual proxies of the rigid entry inside the MPM entry, so the granular reaction on each foot
    is exactly the proxy feedback wrench that the coupler hands back to the rigid solver. This
    class harvests that wrench once per environment step and derives the contact flag and the gait
    timers from it, so that every MDP term in :mod:`.mdp` is a cheap read of a cached tensor.
    """

    cfg: G1MPMEnvCfg

    def __init__(self, cfg: G1MPMEnvCfg, render_mode: str | None = None, **kwargs):
        # command-line overrides such as --num_envs land on the config before the scene is created
        configure_sparse_mpm_capacities(cfg)
        self._contact_state_ready = False
        super().__init__(cfg, render_mode, **kwargs)

    def step(self, action: torch.Tensor):
        self._rigid_contact_valid.fill_(True)
        obs, reward, terminated, truncated, extras = super().step(action)
        # Rewards are computed from the pre-reset state, so the step on which a world diverges
        # carries a NaN reward even though `solver_diverged` already reset it. One NaN sample is
        # enough to destroy a policy update, so it is scrubbed here.
        torch.nan_to_num_(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, reward, terminated, truncated, extras

    def load_managers(self) -> None:
        # MDP terms resolve their parameters against this state, so it must exist beforehand
        self._setup_contact_state()
        super().load_managers()

    @property
    def foot_count(self) -> int:
        """Number of feet tracked against the granular bed."""
        return self._foot_body_ids.shape[1]

    """
    Contact state accessors used by the MDP terms.
    """

    @property
    def foot_contact_force(self) -> torch.Tensor:
        """Granular reaction force on each foot in world frame [N], shape ``(num_envs, foot_count, 3)``."""
        self.update_contact_state()
        return self._foot_contact_force

    @property
    def foot_contact(self) -> torch.Tensor:
        """Contact flag per foot as ``float``, shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact.float()

    @property
    def foot_first_contact(self) -> torch.Tensor:
        """Whether a foot touched down this step as ``float``, shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_first_contact.float()

    @property
    def foot_air_time(self) -> torch.Tensor:
        """Time since each foot last left the supporting surface [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_air_time

    @property
    def foot_contact_time(self) -> torch.Tensor:
        """Time since each foot last touched the supporting surface [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact_time

    def update_contact_state(self) -> None:
        """Harvest the proxy feedback wrench and advance the gait timers, at most once per step."""
        if self._contact_state_step == self.common_step_counter:
            return
        previous_contact = self._foot_contact.clone()
        self._refresh_contact_forces()
        self._contact_state_step = self.common_step_counter
        self._foot_first_contact = self._foot_contact & ~previous_contact

        # A mid-step reset invalidates the force cache a second time, but the gait timers must
        # still advance exactly once per step, so they carry their own guard.
        if self._gait_timer_step == self.common_step_counter:
            return
        # mirrors ContactSensor: the timer of the mode a foot is in keeps running
        self._foot_air_time = torch.where(
            self._foot_contact, torch.zeros_like(self._foot_air_time), self._foot_air_time + self.step_dt
        )
        self._foot_contact_time = torch.where(
            self._foot_contact, self._foot_contact_time + self.step_dt, torch.zeros_like(self._foot_contact_time)
        )
        self._gait_timer_step = self.common_step_counter

    def reset_sand_bed(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> None:
        """Restore the bed and clear the solver history of the selected environments.

        Args:
            env_ids: Indices of the environments to reset.
        """
        env_ids = self._env_ids[slice(None) if env_ids is None else env_ids]
        if env_ids.numel() == 0:
            return

        particle_state = self._sand.data.default_particle_state_w.torch[env_ids].clone()
        if self.cfg.reset_particle_jitter > 0.0:
            jitter = (
                2.0 * torch.rand((env_ids.numel(), self._sand.particles_per_object, 3), device=self.device) - 1.0
            ) * self.cfg.reset_particle_jitter
            particle_state[..., :3] += jitter
        particle_state[..., 3:] = 0.0
        self._sand.write_particle_state_to_sim_index(particle_state, env_ids=env_ids)

        # Clear the constitutive, contact and proxy-feedback history of exactly the reset worlds.
        # Newton reset masks carry one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[env_ids] = True
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )

        self._foot_air_time[env_ids] = 0.0
        self._foot_contact_time[env_ids] = 0.0
        self._foot_first_contact[env_ids] = False
        self._foot_contact[env_ids] = False
        self._foot_contact_force[env_ids] = 0.0
        self._rigid_contact_valid[env_ids] = False
        # the cleared feedback is not visible in the solver buffers until the next solver step
        self._contact_state_step = -1

    """
    Internal helpers.
    """

    def _setup_contact_state(self) -> None:
        """Resolve the proxy feedback buffer and allocate the per-foot caches."""
        NewtonMPMManager.get_model().particle_max_velocity = self.cfg.particle_max_velocity
        self._robot = self.scene["robot"]
        self._sand = self.scene["sand"]
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        _, foot_names = self._robot.find_bodies(self.cfg.foot_body_expr, preserve_order=True)
        self._foot_body_ids = self._resolve_newton_body_ids(foot_names)
        self._coupling_forces = self._resolve_proxy_feedback_buffer()

        foot_shape = (self.num_envs, len(foot_names))
        self._foot_contact_force = torch.zeros((*foot_shape, 3), device=self.device)
        self._foot_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_first_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_air_time = torch.zeros(foot_shape, device=self.device)
        self._foot_contact_time = torch.zeros(foot_shape, device=self.device)

        self._contact_state_step = -1
        self._gait_timer_step = -1
        entry = NewtonManager._solver._entries[RIGID_ENTRY]
        self._rigid_solver = entry.solver
        foot_bodies = self._foot_body_ids.cpu().numpy().flatten()
        body_to_foot = {int(body): i for i, body in enumerate(foot_bodies)}
        local_to_global = entry.body_local_to_global.numpy()
        shape_to_foot = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_foot[shape] = body_to_foot.get(int(local_to_global[body]), -1)
        self._shape_to_foot = wp.array(shape_to_foot, device=self.device)
        self._shape_is_ground = wp.array(
            [
                bool(
                    re.match(
                        r"^/World/envs/env_\d+/(ApproachPlatform|RigidBedFloor|Sand/RigidBedFloor|Testbed/tray)(/|$)",
                        label,
                    )
                )
                for label in entry.solver.model.shape_label
            ],
            dtype=wp.bool,
            device=self.device,
        )
        capacity = self._rigid_solver.mjw_data.contact.geom.shape[0]
        self._contact_indices = wp.array(np.arange(capacity, dtype=np.int32), device=self.device)
        self._rigid_contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._rigid_foot_force = wp.zeros(len(foot_bodies), dtype=wp.vec3, device=self.device)
        self._rigid_contact_valid = torch.zeros((self.num_envs, 1, 1), dtype=torch.bool, device=self.device)
        self._contact_state_ready = True
        self._refresh_contact_forces()

    def _resolve_newton_body_ids(self, foot_names: Sequence[str]) -> torch.Tensor:
        """Map the foot link names to Newton body indices, per environment.

        Newton labels bodies by their full prim path, so the environment index and the link name
        can be recovered from the label without relying on the model's body ordering.

        Args:
            foot_names: Foot link names in the order used by the MDP terms.

        Returns:
            The Newton body indices, shape ``(num_envs, len(foot_names))``.
        """
        pattern = re.compile(r"^/World/envs/env_(\d+)/Robot/(?:.*/)?([^/]+)$")
        body_ids = torch.full((self.num_envs, len(foot_names)), -1, dtype=torch.long, device=self.device)
        name_to_column = {name: column for column, name in enumerate(foot_names)}
        for body_id, label in enumerate(NewtonManager.get_model().body_label):
            match = pattern.match(label)
            if match is None:
                continue
            column = name_to_column.get(match.group(2))
            env_id = int(match.group(1))
            if column is not None and env_id < self.num_envs:
                body_ids[env_id, column] = body_id
        if bool((body_ids < 0).any()):
            raise RuntimeError(f"Could not resolve Newton body indices for the foot bodies {list(foot_names)}.")
        return body_ids

    def _resolve_proxy_feedback_buffer(self) -> wp.array:
        """Return the coupler buffer holding the granular reaction wrench of the proxy bodies.

        Returns:
            The per-body spatial forces, shape ``(body_count,)``, indexed by Newton body index.
        """
        solver = NewtonManager._solver
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == RIGID_ENTRY and mapping.dst_name == MPM_ENTRY:
                return mapping.coupling_forces
        raise RuntimeError(
            f"The coupled solver has no body proxy mapping from {RIGID_ENTRY!r} to {MPM_ENTRY!r}, so the granular"
            " reaction on the feet cannot be observed."
        )

    def _refresh_contact_forces(self) -> None:
        """Combine MPM and rigid-ground reaction forces and update foot contact flags."""
        # the coupler stores the feedback as (force, torque); only the linear part is a contact force
        forces = wp.to_torch(self._coupling_forces, requires_grad=False)[:, :3]
        self._foot_contact_force = forces[self._foot_body_ids].to(self.device)
        # Add support forces from the rigid approach platform and the floor beneath the sand.
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
        rigid_force = wp.to_torch(self._rigid_foot_force).reshape(self.num_envs, self.foot_count, 3)
        self._foot_contact_force += torch.where(self._rigid_contact_valid, rigid_force, 0.0)
        torch.nan_to_num_(self._foot_contact_force, nan=0.0, posinf=0.0, neginf=0.0)
        force_magnitude = torch.norm(self._foot_contact_force, dim=-1)
        self._foot_contact = force_magnitude > self.cfg.foot_contact_force_threshold

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super()._reset_idx(env_ids)
        if self._contact_state_ready:
            self.reset_sand_bed(env_ids)


class G1TiltingMPMEnv(G1MPMEnv):
    """Drive testbed joint targets; never reposition the tray or particles during stepping."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._debug_vis_handle = None
        self._testbed_visualizer = None
        cfg.testbed_motion.validate()
        limit = math.radians(cfg.scene.testbed.spawn.joint_limit_deg)
        if any(
            abs(value) >= limit
            for bounds in (cfg.testbed_motion.roll_range, cfg.testbed_motion.pitch_range)
            for value in bounds
        ):
            raise ValueError("Testbed target ranges must lie strictly inside the mechanical joint limits.")
        super().__init__(cfg, render_mode, **kwargs)
        self._testbed_joint_ids, _ = self.scene["testbed"].find_joints(["roll", "pitch"], preserve_order=True)
        self.testbed_tray_body_id = self.scene["testbed"].find_bodies("tray")[0][0]
        self._testbed_ranges = torch.tensor(
            [cfg.testbed_motion.roll_range, cfg.testbed_motion.pitch_range], device=self.device
        )
        self._testbed_frequency = 2 * math.pi / torch.tensor(cfg.testbed_motion.periods, device=self.device)
        length, width, depth = cfg.scene.testbed.spawn.sand_bed_size
        self._testbed_marker_offset = torch.tensor(
            [length / 2 + 0.3, -width / 2 - 0.3, depth + 0.1], device=self.device
        ).expand(self.num_envs, -1)
        # Arrow assets point along local X; rotate X to the tray's local +Z normal.
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._testbed_normal_arrow_rotation = quat_from_euler_xyz(zeros, zeros - math.pi / 2, zeros)
        opposite_gravity = -torch.tensor(cfg.sim.gravity, device=self.device)
        pitch = torch.atan2(-opposite_gravity[2], torch.linalg.vector_norm(opposite_gravity[:2]))
        yaw = torch.atan2(opposite_gravity[1], opposite_gravity[0])
        self._opposite_gravity_arrow_rotation = quat_from_euler_xyz(zeros, zeros + pitch, zeros + yaw)
        self._testbed_marker_indices = torch.cat((zeros, zeros + 1)).long()
        self.set_testbed_debug_vis(cfg.testbed_motion.debug_vis)

    def set_testbed_debug_vis(self, debug_vis: bool) -> None:
        """Toggle the tray-normal and opposite-gravity arrows independently of command markers."""
        if debug_vis:
            if self._testbed_visualizer is None:
                self._testbed_visualizer = VisualizationMarkers(self.cfg.testbed_motion.visualizer_cfg)
            self._testbed_visualizer.set_visibility(True)
            if self._debug_vis_handle is None:
                self._debug_vis_handle = self.sim.vis_marker_registry.add_debug_vis_callback(self)
        else:
            if self._testbed_visualizer is not None:
                self._testbed_visualizer.set_visibility(False)
            self.sim.vis_marker_registry.clear_debug_vis_callback(self)

    def _debug_vis_callback(self, event) -> None:
        """Compare the solved tray normal with opposite gravity from the same arrow origin."""
        testbed = self.scene["testbed"]
        orientation = testbed.data.body_quat_w.torch[:, self.testbed_tray_body_id]
        position = testbed.data.body_pos_w.torch[:, self.testbed_tray_body_id]
        marker_position = position + quat_apply(orientation, self._testbed_marker_offset)
        self._testbed_visualizer.visualize(
            translations=torch.cat((marker_position, marker_position)),
            orientations=torch.cat(
                (quat_mul(orientation, self._testbed_normal_arrow_rotation), self._opposite_gravity_arrow_rotation)
            ),
            marker_indices=self._testbed_marker_indices,
        )

    def close(self):
        if self._debug_vis_handle is not None:
            self.sim.vis_marker_registry.clear_debug_vis_callback(self)
        super().close()

    def step(self, action):
        motion = self.cfg.testbed_motion
        time = (self.episode_length_buf[:, None] * self.step_dt - motion.settling_time).clamp_min(0.0)
        ramp = (time / motion.ramp_time).clamp(max=1.0)
        envelope = 0.5 * (1.0 - torch.cos(math.pi * ramp))
        envelope_velocity = torch.where(
            time < motion.ramp_time, 0.5 * math.pi / motion.ramp_time * torch.sin(math.pi * ramp), 0.0
        )
        midpoint = self._testbed_ranges.mean(dim=-1)
        amplitude = 0.5 * (self._testbed_ranges[:, 1] - self._testbed_ranges[:, 0])
        phase = time * self._testbed_frequency
        wave = midpoint + amplitude * torch.sin(phase)
        target = envelope * wave
        velocity = envelope_velocity * wave + envelope * amplitude * self._testbed_frequency * torch.cos(phase)
        testbed = self.scene["testbed"]
        testbed.set_joint_position_target(target, joint_ids=self._testbed_joint_ids)
        testbed.set_joint_velocity_target(velocity, joint_ids=self._testbed_joint_ids)
        return super().step(action)

    def _reset_idx(self, env_ids):
        # Discontinuous resets are confined to episode boundaries, together with the robot and sand.
        testbed = self.scene["testbed"]
        positions = testbed.data.default_joint_pos.torch[env_ids].clone()
        velocities = torch.zeros_like(positions)
        testbed.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)
        testbed.set_joint_position_target(positions, env_ids=env_ids)
        testbed.set_joint_velocity_target(velocities, env_ids=env_ids)
        super()._reset_idx(env_ids)
