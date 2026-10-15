# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone G1 environment with two-way MPM coupling and alpha moving terrain."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager
from isaaclab_newton.physics.newton_manager_cfg import NewtonBuilderCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.physics import PhysicsEvent

from .env_cfg.physics_cfg import MPM_ENTRY, RIGID_ENTRY
from .mpm_env_cfg import G1MovingPatchEnvCfg
from .util.particles import MovingPatchParticles
from .util.visualizer import MovingPatchGLVisualizer, MovingPatchRTXVisualizer


class G1MovingPatchEnv(ManagerBasedRLEnv):
    """G1 walking on a granular bed simulated by Newton's implicit MPM solver.

    The coupled solver exposes terrain contact forces through proxy feedback buffers.
    This environment reads each foot's feedback wrench once per environment step
    and derives contact flags and gait timers for the local MDP functions.
    """

    cfg: G1MovingPatchEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.moving_patch_particle: MovingPatchParticles | None = None
        if cfg.foot_contact_margin_range is not None:
            lower, upper = cfg.foot_contact_margin_range
            if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower <= upper):
                raise ValueError("foot_contact_margin_range must contain finite, nonnegative, ordered bounds.")
        margin_callback = NewtonManager.register_callback(self._set_contact_margin, PhysicsEvent.MODEL_INIT)
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            margin_callback.deregister()

    def _set_contact_margin(self, payload=None):
        """Assign sampled margins before model finalization when configured."""
        if self.cfg.foot_contact_margin_range is None:
            return
        lower, upper = self.cfg.foot_contact_margin_range
        ratios = torch.linspace(0.0, 1.0, self.num_envs)
        ratios = ratios[torch.randperm(self.num_envs)]
        self._set_foot_contact_margins((lower + ratios * (upper - lower)).tolist())

    def _set_foot_contact_margins(self, margins: Sequence[float]) -> None:
        """Set each environment's foot shape margins before the solvers copy them."""
        builder = self.sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self.cfg.sim.physics))
        matched_envs = set()
        for shape_index, body_index in enumerate(builder.shape_body):
            if body_index < 0 or not re.fullmatch(
                self.cfg.foot_body_expr, builder.body_label[body_index].rsplit("/", 1)[-1]
            ):
                continue
            env_id = builder.body_world[body_index]
            if 0 <= env_id < self.num_envs:
                builder.shape_margin[shape_index] = margins[env_id]
                matched_envs.add(env_id)
        if len(matched_envs) != self.num_envs:
            raise ValueError("Could not find foot shapes in every environment when setting contact margins.")

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        self.moving_patch_particle.check_terrain_queries()
        # Rewards are computed from the pre-reset state, so the step on which a world diverges
        # carries a NaN reward even though `solver_diverged` already reset it. One NaN sample is
        # enough to destroy a policy update, so it is scrubbed here.
        torch.nan_to_num_(reward, nan=0.0, posinf=0.0, neginf=0.0)
        # Preserve pre-reset diagnostics and give the logger a fresh per-step snapshot.
        observation_metrics = extras.pop("_observation_metrics", None)
        if observation_metrics is not None:
            extras["log"] = {**extras.get("log", {}), **observation_metrics}
        return obs, reward, terminated, truncated, extras

    def load_managers(self) -> None:
        # MDP terms resolve their parameters against this state, so it must exist beforehand
        self._setup_moving_patch_particle()
        self._setup_contact_state()
        super().load_managers()

    def _setup_moving_patch_particle(self) -> None:
        """Create particles from this scene's terrain and attach physics callbacks."""
        terrain = self.scene.terrain
        self.moving_patch_particle = MovingPatchParticles(
            model=NewtonManager.get_model(),
            state=NewtonManager.get_state_0(),
            coupled_solver=NewtonManager._solver,
            terrain=self.cfg.scene.terrain.moving_patch_terrain,
            entry_name=MPM_ENTRY,
            background_mesh=terrain.background_mesh,  # type: ignore
        )
        NewtonManager.register_state_force_callback(self._update_moving_patch_particle)
        NewtonManager.register_post_step_callback(self._restore_boundary_particles)
        for visualizer in self.sim._visualizers:
            if isinstance(visualizer, (MovingPatchGLVisualizer, MovingPatchRTXVisualizer)):
                visualizer.moving_patch_particle = self.moving_patch_particle

    def _update_moving_patch_particle(self, state: newton.State) -> None:
        """Recycle and pin boundary particles before each solver substep."""
        self.moving_patch_particle.update(state)

    def _restore_boundary_particles(self) -> None:
        """Remove boundary drift after the final substep of each physics step."""
        self.moving_patch_particle.restore_boundary_particles(NewtonManager.get_state_0())

    def close(self) -> None:
        """Detach this environment's callbacks before releasing its scene."""
        if getattr(self, "_is_closed", True):
            return
        if self._update_moving_patch_particle in NewtonManager._state_force_callbacks:
            # Newton currently exposes registration, but no removal method, for this hook.
            NewtonManager._state_force_callbacks.remove(self._update_moving_patch_particle)
        NewtonManager.unregister_post_step_callback(self._restore_boundary_particles)
        for visualizer in self.sim._visualizers:
            if isinstance(visualizer, (MovingPatchGLVisualizer, MovingPatchRTXVisualizer)):
                visualizer.moving_patch_particle = None
        self.moving_patch_particle = None
        super().close()

    """
    properties
    """

    @property
    def foot_count(self) -> int:
        """Number of feet tracked against the granular bed."""
        return self._foot_body_ids.shape[1]

    @property
    def foot_contact_force(self) -> torch.Tensor:
        """Granular reaction force in world frame [N], shape ``(num_envs, foot_count, 3)``.

        Nonfinite components are reported as zero; the solver's feedback buffer is unchanged.
        """
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
        """Time since each foot last left the bed [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_air_time

    @property
    def foot_contact_time(self) -> torch.Tensor:
        """Time since each foot last touched the bed [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact_time

    """
    Contact state accessors used by the MDP terms.
    """

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

    def reset_mpm_state(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Restore the bed and clear the solver history of the selected environments.

        Args:
            env_ids: Environment indices or a slice selecting environments to reset.
        """
        particle_state = self._sand.data.default_particle_state_w.torch[env_ids].clone()
        if particle_state.shape[0] == 0:
            return
        if self.cfg.reset_particle_jitter > 0.0:
            jitter = (
                2.0 * torch.rand(particle_state.shape[:2] + (3,), device=self.device) - 1.0
            ) * self.cfg.reset_particle_jitter
            particle_state[..., :3] += jitter
        particle_state[..., 3:] = 0.0
        self._sand.write_particle_state_to_sim_index(particle_state, env_ids=env_ids)

        # Clear the constitutive, contact and proxy-feedback history of exactly the reset worlds.
        # Newton reset masks carry one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[: self.num_envs][env_ids] = True
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )

        self._foot_air_time[env_ids] = 0.0
        self._foot_contact_time[env_ids] = 0.0
        self._foot_first_contact[env_ids] = False
        # the cleared feedback is not visible in the solver buffers until the next solver step
        self._contact_state_step = -1
        patch = self.moving_patch_particle
        if patch is not None:
            patch.reset(wp.from_torch(world_mask, dtype=wp.bool))

    """
    Internal helpers.
    """

    def _setup_contact_state(self) -> None:
        """Resolve the proxy feedback buffer and allocate the per-foot caches."""
        NewtonMPMManager.get_model().particle_max_velocity = self.cfg.particle_max_velocity
        self._robot = self.scene["robot"]
        self._sand = self.scene["sand"]

        _, foot_names = self._robot.find_bodies(self.cfg.foot_body_expr, preserve_order=True)
        self._foot_body_ids = self._resolve_newton_body_ids(foot_names)
        self._coupling_force_body_ids = self._foot_body_ids
        self._coupling_forces = self._resolve_coupling_feedback_buffer()

        foot_shape = (self.num_envs, len(foot_names))
        self._foot_contact_force = torch.zeros((*foot_shape, 3), device=self.device)
        self._foot_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_first_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_air_time = torch.zeros(foot_shape, device=self.device)
        self._foot_contact_time = torch.zeros(foot_shape, device=self.device)

        self._contact_state_step = -1
        self._gait_timer_step = -1
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
        pattern = re.compile(r"^/World/envs/env_(\d+)/Robot/.*/([^/]+)$")
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

    def _resolve_coupling_feedback_buffer(self) -> wp.array:
        """Return the MPM reaction-wrench buffer indexed by global robot body IDs."""
        solver = NewtonManager._solver
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == RIGID_ENTRY and mapping.dst_name == MPM_ENTRY:
                return mapping.coupling_forces
        raise RuntimeError(
            f"The coupled solver has no body proxy mapping from {RIGID_ENTRY!r} to {MPM_ENTRY!r}, so the granular"
            " reaction on the feet cannot be observed."
        )

    def _refresh_contact_forces(self) -> None:
        """Read feedback, replace nonfinite components with zero, and rebuild contact flags."""
        # the coupler stores the feedback as (force, torque); only the linear part is a contact force
        forces = wp.to_torch(self._coupling_forces, requires_grad=False)[:, :3]
        self._foot_contact_force = forces[self._coupling_force_body_ids].to(self.device)
        # Retain evidence of invalid feedback even if a reset clears it later in this step.
        nonfinite_fraction = (~torch.isfinite(self._foot_contact_force)).float().mean()
        metrics = self.extras.setdefault("_observation_metrics", {})
        key = "Metrics/mpm/contact_force_nonfinite_fraction"
        metrics[key] = torch.maximum(metrics.get(key, nonfinite_fraction), nonfinite_fraction)
        torch.nan_to_num_(self._foot_contact_force, nan=0.0, posinf=0.0, neginf=0.0)
        force_magnitude = torch.norm(self._foot_contact_force, dim=-1)
        self._foot_contact = force_magnitude > self.cfg.foot_contact_force_threshold

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super()._reset_idx(env_ids)  # type: ignore
        self.reset_mpm_state(env_ids)
