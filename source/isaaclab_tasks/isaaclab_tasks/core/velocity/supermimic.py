# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable heads for the shared flat-ground velocity MDP.

Reward weights, entity selections, actuator gains and action mappings come from
the resolved robot environment. Contact heads predict sensor events; they do not
expose privileged contact or mass inputs to the actor. Torque uses a clipped PD
approximation, and joint acceleration uses policy-rate velocity differences.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat

from .mdp.observations import _feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelocityStateRepresentation:
    """Floating-base coordinates with world-frame SO(3) tangent increments.

    Contact clocks are observed features, carried through integration and then
    advanced analytically by :meth:`VelocityTask.post_dynamics`. They are never
    learned targets. All methods preserve arbitrary leading batch dimensions.
    """

    def __init__(self, num_joints: int, num_feet: int, device: str = "cpu") -> None:
        self.widths = (3, 3, 4, 1, num_joints, num_joints, 3 * num_feet, num_feet, num_feet, num_feet)
        self.increment_widths = (3, 3, 3, 1, num_joints, num_joints, 3 * num_feet)
        self.increment_dim = sum(self.increment_widths)
        self.error_floor = torch.full((self.increment_dim,), 0.01, device=device)

    def unflatten(self, flat: torch.Tensor) -> State:
        """Decode the noise-free ``state`` observation group."""
        return flat.split(self.widths, dim=-1)

    def features(self, state: State) -> torch.Tensor:
        """Encode physical coordinates and observed contact clocks."""
        return torch.cat(state, dim=-1)

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Encode translation/joint differences and the shortest rotation delta."""
        rotation = quat_mul(next_state[2], quat_conjugate(state[2]))
        rotation = torch.where(rotation[..., 3:] < 0, -rotation, rotation)
        magnitude = torch.linalg.vector_norm(rotation[..., :3], dim=-1, keepdim=True)
        # The clamped denominator preserves finite derivatives at zero rotation.
        scale = 2 * torch.atan2(magnitude, rotation[..., 3:]) / magnitude.clamp_min(1e-8)
        tangent = rotation[..., :3] * torch.where(magnitude > 1e-6, scale, 2.0)
        return torch.cat(tuple(tangent if i == 2 else next_state[i] - state[i] for i in range(7)), dim=-1)

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Integrate learned coordinates while carrying the analytic clocks."""
        parts = increment.split(self.increment_widths, dim=-1)
        angle = torch.linalg.vector_norm(parts[2], dim=-1, keepdim=True)
        delta = torch.cat((0.5 * torch.sinc(angle / (2 * math.pi)) * parts[2], torch.cos(angle / 2)), dim=-1)
        orientation = normalize(quat_mul(delta, state[2]))
        return tuple(orientation if i == 2 else state[i] + parts[i] for i in range(7)) + state[7:]

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Measure learned-coordinate error, excluding the analytic clocks."""
        return self.increments(target, prediction)


class VelocityTask(DifferentiableTask):
    """Robot-parameterized differentiable task for the shared flat velocity MDP.

    G1 flat is the first consumer. Commands are held over an imagined horizon;
    real command resampling is read from the next physical ``goal`` observation.
    Analytic contact clocks use policy-step resolution, approximating contact
    changes within physics substeps. Real reward parity uses observed clocks.
    """

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu") -> None:
        owner = env.unwrapped
        robot = owner.scene["robot"]
        data = robot.data
        self.device = device
        self.dt = owner.step_dt
        self.num_joints = data.joint_pos.shape[-1]
        self.num_actions = owner.action_manager.total_action_dim
        self.foot_ids, self.sensor_foot_ids = _feet(owner)
        self.num_feet = len(self.foot_ids)
        sensor = owner.scene.sensors[owner.cfg.rewards.feet_air_time.params["sensor_cfg"].name]
        self.num_bodies = sensor.num_sensors
        self.num_labels = self.num_bodies + 2 * self.num_feet
        self.representation = VelocityStateRepresentation(self.num_joints, self.num_feet, device)
        self.default_pos = data.default_joint_pos.torch[0].to(device).clone()
        self.default_vel = data.default_joint_vel.torch[0].to(device).clone()
        self.limits = data.soft_joint_pos_limits.torch[0].to(device).clone()
        policy_cfg = owner.cfg.observations.policy
        if policy_cfg.height_scan is not None:
            raise ValueError("VelocityTask currently supports flat velocity observations without a height scan.")
        terms = (
            ("base_lin_vel", 3),
            ("base_ang_vel", 3),
            ("projected_gravity", 3),
            ("joint_pos", self.num_joints),
            ("joint_vel", self.num_joints),
            ("actions", self.num_actions),
        )
        self.actor_terms = [
            (index, width, getattr(policy_cfg, name))
            for index, (name, width) in enumerate(terms)
            if getattr(policy_cfg, name) is not None
        ]
        self.enable_corruption = policy_cfg.enable_corruption
        for _, _, term in self.actor_terms:
            if term.clip is not None or term.scale is not None:
                raise ValueError("VelocityTask currently requires unscaled, unclipped policy observation terms.")
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.rewards = {name: owner.reward_manager.get_term_cfg(name) for name in owner.reward_manager.active_terms}
        self.terminations = {
            name: owner.termination_manager.get_term_cfg(name)
            for name in owner.termination_manager.active_terms
            if not owner.termination_manager.get_term_cfg(name).time_out
        }
        supported = {
            "is_terminated",
            "track_lin_vel_xy_exp",
            "track_lin_vel_xy_yaw_frame_exp",
            "track_ang_vel_z_exp",
            "track_ang_vel_z_world_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
            "joint_torques_l2",
            "joint_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "feet_air_time_positive_biped",
            "feet_slide",
            "joint_pos_limits",
            "joint_deviation_l1",
        }
        for name, cfg in self.rewards.items():
            if cfg.weight and cfg.func.__name__ not in supported:
                raise ValueError(f"Unsupported differentiable velocity reward: {name} ({cfg.func.__name__}).")
        sensor_name = owner.cfg.rewards.feet_air_time.params["sensor_cfg"].name
        for cfg in (*self.rewards.values(), *self.terminations.values()):
            if "sensor_cfg" in cfg.params and cfg.params["sensor_cfg"].name != sensor_name:
                raise ValueError("Velocity contact terms must use the feet_air_time sensor.")
            if cfg.func.__name__ == "feet_slide":
                sensor_ids, asset_ids = cfg.params["sensor_cfg"].body_ids, cfg.params["asset_cfg"].body_ids
                if (
                    isinstance(sensor_ids, slice)
                    or isinstance(asset_ids, slice)
                    or set(torch.as_tensor(sensor_ids).tolist()) != set(self.sensor_foot_ids)
                    or set(torch.as_tensor(asset_ids).tolist()) != set(self.foot_ids)
                ):
                    raise ValueError("Velocity sliding and air-time terms must select the same feet.")
        for cfg in self.terminations.values():
            if cfg.func.__name__ != "illegal_contact" or cfg.params["threshold"] != 1:
                raise ValueError("VelocityTask currently supports the shared 1 N illegal-contact termination.")
        if len(owner.action_manager.active_terms) != 1:
            raise ValueError("VelocityTask requires one joint-position action term.")
        action = owner.action_manager.get_term(owner.action_manager.active_terms[0])
        if type(action) is not JointPositionAction or action.cfg.clip is not None:
            raise ValueError("VelocityTask requires an unclipped JointPositionAction.")
        self.action_joints = action._joint_ids
        self.action_scale = torch.as_tensor(action._scale, device=device)
        self.action_offset = torch.as_tensor(action._offset, device=device)
        if self.action_scale.ndim > 1:
            self.action_scale = self.action_scale[0].clone()
        if self.action_offset.ndim > 1:
            self.action_offset = self.action_offset[0].clone()
        self.stiffness = torch.zeros(self.num_joints, device=device)
        self.damping = torch.zeros_like(self.stiffness)
        self.effort_limit = torch.full_like(self.stiffness, torch.inf)
        for cfg in robot.cfg.actuators.values():
            if not isinstance(cfg, (IdealPDActuatorCfg, ImplicitActuatorCfg)):
                raise ValueError("VelocityTask's torque approximation requires PD actuator gains.")
            ids, names = robot.find_joints(cfg.joint_names_expr)
            for parameter in ("stiffness", "damping"):
                values = resolve_joint_parameter(getattr(cfg, parameter), None, names, 1, device)
                getattr(self, parameter)[ids] = values[0]
            limit = cfg.actuator_effort_limit if cfg.actuator_effort_limit is not None else cfg.joint_effort_limit
            self.effort_limit[ids] = resolve_joint_parameter(limit, torch.inf, names, 1, device)[0]

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Return only the default policy's deployable proprioceptive inputs."""
        gravity = quat_apply_inverse(state[2], self.gravity.expand_as(state[0]))
        parts = (
            state[0],
            state[1],
            gravity,
            state[4] - self.default_pos,
            state[5] - self.default_vel,
            prev_actions[..., -1, :],
        )
        return torch.cat(tuple(parts[index] for index, _, _ in self.actor_terms), dim=-1)

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Use the commanded xy velocity and yaw rate directly."""
        return goal

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Match the flat policy's independent uniform observation corruption."""
        if not self.enable_corruption:
            return obs
        parts = obs.split(tuple(width for _, width, _ in self.actor_terms), dim=-1)
        return torch.cat(
            tuple(
                term.noise.func(part, term.noise) if term.noise is not None else part
                for part, (_, _, term) in zip(parts, self.actor_terms)
            ),
            dim=-1,
        )

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Condition dynamics on log-scaled kg masses and raw recent actions.

        The fixed ``log1p(mass / 1 kg)`` scale is independent of the randomized
        environment batch, so a fresh checkpoint load preserves its meaning.
        """
        return torch.cat((torch.log1p(context), prev_actions.flatten(-2)), dim=-1)

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Allow the critic privileged physical state and mass information."""
        return torch.cat(
            (self.representation.features(state), goal, self.dynamics_context(state, context, prev_actions)), dim=-1
        )

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold the command over the short imagined horizon."""
        return goal, context

    def post_dynamics(self, state: State, next_state: State, labels: torch.Tensor) -> State:
        """Advance contact clocks from soft contact events at policy resolution."""
        contact = labels[..., : self.num_feet]
        first = labels[..., self.num_feet + self.num_bodies :]
        air = (1 - contact) * (state[7] + self.dt)
        stance = contact * (state[8] + self.dt)
        last_air = first * state[7] + (1 - first) * state[9]
        return next_state[:7] + (air, stance, last_air)

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Use predicted force-history contacts at the configured illegal bodies."""
        if labels is None:
            raise ValueError("Velocity termination needs predicted or observed contact labels.")
        survival = torch.ones_like(next_state[0][..., 0])
        contacts = labels[..., self.num_feet : self.num_feet + self.num_bodies]
        for cfg in self.terminations.values():
            ids = cfg.params["sensor_cfg"].body_ids
            survival = survival * (1 - contacts[..., ids]).prod(-1)
        return 1 - survival

    def reward_terms(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute weighted, dt-scaled terms from the resolved RewardManager cfg.

        ``joint_torques_l2`` approximates motor dynamics with clipped PD torque;
        ``joint_acc_l2`` uses a policy-rate finite difference rather than the last
        physics-substep acceleration. Contacts/timers are exact with observed
        labels/state and approximated by the world model during imagination.
        """
        if labels is None:
            raise ValueError("Velocity rewards need predicted or observed contact labels.")
        lin, ang, orientation, _, q, dq = next_state[:6]
        gravity = quat_apply_inverse(orientation, self.gravity.expand_as(lin))
        contacts = labels[..., self.num_feet : self.num_feet + self.num_bodies]
        results = {}
        for name, cfg in self.rewards.items():
            kind, params = cfg.func.__name__, cfg.params
            ids = params["asset_cfg"].joint_ids if "asset_cfg" in params else slice(None)
            value = torch.zeros_like(lin[..., 0])
            if cfg.weight == 0:
                results[name] = value
                continue
            if kind in ("track_lin_vel_xy_exp", "track_lin_vel_xy_yaw_frame_exp"):
                velocity = (
                    lin
                    if kind == "track_lin_vel_xy_exp"
                    else quat_apply_inverse(yaw_quat(orientation), quat_apply(orientation, lin))
                )
                value = torch.exp(-(goal[..., :2] - velocity[..., :2]).square().sum(-1) / params["std"] ** 2)
            elif kind in ("track_ang_vel_z_exp", "track_ang_vel_z_world_exp"):
                velocity = ang if kind == "track_ang_vel_z_exp" else quat_apply(orientation, ang)
                value = torch.exp(-(goal[..., 2] - velocity[..., 2]).square() / params["std"] ** 2)
            elif kind == "lin_vel_z_l2":
                value = lin[..., 2].square()
            elif kind == "ang_vel_xy_l2":
                value = ang[..., :2].square().sum(-1)
            elif kind == "flat_orientation_l2":
                value = gravity[..., :2].square().sum(-1)
            elif kind == "joint_torques_l2":
                target = self.default_pos.expand_as(q).clone()
                target[..., self.action_joints] = actions * self.action_scale + self.action_offset
                torque = (self.stiffness * (target - q) - self.damping * dq).clamp(
                    -self.effort_limit, self.effort_limit
                )
                value = torque[..., ids].square().sum(-1)
            elif kind == "joint_acc_l2":
                value = ((dq - state[5]) / self.dt)[..., ids].square().sum(-1)
            elif kind == "action_rate_l2":
                value = (actions - prev_actions[..., -1, :]).square().sum(-1)
            elif kind == "joint_deviation_l1":
                value = (q - self.default_pos)[..., ids].abs().sum(-1)
            elif kind == "joint_pos_limits":
                value = ((self.limits[..., 0] - q).relu() + (q - self.limits[..., 1]).relu())[..., ids].sum(-1)
            elif kind == "feet_slide":
                foot_velocity = next_state[6].reshape(*lin.shape[:-1], self.num_feet, 3)[..., :2]
                value = (foot_velocity.norm(dim=-1) * contacts[..., self.sensor_foot_ids]).sum(-1)
            elif kind == "feet_air_time_positive_biped":
                contact = labels[..., : self.num_feet]
                # Probability of exactly one stance foot, valid for soft labels.
                single = sum(
                    contact[..., i] * torch.cat((1 - contact[..., :i], 1 - contact[..., i + 1 :]), -1).prod(-1)
                    for i in range(self.num_feet)
                )
                in_mode = contact * next_state[8] + (1 - contact) * next_state[7]
                value = in_mode.amin(-1).clamp(max=params["threshold"]) * single
                value = value * (goal[..., :2].norm(dim=-1) > 0.1)
            elif kind == "feet_air_time":
                first = labels[..., self.num_feet + self.num_bodies :]
                value = ((next_state[9] - params["threshold"]) * first).sum(-1)
                value = value * (goal[..., :2].norm(dim=-1) > 0.1)
            elif kind == "is_terminated":
                value = self.termination_prob(next_state, goal, labels)
            results[name] = value * cfg.weight * self.dt
        return results

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum the physical MDP's weighted reward terms."""
        return torch.stack(
            tuple(self.reward_terms(state, actions, next_state, goal, prev_actions, labels).values()), -1
        ).sum(-1)
