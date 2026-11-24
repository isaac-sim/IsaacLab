# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable heads for the shared velocity MDP on flat and rough terrain.

Reward weights, entity selections, actuator gains and action mappings come from
the resolved robot environment. Contact heads predict sensor events; they do not
expose privileged contact or mass inputs to the actor. Torque uses a clipped PD
approximation, and joint acceleration uses policy-rate velocity differences.
Feet and the height scanner follow from differentiable forward kinematics, and
terrain heights from a heightfield of the terrain mesh.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from newton import GeoType, ShapeFlags
from rsl_rl.env.differentiable_task import DifferentiableTask, State
from tensordict import TensorDict

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from isaaclab_tasks.core.motion_tracking.supermimic import ArticulationKinematics

from .mdp.observations import _feet

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

HEIGHTFIELD_RESOLUTION = 0.025
"""Grid spacing [m] of the terrain heightfield; interpolation blurs vertical steps over one cell."""


def _index(ids: slice | list[int] | torch.Tensor, device: str) -> slice | torch.Tensor:
    """Return a slice unchanged and other indices as a device tensor, which CUDA graph capture can replay."""
    return ids if isinstance(ids, slice) else torch.as_tensor(ids, dtype=torch.long, device=device)


def _product(values: torch.Tensor) -> torch.Tensor:
    """Multiply along the last dimension; unlike ``prod``, its backward needs no host synchronization."""
    result = values[..., 0]
    for index in range(1, values.shape[-1]):
        result = result * values[..., index]
    return result


def _transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a Newton ``(x, y, z, qx, qy, qz, qw)`` transform to points."""
    pose = torch.as_tensor(transform, dtype=torch.float64)
    points = torch.as_tensor(points, dtype=torch.float64)
    return (quat_apply(pose[3:].expand(len(points), 4), points) + pose[:3]).numpy()


class Heightfield:
    """Terrain heights on a regular grid with a bilinear lookup that is differentiable in ``x`` and ``y``.

    Lookups clamp to the grid and reflect ``y`` about :attr:`mirror_y`, the upper ``y`` edge of the grid, so a world
    mirrored about that line sees the mirrored terrain.
    """

    def __init__(self, heights: torch.Tensor, lower: tuple[float, float], resolution: float) -> None:
        self.shape = tuple(heights.shape)
        self.heights = heights.flatten()
        self.lower, self.resolution = lower, resolution
        self.mirror_y = lower[1] + (self.shape[1] - 1) * resolution

    @classmethod
    def from_model(cls, model, device: str) -> Heightfield:
        """Ray-cast the static ground of a Newton model once on a grid: horizontal planes and at most one terrain mesh.

        Where both exist, the ground is the higher of the two.
        """
        static = np.flatnonzero(model.shape_body.numpy() == -1)
        kinds, transforms = model.shape_type.numpy()[static], model.shape_transform.numpy()[static]
        planes = [float(t[2]) for k, t in zip(kinds, transforms) if k == GeoType.PLANE]
        meshes = [i for i, k in zip(static, kinds) if k == GeoType.MESH]
        if len(planes) + len(meshes) != len(static) or len(meshes) > 1 or not static.size:
            raise ValueError("VelocityTask expects ground planes and at most one terrain mesh.")
        if not meshes:
            return cls(torch.full((2, 2), max(planes), device=device), (-1.0, -1.0), 1.0)
        (shape,) = meshes
        transform = model.shape_transform.numpy()[shape]
        source = model.shape_source[shape]
        points = _transform(transform, source.vertices * model.shape_scale.numpy()[shape])
        mesh = convert_to_warp_mesh(points, source.indices, device)
        lower, upper = points.min(0).tolist(), points.max(0).tolist()
        x, y = (
            torch.arange(lower[axis], upper[axis] + HEIGHTFIELD_RESOLUTION, HEIGHTFIELD_RESOLUTION, device=device)
            for axis in (0, 1)
        )
        top = torch.tensor(upper[2] + 1.0, device=device)
        rows = []
        for chunk in x.split(256):
            starts = torch.stack(torch.broadcast_tensors(chunk[:, None], y, top), -1)
            down = torch.zeros_like(starts)
            down[..., 2] = -1.0
            rows.append(raycast_mesh(starts, down, mesh)[0][..., 2])
        heights = torch.cat(rows)
        if planes:
            heights = torch.where(torch.isfinite(heights), heights.clamp(min=max(planes)), max(planes))
        return cls(heights, (lower[0], lower[1]), HEIGHTFIELD_RESOLUTION)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the terrain height [m] at world coordinates of any shared shape."""
        rows, cols = self.shape
        y = torch.minimum(y, 2 * self.mirror_y - y)
        u = ((x - self.lower[0]) / self.resolution).clamp(0, rows - 1)
        v = ((y - self.lower[1]) / self.resolution).clamp(0, cols - 1)
        i, j = u.detach().floor().clamp(max=rows - 2), v.detach().floor().clamp(max=cols - 2)
        index = i.long() * cols + j.long()
        low, high = (torch.lerp(self.heights[index + k], self.heights[index + k + cols], u - i) for k in (0, 1))
        return torch.lerp(low, high, v - j)


def _sole_points(model, body_name: str) -> torch.Tensor:
    """Return the lowest collision-shape vertices of a body in its frame, e.g. the four sole corners of a box foot."""
    body = [label.rsplit("/", 1)[-1] for label in model.body_label].index(body_name)
    points = []
    for shape in np.flatnonzero(model.shape_body.numpy() == body):
        if not model.shape_flags.numpy()[shape] & ShapeFlags.COLLIDE_SHAPES:
            continue
        source = model.shape_source[shape]
        if source is None:
            raise ValueError(f"The collision shapes of '{body_name}' must be meshes.")
        vertices = source.vertices * model.shape_scale.numpy()[shape]
        points.append(_transform(model.shape_transform.numpy()[shape], vertices))
    points = np.concatenate(points)
    return torch.tensor(points[points[:, 2] < points[:, 2].min() + 1e-3], dtype=torch.float32)


class VelocityStateRepresentation:
    """Floating-base coordinates with world-frame SO(3) tangent increments.

    The world position stays out of the dynamics features, which see the terrain
    only through task-owned terrain heights. Contact clocks are observed features,
    carried through integration and then advanced analytically by
    :meth:`VelocityTask.post_dynamics`. They are never learned targets. All
    methods preserve arbitrary leading batch dimensions.
    """

    def __init__(self, num_joints: int, num_feet: int, device: str = "cpu") -> None:
        self.widths = (3, 3, 4, 3, num_joints, num_joints, num_feet, num_feet, num_feet)
        self.increment_widths = (3, 3, 3, 3, num_joints, num_joints)
        self.increment_dim = sum(self.increment_widths)
        self.error_floor = torch.full((self.increment_dim,), 0.01, device=device)

    def unflatten(self, flat: torch.Tensor) -> State:
        """Decode the noise-free ``state`` observation group."""
        return flat.split(self.widths, dim=-1)

    def features(self, state: State) -> torch.Tensor:
        """Encode velocities, orientation, joints and observed contact clocks."""
        return torch.cat((*state[:3], *state[4:]), dim=-1)

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Encode position, velocity and joint differences and the shortest rotation delta."""
        rotation = quat_mul(next_state[2], quat_conjugate(state[2]))
        rotation = torch.where(rotation[..., 3:] < 0, -rotation, rotation)
        magnitude = torch.linalg.vector_norm(rotation[..., :3], dim=-1, keepdim=True)
        # The clamped denominator preserves finite derivatives at zero rotation.
        scale = 2 * torch.atan2(magnitude, rotation[..., 3:]) / magnitude.clamp_min(1e-8)
        tangent = rotation[..., :3] * torch.where(magnitude > 1e-6, scale, 2.0)
        return torch.cat(tuple(tangent if i == 2 else next_state[i] - state[i] for i in range(6)), dim=-1)

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Integrate learned coordinates while carrying the analytic clocks."""
        parts = increment.split(self.increment_widths, dim=-1)
        angle = torch.linalg.vector_norm(parts[2], dim=-1, keepdim=True)
        delta = torch.cat((0.5 * torch.sinc(angle / (2 * math.pi)) * parts[2], torch.cos(angle / 2)), dim=-1)
        orientation = normalize(quat_mul(delta, state[2]))
        return tuple(orientation if i == 2 else state[i] + parts[i] for i in range(6)) + state[6:]

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Measure learned-coordinate error, excluding the analytic clocks."""
        return self.increments(target, prediction)


class VelocityTask(DifferentiableTask):
    """Robot-parameterized differentiable task for the shared velocity MDP.

    Commands are held over an imagined horizon; real command resampling and
    heading control are read from the next physical ``goal`` observation.
    Analytic contact clocks use policy-step resolution, approximating contact
    changes within physics substeps. Real reward parity uses observed clocks.
    A height scan samples the terrain heightfield at the scanner's ray starts.
    """

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu") -> None:
        owner = env.unwrapped
        robot = owner.scene["robot"]
        data = robot.data
        model = owner.sim.backend_manager.backend.model
        self.device = device
        self.dt = owner.step_dt
        self.num_joints = data.joint_pos.shape[-1]
        self.num_actions = owner.action_manager.total_action_dim
        self.foot_ids, self.sensor_foot_ids = _feet(owner)
        self.num_feet = len(self.foot_ids)
        sensor_name = owner.cfg.rewards.feet_air_time.params["sensor_cfg"].name
        self.num_bodies = owner.scene.sensors[sensor_name].num_sensors
        self.num_labels = self.num_bodies + 2 * self.num_feet
        self.representation = VelocityStateRepresentation(self.num_joints, self.num_feet, device)
        self.default_pos = data.default_joint_pos.torch[0].to(device).clone()
        self.default_vel = data.default_joint_vel.torch[0].to(device).clone()
        self.limits = data.soft_joint_pos_limits.torch[0].to(device).clone()
        self.root_com = data.body_com_pos_b.torch[0, 0].to(device).clone()
        self.terrain = Heightfield.from_model(model, device)
        bodies = [robot.body_names[index] for index in self.foot_ids]
        self.sole_points = torch.stack([_sole_points(model, name) for name in bodies]).to(device)
        policy_cfg = owner.cfg.observations.policy
        terms = [
            ("base_lin_vel", 3),
            ("base_ang_vel", 3),
            ("projected_gravity", 3),
            ("velocity_commands", 3),
            ("joint_pos", self.num_joints),
            ("joint_vel", self.num_joints),
            ("actions", self.num_actions),
        ]
        scan_cfg, self.scan_points = policy_cfg.height_scan, None
        if scan_cfg is not None:
            scanner = owner.scene.sensors[scan_cfg.params["sensor_cfg"].name]
            scanner_body = scanner.cfg.prim_path.rsplit("/", 1)[-1]
            if scanner.cfg.ray_alignment != "yaw" or scanner_body not in robot.body_names:
                raise ValueError("VelocityTask requires a yaw-aligned height scanner on a robot body.")
            # local ray starts include the scanner offset; downward rays hit the terrain below them
            self.scan_points = scanner.ray_starts.torch[0, :, :2].to(device).clone()
            self.scan_offset = scan_cfg.params.get("offset", 0.5)
            terms.append(("height_scan", len(self.scan_points)))
            bodies.append(scanner_body)
        self.kinematics = ArticulationKinematics(owner, "robot", device, body_names=bodies)
        self._cached_state, self._cached_bodies = None, None
        self.actor_terms = [
            (index, width, getattr(policy_cfg, name))
            for index, (name, width) in enumerate(terms)
            if getattr(policy_cfg, name) is not None
        ]
        self.enable_corruption = policy_cfg.enable_corruption
        for _, _, term in self.actor_terms:
            if term.scale is not None:
                raise ValueError("VelocityTask currently requires unscaled policy observation terms.")
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
        self.action_joints = _index(action._joint_ids, device)
        self.sensor_feet = _index(self.sensor_foot_ids, device)
        self.term_joints = {
            name: _index(cfg.params["asset_cfg"].joint_ids if "asset_cfg" in cfg.params else slice(None), device)
            for name, cfg in self.rewards.items()
        }
        self.termination_bodies = {
            name: _index(cfg.params["sensor_cfg"].body_ids, device) for name, cfg in self.terminations.items()
        }
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

    def _bodies(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feet and scanner poses and velocities, reused while the same state is passed to several heads."""
        if self._cached_state is not state[0]:
            lin, ang, orientation, position, q, dq = state[:6]
            ang = quat_apply(orientation, ang)
            # the state holds the root center-of-mass velocity; kinematics starts from the root link origin
            offset = quat_apply(orientation, self.root_com.expand_as(lin))
            lin = quat_apply(orientation, lin) - torch.cross(ang, offset, dim=-1)
            self._cached_state = state[0]
            self._cached_bodies = self.kinematics.forward((position, orientation, lin, ang, q, dq))
        return self._cached_bodies

    def _height_scan(self, state: State) -> torch.Tensor:
        """Scanner height above the terrain at the yaw-rotated ray starts, minus the observation offset."""
        position, orientation = (value[..., -1, :] for value in self._bodies(state)[:2])
        x, y, z, w = orientation.unbind(-1)
        heading = normalize(torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), -1))
        cos, sin = heading[..., :1], heading[..., 1:]
        ray_x, ray_y = self.scan_points.unbind(-1)
        heights = self.terrain(
            position[..., :1] + cos * ray_x - sin * ray_y, position[..., 1:2] + sin * ray_x + cos * ray_y
        )
        return position[..., 2:] - heights - self.scan_offset

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> TensorDict:
        """Return the default policy's deployable inputs, including the command."""
        gravity = quat_apply_inverse(state[2], self.gravity.expand_as(state[0]))
        parts = [
            state[0],
            state[1],
            gravity,
            goal,
            state[4] - self.default_pos,
            state[5] - self.default_vel,
            prev_actions[..., -1, :],
        ]
        if self.scan_points is not None:
            parts.append(self._height_scan(state))
        frame = torch.cat(
            tuple(
                parts[index] if term.clip is None else parts[index].clamp(*term.clip)
                for index, _, term in self.actor_terms
            ),
            dim=-1,
        )
        return TensorDict({"policy": frame}, batch_size=frame.shape[:-1])

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Match the policy's independent uniform observation corruption, clipping noisy terms again.

        The policy clips after adding noise, so clipped terms differ from it only within the noise amplitude of
        the clip bounds.
        """
        if not self.enable_corruption:
            return obs
        parts = obs.split(tuple(width for _, width, _ in self.actor_terms), dim=-1)
        noisy = []
        for part, (_, _, term) in zip(parts, self.actor_terms):
            if term.noise is not None:
                part = term.noise.func(part, term.noise)
                if term.clip is not None:
                    part = part.clamp(*term.clip)
            noisy.append(part)
        return torch.cat(noisy, dim=-1)

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Condition dynamics on log-scaled kg masses, raw recent actions and the terrain below the robot.

        The fixed ``log1p(mass / 1 kg)`` scale is independent of the randomized
        environment batch, so a fresh checkpoint load preserves its meaning. The
        terrain enters as the root height and the sole-point heights above it.
        """
        position, orientation = (value[..., : self.num_feet, :] for value in self._bodies(state)[:2])
        shape = (*orientation.shape[:-1], len(self.sole_points[0]), 3)
        soles = position.unsqueeze(-2) + quat_apply(
            orientation.unsqueeze(-2).expand(*shape[:-1], 4), self.sole_points.expand(shape)
        )
        clearance = soles[..., 2] - self.terrain(soles[..., 0], soles[..., 1])
        root = state[3]
        height = root[..., 2:] - self.terrain(root[..., :1], root[..., 1:2])
        return torch.cat((torch.log1p(context), prev_actions.flatten(-2), height, clearance.flatten(-2)), dim=-1)

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        """Allow the critic privileged physical state, mass and terrain information."""
        values = [self.representation.features(state), goal, self.dynamics_context(state, context, prev_actions)]
        if self.scan_points is not None:
            values.append(self._height_scan(state))
        return torch.cat(values, dim=-1)

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold the command over the short imagined horizon."""
        return goal, context

    def post_dynamics(
        self, state: State, next_state: State, labels: torch.Tensor, actions: torch.Tensor, context: torch.Tensor
    ) -> State:
        """Advance contact clocks from soft contact events at policy resolution."""
        contact = labels[..., : self.num_feet]
        first = labels[..., self.num_feet + self.num_bodies :]
        air = (1 - contact) * (state[6] + self.dt)
        stance = contact * (state[7] + self.dt)
        last_air = first * state[6] + (1 - first) * state[8]
        return next_state[:6] + (air, stance, last_air)

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Use predicted force-history contacts at the configured illegal bodies."""
        if labels is None:
            raise ValueError("Velocity termination needs predicted or observed contact labels.")
        survival = torch.ones_like(next_state[0][..., 0])
        contacts = labels[..., self.num_feet : self.num_feet + self.num_bodies]
        for ids in self.termination_bodies.values():
            survival = survival * _product(1 - contacts[..., ids])
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
            ids = self.term_joints[name]
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
                foot_velocity = self._bodies(next_state)[2][..., : self.num_feet, :2]
                value = (foot_velocity.norm(dim=-1) * contacts[..., self.sensor_feet]).sum(-1)
            elif kind == "feet_air_time_positive_biped":
                contact = labels[..., : self.num_feet]
                # Probability of exactly one stance foot, valid for soft labels.
                single = sum(
                    contact[..., i] * _product(torch.cat((1 - contact[..., :i], 1 - contact[..., i + 1 :]), -1))
                    for i in range(self.num_feet)
                )
                in_mode = contact * next_state[7] + (1 - contact) * next_state[6]
                value = in_mode.amin(-1).clamp(max=params["threshold"]) * single
                value = value * (goal[..., :2].norm(dim=-1) > 0.1)
            elif kind == "feet_air_time":
                first = labels[..., self.num_feet + self.num_bodies :]
                value = ((next_state[8] - params["threshold"]) * first).sum(-1)
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
