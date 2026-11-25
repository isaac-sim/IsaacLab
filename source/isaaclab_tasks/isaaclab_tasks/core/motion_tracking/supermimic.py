# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable heads for the shared motion tracking MDP.

The world model predicts root and joint velocity increments. Poses are integrated analytically.
Tracked-body states follow from differentiable forward kinematics, so rewards and terminations are analytic
functions of that state.
Reward weights, kernel widths, tracked bodies and the reference motion come from the resolved environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from newton import JointType
from rsl_rl.env.differentiable_task import DifferentiableTask, State
from tensordict import TensorDict

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.math import (
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _rotation_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Return the shortest rotation vector of XYZW quaternions with finite derivatives at zero rotation."""
    rotation = torch.where(rotation[..., 3:] < 0, -rotation, rotation)
    magnitude = torch.linalg.vector_norm(rotation[..., :3], dim=-1, keepdim=True)
    scale = 2 * torch.atan2(magnitude, rotation[..., 3:]) / magnitude.clamp_min(1e-8)
    return rotation[..., :3] * torch.where(magnitude > 1e-6, scale, 2.0)


def _angle_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared smallest rotation angle between XYZW quaternions, equal to ``quat_error_magnitude`` squared."""
    return _rotation_vector(quat_mul(a, quat_conjugate(b))).square().sum(-1)


def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply broadcastable quaternions."""
    shape = torch.broadcast_shapes(a.shape, b.shape)
    return quat_mul(a.expand(shape), b.expand(shape))


def _apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate broadcastable vectors."""
    shape = torch.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    return quat_apply(quat.expand(*shape, 4), vec.expand(*shape, 3))


class ArticulationKinematics:
    """Differentiable forward kinematics of a floating-base articulation with revolute joints.

    The tree is read from the Newton model of the simulated asset. Joint frames must coincide with their child link
    frames. Outputs follow the asset's body order; linear velocities are taken at each body's center of mass, as
    reported by ``body_lin_vel_w``.
    """

    def __init__(self, env: ManagerBasedRLEnv, asset_name: str, device: str) -> None:
        robot = env.scene[asset_name]
        model = env.sim.backend_manager.backend.model
        articulation = int(robot._root_view.articulation_ids.numpy().flat[0])
        start, end = (
            int(values.numpy()[articulation]) for values in (model.articulation_start, model.articulation_end)
        )
        joint_type, parent, child = (
            values.numpy() for values in (model.joint_type, model.joint_parent, model.joint_child)
        )
        offsets, child_frames = model.joint_X_p.numpy(), model.joint_X_c.numpy()
        axes, dof_start = model.joint_axis.numpy(), model.joint_qd_start.numpy()
        body_of = {
            int(index): robot.body_names.index(label.rsplit("/", 1)[-1])
            for index, label in enumerate(model.body_label)
            if label.rsplit("/", 1)[-1] in robot.body_names
        }
        self.num_bodies = len(robot.body_names)
        self.root = None
        links = []
        for joint in range(start, end):
            name = model.joint_label[joint].rsplit("/", 1)[-1]
            if parent[joint] < 0:
                self.root = body_of[int(child[joint])]
                continue
            if joint_type[joint] != JointType.REVOLUTE or abs(child_frames[joint] - [0, 0, 0, 0, 0, 0, 1]).max() > 1e-6:
                raise ValueError(f"Joint '{name}' must be revolute with its frame at the child link origin.")
            links.append(
                (
                    body_of[int(child[joint])],
                    body_of[int(parent[joint])],
                    robot.joint_names.index(name),
                    offsets[joint].tolist(),
                    axes[dof_start[joint]].tolist(),
                )
            )
        # group links by depth; slots number bodies in the order they are computed, starting with the root
        slot, self.levels = {self.root: 0}, []
        while links:
            ready = [link for link in links if link[1] in slot]
            if not ready:
                raise ValueError("The articulation is not a tree rooted at its floating base.")
            self.levels.append(
                {
                    "parent": torch.tensor([slot[link[1]] for link in ready], device=device),
                    "joint": torch.tensor([link[2] for link in ready], device=device),
                    "offset_pos": torch.tensor([link[3][:3] for link in ready], device=device),
                    "offset_quat": torch.tensor([link[3][3:] for link in ready], device=device),
                    "axis": torch.tensor([link[4] for link in ready], device=device),
                }
            )
            slot.update({link[0]: len(slot) + i for i, link in enumerate(ready)})
            links = [link for link in links if link[0] not in slot]
        if len(slot) != self.num_bodies:
            raise ValueError("Every body of the articulation must be connected to its floating base.")
        self.slot_of_body = torch.tensor([slot[body] for body in range(self.num_bodies)], device=device)
        self.com = robot.data.body_com_pos_b.torch[0].to(device).clone()

    def to(self, device: torch.device | str) -> ArticulationKinematics:
        """Move the tree constants."""
        self.levels = [{name: value.to(device) for name, value in level.items()} for level in self.levels]
        self.slot_of_body, self.com = self.slot_of_body.to(device), self.com.to(device)
        return self

    def forward(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return world positions, orientations, center-of-mass linear velocities and angular velocities.

        Shapes are ``[..., num_bodies, 3]`` and ``[..., num_bodies, 4]``; positions share the frame of the root.
        """
        root_pos, root_quat, root_lin, root_ang, q, dq = state
        pos, quat, lin, ang = (value.unsqueeze(-2) for value in (root_pos, root_quat, root_lin, root_ang))
        for level in self.levels:
            parent = level["parent"]
            par_pos, par_quat, par_lin, par_ang = (value.index_select(-2, parent) for value in (pos, quat, lin, ang))
            angle, rate = q.index_select(-1, level["joint"]), dq.index_select(-1, level["joint"])
            frame = _mul(par_quat, level["offset_quat"])
            child_pos = par_pos + _apply(par_quat, level["offset_pos"])
            child_quat = _mul(frame, quat_from_angle_axis(angle, level["axis"].expand(*angle.shape, 3)))
            child_ang = par_ang + _apply(frame, level["axis"]) * rate.unsqueeze(-1)
            child_lin = par_lin + torch.cross(par_ang, child_pos - par_pos, dim=-1)
            pos, quat, lin, ang = (
                torch.cat((value, child), -2)
                for value, child in ((pos, child_pos), (quat, child_quat), (lin, child_lin), (ang, child_ang))
            )
        pos, quat, lin, ang = (value.index_select(-2, self.slot_of_body) for value in (pos, quat, lin, ang))
        return pos, quat, lin + torch.cross(ang, _apply(quat, self.com), dim=-1), ang


class MotionStateRepresentation:
    """Joint-space dynamics with root-frame velocity increments and analytic pose integration.

    State tuples contain root position, XYZW orientation, world linear/angular velocity,
    joint positions and joint velocities. For 29 joints the network consumes 68 features
    and predicts 35 velocity increments; root and joint poses are integrated, not predicted.
    """

    def __init__(self, num_joints: int, dt: float, decimation: int, device: str = "cpu") -> None:
        self.widths = (3, 4, 3, 3, num_joints, num_joints)
        self.increment_dim = 6 + num_joints
        self.error_floor = torch.full((self.increment_dim,), 0.01, device=device)
        self.dt = dt
        self.substep_coeff = (decimation + 1) / (2 * decimation)

    def unflatten(self, flat: torch.Tensor) -> State:
        """Decode the noise-free ``state`` observation group."""
        return flat.split(self.widths, dim=-1)

    def features(self, state: State) -> torch.Tensor:
        """Encode height, projected gravity, local root velocities and joint state."""
        pos, quat, lin, ang, q, dq = state
        gravity = torch.zeros_like(lin)
        gravity[..., 2] = -1
        return torch.cat(
            (
                pos[..., 2:],
                quat_apply_inverse(quat, gravity),
                quat_apply_inverse(quat, lin),
                quat_apply_inverse(quat, ang),
                q,
                dq,
            ),
            -1,
        )

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Express successor velocity changes in the current root frame."""
        return torch.cat(
            (
                quat_apply_inverse(state[1], next_state[2] - state[2]),
                quat_apply_inverse(state[1], next_state[3] - state[3]),
                next_state[5] - state[5],
            ),
            -1,
        )

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Compose constant-acceleration substeps and normalize the integrated orientation."""
        pos, quat, lin, ang, q, dq = state
        dv, dw, ddq = increment.split((3, 3, self.widths[-1]), dim=-1)
        local_lin, local_ang = quat_apply_inverse(quat, lin), quat_apply_inverse(quat, ang)
        c = self.substep_coeff
        next_pos = pos + quat_apply(quat, local_lin + c * dv) * self.dt
        omega = quat_apply(quat, local_ang + c * dw) * self.dt
        magnitude = torch.sqrt(omega.square().sum(-1, keepdim=True) + 1e-8)
        delta = torch.cat((torch.sin(magnitude / 2) * (omega / magnitude), torch.cos(magnitude / 2)), -1)
        next_quat = normalize(quat_mul(delta, quat))
        return (
            next_pos,
            next_quat,
            quat_apply(quat, local_lin + dv),
            quat_apply(quat, local_ang + dw),
            q + (dq + c * ddq) * self.dt,
            dq + ddq,
        )

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Measure velocity errors in the target root frame."""
        return self.increments(target, prediction)


class MotionTrackingTask(DifferentiableTask):
    """Differentiable task for the shared motion tracking MDP.

    The goal is the index of the current reference frame; it advances by one frame per imagined step. The actor
    receives the default policy's proprioceptive terms and reference-anchor orientation per frame, and the reference
    joint command as its goal. Terminations use sigmoids around the environment's thresholds.
    """

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu") -> None:
        owner = env.unwrapped
        robot = owner.scene["robot"]
        data = robot.data
        self.device = device
        self.dt = owner.step_dt
        self.num_joints = data.joint_pos.shape[-1]
        self.num_actions = owner.action_manager.total_action_dim
        self.representation = MotionStateRepresentation(self.num_joints, self.dt, owner.cfg.decimation, device)
        self.kinematics = ArticulationKinematics(owner, "robot", device)
        self._cached_state, self._cached_bodies = None, None
        self.rewards = {name: owner.reward_manager.get_term_cfg(name) for name in owner.reward_manager.active_terms}
        self.terminations = {
            name: owner.termination_manager.get_term_cfg(name)
            for name in owner.termination_manager.active_terms
            if not owner.termination_manager.get_term_cfg(name).time_out
        }
        supported_rewards = {
            "motion_global_anchor_position_error_exp",
            "motion_global_anchor_orientation_error_exp",
            "motion_relative_body_position_error_exp",
            "motion_relative_body_orientation_error_exp",
            "motion_global_body_linear_velocity_error_exp",
            "motion_global_body_angular_velocity_error_exp",
            "action_rate_l2",
            "joint_pos_limits",
            "self_collision_cost",
        }
        supported_terminations = {"bad_anchor_pos_z_only", "bad_anchor_ori", "bad_motion_body_pos_z_only"}
        for name, cfg in self.rewards.items():
            if cfg.weight and cfg.func.__name__ not in supported_rewards:
                raise ValueError(f"Unsupported differentiable motion tracking reward: {name} ({cfg.func.__name__}).")
        for name, cfg in self.terminations.items():
            if cfg.func.__name__ not in supported_terminations:
                raise ValueError(f"Unsupported differentiable motion tracking termination: {name}.")
        names = {cfg.params.get("command_name") for cfg in (*self.rewards.values(), *self.terminations.values())} - {
            None
        }
        if len(names) != 1:
            raise ValueError("Motion tracking terms must share one motion command.")
        command = owner.command_manager.get_term(names.pop())
        motion = command.motion
        self.last_frame = motion.time_step_total - 1
        self.clip_ends = motion.motion_ends.to(device)
        self.ref_joint_pos = motion.joint_pos.to(device)
        self.ref_joint_vel = motion.joint_vel.to(device)
        self.ref_body_pos = motion.body_pos_w.to(device)
        self.ref_body_quat = motion.body_quat_w.to(device)
        self.ref_body_lin = motion.body_lin_vel_w.to(device)
        self.ref_body_ang = motion.body_ang_vel_w.to(device)
        self.tracked = command.body_indices.to(device)
        self.robot_anchor = command.robot_anchor_body_index
        self.ref_anchor = command.motion_anchor_body_index
        body_names = list(command.cfg.body_names)
        self.termination_bodies = {
            name: torch.tensor([body_names.index(body) for body in cfg.params["body_names"]], device=device)
            for name, cfg in self.terminations.items()
            if "body_names" in cfg.params
        }
        self.limit_joints = {
            name: torch.as_tensor(range(self.num_joints) if isinstance(ids, slice) else ids, device=device)
            for name, cfg in self.rewards.items()
            if cfg.func.__name__ == "joint_pos_limits"
            for ids in (cfg.params["asset_cfg"].joint_ids,)
        }
        self.default_pos = data.default_joint_pos.torch[0].to(device).clone()
        self.default_vel = data.default_joint_vel.torch[0].to(device).clone()
        self.limits = data.soft_joint_pos_limits.torch[0].to(device).clone()
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)
        action_terms = owner.action_manager.active_terms
        action = owner.action_manager.get_term(action_terms[0]) if len(action_terms) == 1 else None
        if not isinstance(action, JointPositionAction):
            raise ValueError("MotionTrackingTask requires one joint-position action term.")
        joint_ids = action._joint_ids
        self.action_joints = torch.as_tensor(
            range(self.num_joints) if isinstance(joint_ids, slice) else joint_ids, device=device
        )
        self.action_scale = torch.as_tensor(action._scale, dtype=torch.float32, device=device).reshape(-1)[
            : self.num_actions
        ]
        self.action_offset = torch.as_tensor(action._offset, dtype=torch.float32, device=device).reshape(-1)[
            : self.num_actions
        ]
        policy_cfg = owner.cfg.observations.policy
        terms = (
            ("motion_anchor_ori_b", 6),
            ("base_ang_vel", 3),
            ("joint_pos", self.num_joints),
            ("joint_vel", self.num_joints),
            ("actions", self.num_actions),
        )
        self.actor_terms = [
            (index, width, getattr(policy_cfg, name))
            for index, (name, width) in enumerate(terms)
            if getattr(policy_cfg, name, None) is not None
        ]
        self.enable_corruption = policy_cfg.enable_corruption
        for _, _, term in self.actor_terms:
            if term.clip is not None or term.scale is not None:
                raise ValueError("MotionTrackingTask requires unscaled, unclipped policy observation terms.")
        collisions = [cfg for cfg in self.rewards.values() if cfg.func.__name__ == "self_collision_cost"]
        self.num_labels = (
            owner.scene.sensors[collisions[0].params["sensor_cfg"].name].cfg.history_length if collisions else 0
        )

    def world_model_loss(self, prediction: State, target: State, scale: torch.Tensor) -> torch.Tensor:
        """Fit integrated root pose and joint state with fixed physical error scales."""
        orientation_error = (matrix_from_quat(prediction[1]) - matrix_from_quat(target[1])).flatten(-2)
        errors = (
            (prediction[0] - target[0]) / 0.01,
            orientation_error * (0.5**0.5) / 0.02,
            (prediction[4] - target[4]) / 0.1,
            (prediction[5] - target[5]) / 3.14,
        )
        return torch.cat(errors, -1).square().mean(-1)

    def _bodies(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward kinematics of a state, reused while the same state is passed to several heads."""
        if self._cached_state is not state:
            self._cached_state, self._cached_bodies = state, self.kinematics.forward(state)
        return self._cached_bodies

    def _frame(self, goal: torch.Tensor) -> torch.Tensor:
        return goal[..., 0].long()

    def _anchor_orientation(self, state: State, frame: torch.Tensor) -> torch.Tensor:
        """Reference anchor orientation in the robot anchor frame, as the first two rotation-matrix columns."""
        _, quat, _, _ = self._bodies(state)
        relative = quat_mul(quat_conjugate(quat[..., self.robot_anchor, :]), self.ref_body_quat[frame, self.ref_anchor])
        return matrix_from_quat(relative)[..., :2].flatten(-2)

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> TensorDict:
        """Return the default policy's terms other than the reference joint command."""
        frame = self._frame(goal)
        return self.policy_observations(
            state, self._joint_command(goal), self.ref_body_quat[frame, self.ref_anchor], prev_actions[..., -1, :]
        )

    def policy_observations(
        self, state: State, joint_command: torch.Tensor, anchor_quat: torch.Tensor, previous_action: torch.Tensor
    ) -> TensorDict:
        """Build policy inputs from physical state and explicit reference channels, also used by deployment."""
        _, quat, _, _ = self._bodies(state)
        relative = quat_mul(quat_conjugate(quat[..., self.robot_anchor, :]), anchor_quat)
        parts = (
            matrix_from_quat(relative)[..., :2].flatten(-2),
            quat_apply_inverse(state[1], state[3]),
            state[4] - self.default_pos,
            state[5] - self.default_vel,
            previous_action,
        )
        frame = torch.cat(tuple(parts[index] for index, _, _ in self.actor_terms), dim=-1)
        return TensorDict({"policy": frame, "actor_goal": joint_command}, batch_size=frame.shape[:-1])

    def _joint_command(self, goal: torch.Tensor) -> torch.Tensor:
        """Return the reference joint positions and velocities."""
        frame = self._frame(goal)
        return torch.cat((self.ref_joint_pos[frame], self.ref_joint_vel[frame]), dim=-1)

    def reference_action(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Return the reference joint positions of the current frame as joint-position actions."""
        target = self.ref_joint_pos[self._frame(goal)][..., self.action_joints]
        return (target - self.action_offset) / self.action_scale

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Match the default policy's independent uniform observation corruption."""
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
        """Condition dynamics on log-scaled body masses [kg] and recent actions."""
        return torch.cat((torch.log1p(context), prev_actions.flatten(-2)), dim=-1)

    def critic_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Give the critic the physical state, the reference command and anchor, and the dynamics context."""
        frame = self._frame(goal)
        anchor = torch.cat(
            (
                self.ref_body_pos[frame, self.ref_anchor] - state[0],
                self._anchor_orientation(state, frame),
                (frame.float() / self.last_frame)[..., None],
            ),
            dim=-1,
        )
        return torch.cat(
            (
                self.representation.features(state),
                self._joint_command(goal),
                anchor,
                self.dynamics_context(state, context, prev_actions),
            ),
            dim=-1,
        )

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step to the next reference frame, holding the last one."""
        end = self.clip_ends[torch.searchsorted(self.clip_ends, self._frame(goal).contiguous(), right=True)]
        return torch.minimum(goal + 1, (end - 1)[..., None]), context

    def _tracking(self, state: State, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        """Robot and reference quantities compared by the tracking rewards and terminations."""
        pos, quat, lin, ang = self._bodies(state)
        robot_anchor_pos, robot_anchor_quat = pos[..., self.robot_anchor, :], quat[..., self.robot_anchor, :]
        ref_pos, ref_quat = self.ref_body_pos[frame], self.ref_body_quat[frame]
        ref_anchor_pos, ref_anchor_quat = ref_pos[..., self.ref_anchor, :], ref_quat[..., self.ref_anchor, :]
        yaw = yaw_quat(quat_mul(robot_anchor_quat, quat_conjugate(ref_anchor_quat)))[..., None, :]
        aligned_anchor = torch.cat((robot_anchor_pos[..., :2], ref_anchor_pos[..., 2:]), dim=-1)
        return {
            "anchor_pos": robot_anchor_pos,
            "anchor_quat": robot_anchor_quat,
            "ref_anchor_pos": ref_anchor_pos,
            "ref_anchor_quat": ref_anchor_quat,
            "body_pos": pos[..., self.tracked, :],
            "body_quat": quat[..., self.tracked, :],
            "body_lin": lin[..., self.tracked, :],
            "body_ang": ang[..., self.tracked, :],
            "ref_body_pos": aligned_anchor[..., None, :] + _apply(yaw, ref_pos - ref_anchor_pos[..., None, :]),
            "ref_body_quat": _mul(yaw, ref_quat),
            "ref_body_lin": self.ref_body_lin[frame],
            "ref_body_ang": self.ref_body_ang[frame],
        }

    def _termination_margins(self, next_state: State, goal: torch.Tensor) -> list[torch.Tensor]:
        """Normalized distances past each termination threshold; positive values terminate."""
        values = self._tracking(next_state, self._frame(goal))
        margins = []
        for name, cfg in self.terminations.items():
            kind, params = cfg.func.__name__, cfg.params
            if kind == "bad_anchor_pos_z_only":
                error = (values["ref_anchor_pos"][..., 2] - values["anchor_pos"][..., 2]).abs()
                margins.append((error - params["threshold"]) / 0.02)
            elif kind == "bad_anchor_ori":
                gravity = self.gravity.expand_as(values["anchor_pos"])
                error = (
                    quat_apply_inverse(values["ref_anchor_quat"], gravity)[..., 2]
                    - quat_apply_inverse(values["anchor_quat"], gravity)[..., 2]
                ).abs()
                margins.append((error - params["threshold"]) / 0.04)
            else:
                ids = self.termination_bodies[name]
                error = (values["ref_body_pos"][..., ids, 2] - values["body_pos"][..., ids, 2]).abs()
                margins.append(((error - params["threshold"]) / 0.02).amax(-1))
        return margins

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return one minus the product of sigmoid survival probabilities at the termination thresholds."""
        survival = torch.ones_like(next_state[0][..., 0])
        for margin in self._termination_margins(next_state, goal):
            survival = survival * (1 - torch.sigmoid(margin))
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

        The successor is compared with the pre-transition reference frame, as in the environment step.
        """
        values = self._tracking(next_state, self._frame(goal))
        q = next_state[4]
        results = {}
        for name, cfg in self.rewards.items():
            kind, params = cfg.func.__name__, cfg.params
            value = torch.zeros_like(q[..., 0])
            if cfg.weight == 0:
                results[name] = value
                continue
            if kind == "motion_global_anchor_position_error_exp":
                error = (values["ref_anchor_pos"] - values["anchor_pos"]).square().sum(-1)
            elif kind == "motion_global_anchor_orientation_error_exp":
                error = _angle_squared(values["ref_anchor_quat"], values["anchor_quat"])
            elif kind == "motion_relative_body_position_error_exp":
                error = (values["ref_body_pos"] - values["body_pos"]).square().sum(-1).mean(-1)
            elif kind == "motion_relative_body_orientation_error_exp":
                error = _angle_squared(values["ref_body_quat"], values["body_quat"]).mean(-1)
            elif kind == "motion_global_body_linear_velocity_error_exp":
                error = (values["ref_body_lin"] - values["body_lin"]).square().sum(-1).mean(-1)
            elif kind == "motion_global_body_angular_velocity_error_exp":
                error = (values["ref_body_ang"] - values["body_ang"]).square().sum(-1).mean(-1)
            if kind.startswith("motion_"):
                value = torch.exp(-error / params["std"] ** 2)
            elif kind == "action_rate_l2":
                value = (actions - prev_actions[..., -1, :]).square().sum(-1)
            elif kind == "joint_pos_limits":
                ids = self.limit_joints[name]
                value = ((self.limits[..., 0] - q).relu() + (q - self.limits[..., 1]).relu())[..., ids].sum(-1)
            elif kind == "self_collision_cost":
                if labels is None:
                    raise ValueError("The self-collision penalty needs predicted or observed labels.")
                value = labels.sum(-1)
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
        """Sum the motion tracking MDP's weighted reward terms."""
        return torch.stack(
            tuple(self.reward_terms(state, actions, next_state, goal, prev_actions, labels).values()), -1
        ).sum(-1)
