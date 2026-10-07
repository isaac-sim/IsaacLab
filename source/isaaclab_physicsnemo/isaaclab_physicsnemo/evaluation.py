# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate learned dynamics against recorded data and against teacher rollouts."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from physicsnemo_robotics.dynamics import DynamicsArtifact, DynamicsFrame, FeatureEncoder, StateCodec

from .recording import CONTROL_KEY

if TYPE_CHECKING:
    from physicsnemo_robotics.data import TrajectoryReader

    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


@dataclass(frozen=True)
class HorizonError:
    """Prefix-averaged rollout error of a dynamics artifact for one horizon."""

    horizon: int
    """Number of autoregressive predictions averaged into the error."""

    nrmse: float
    """Normalized root-mean-square error, averaged over state channels."""

    trajectories: int
    """Number of evaluated trajectories long enough for the horizon."""


@dataclass(frozen=True)
class JointRollout:
    """Generalized Newton joint state of every environment over one rollout."""

    joint_q: torch.Tensor
    """Joint positions [m or rad, depending on joint type], shape [steps + 1, num_envs, dof_q]."""

    joint_qd: torch.Tensor
    """Joint velocities [m/s or rad/s, depending on joint type], shape [steps + 1, num_envs, dof_qd]."""

    active: torch.Tensor
    """Whether the environment has not been reset since the rollout started, shape [steps + 1, num_envs]."""

    revolute_q: torch.Tensor
    """Whether each joint-position coordinate is a revolute angle, shape [dof_q]."""

    quaternion_q_starts: tuple[int, ...]
    """First joint-position coordinate of each ``(x, y, z, w)`` orientation quaternion."""


@dataclass(frozen=True)
class RolloutError:
    """Open-loop error of a learned rollout against its teacher rollout.

    Revolute angles are compared modulo a full turn and orientation quaternions up to their sign, so
    that equivalent configurations in different coordinates do not count as errors.
    """

    joint_q_rmse: torch.Tensor
    """Joint-position root-mean-square error per step over active environments, shape [steps + 1]."""

    joint_qd_rmse: torch.Tensor
    """Joint-velocity root-mean-square error per step over active environments, shape [steps + 1]."""

    active_envs: torch.Tensor
    """Number of environments active in both rollouts per step, shape [steps + 1]."""


def evaluate_rollout_nrmse(
    artifact: DynamicsArtifact,
    reader: TrajectoryReader,
    codec: StateCodec,
    features: FeatureEncoder,
    *,
    evaluation_indices: Sequence[int],
    reference_indices: Sequence[int],
    horizons: Sequence[int] = (1, 15, 100, 200),
    max_trajectories: int = 2048,
    device: str | torch.device = "cuda:0",
) -> tuple[HorizonError, ...]:
    """Measure recursive-rollout accuracy with the prefix-averaged NRMSE of the FlashNeRD evaluation.

    Each evaluated trajectory starts from its first ``history`` recorded states and is rolled out
    autoregressively with its recorded joint efforts, so predictions become inputs and errors compound.
    The error of a predicted state is its codec transition from the recorded state, which wraps
    continuous angles and expresses orientation errors as rotation vectors. Each channel is normalized
    by the standard deviation of the reference states in the same coordinates. ``E_H`` is the mean over
    channels of the root-mean-square normalized error over the first ``H`` predictions of every
    trajectory with at least ``history + H`` states.

    Args:
        artifact: Artifact to evaluate.
        reader: Recorded trajectories with ``state`` and :data:`~isaaclab_physicsnemo.CONTROL_KEY` fields.
        codec: State codec of the artifact, rebuilt against the recording scene.
        features: Feature encoder of the artifact, which reads the recorded joint efforts.
        evaluation_indices: Reader indices of the held-out trajectories to roll out.
        reference_indices: Reader indices of the trajectories that define the normalization, typically
            the training trajectories.
        horizons: Numbers of predictions to average over.
        max_trajectories: Maximum number of trajectories evaluated per horizon.
        device: Inference device.

    Returns:
        One error per horizon that at least one trajectory is long enough for.
    """
    operator = artifact.as_operator(codec=codec, features=features, device=device)
    history = operator.history

    def read(index: int, *, controls: bool = True) -> dict[str, torch.Tensor]:
        """Read the state timeline and, optionally, the one-shorter transition timeline of a trajectory."""
        length = reader.trajectory_length(index)
        fields = reader.read_slice(index, 0, length, ("state",))
        if controls:
            fields |= reader.read_slice(index, 0, length - 1, (CONTROL_KEY,))
        return fields

    trajectories = [read(index) for index in evaluation_indices]
    reference = torch.cat([read(index, controls=False)["state"] for index in reference_indices]).to(device)
    anchor = reference[:1].expand_as(reference)
    spread = codec.state_to_delta(anchor, reference).std(dim=0)
    spread = torch.where(spread > 1.0e-8, spread, torch.ones_like(spread))

    errors = []
    for horizon in horizons:
        selected = [data for data in trajectories if data["state"].shape[0] >= history + horizon][:max_trajectories]
        if not selected:
            continue
        states = torch.stack([data["state"][: history + horizon] for data in selected]).to(device)
        controls = torch.stack([data[CONTROL_KEY][: history + horizon - 1] for data in selected]).to(device)
        controls = controls.flatten(start_dim=2)
        rollout = operator.rollout(
            initial_history=states[:, :history],
            steps=horizon,
            feature_history=controls[:, :history],
            frame_factory=lambda current, index: DynamicsFrame(current, features=controls[:, history + index]),
        )
        recorded = states[:, history:].flatten(end_dim=1)
        predicted = rollout.states[:, 1:].to(device).flatten(end_dim=1)
        normalized = codec.state_to_delta(recorded, predicted) / spread
        nrmse = normalized.square().mean(dim=0).sqrt().mean().item()
        errors.append(HorizonError(horizon=horizon, nrmse=nrmse, trajectories=len(selected)))
    return tuple(errors)


def record_joint_rollout(
    env: DirectRLEnv | ManagerBasedRLEnv,
    actions: Callable[[torch.Tensor | None], torch.Tensor],
    steps: int,
) -> JointRollout:
    """Step an environment and record the Newton joint state after each step.

    Run the teacher and the learned environment from the same seed with identically seeded action
    sources to obtain comparable rollouts. An environment is marked inactive from the step at which
    it is first reset, because its state no longer continues the compared trajectory.

    Args:
        env: Environment after its first reset.
        actions: Action source called with the previous step's dones, such as
            :class:`~isaaclab_physicsnemo.SmoothRandomActions`.
        steps: Number of environment steps.

    Returns:
        The recorded rollout.
    """
    manager = env.sim.physics_manager
    worlds = env.num_envs
    revolute_q, quaternion_q_starts = _joint_q_layout(manager.get_model())

    def joint_state() -> tuple[torch.Tensor, torch.Tensor]:
        state = manager.get_state_0()
        return (
            wp.to_torch(state.joint_q).reshape(worlds, -1).clone(),
            wp.to_torch(state.joint_qd).reshape(worlds, -1).clone(),
        )

    joint_q, joint_qd = joint_state()
    positions, velocities = [joint_q], [joint_qd]
    active = [torch.ones(worlds, dtype=torch.bool, device=joint_q.device)]
    dones = None
    with torch.inference_mode():
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(actions(dones))
            dones = terminated | truncated
            joint_q, joint_qd = joint_state()
            positions.append(joint_q)
            velocities.append(joint_qd)
            active.append(active[-1] & ~dones)
    return JointRollout(
        torch.stack(positions),
        torch.stack(velocities),
        torch.stack(active),
        revolute_q.to(joint_q.device),
        quaternion_q_starts,
    )


def compare_rollouts(teacher: JointRollout, learned: JointRollout) -> RolloutError:
    """Compute the per-step open-loop error of a learned rollout over environments active in both.

    Args:
        teacher: Rollout of the teacher solver.
        learned: Rollout of the learned solver with the same seed and actions.

    Returns:
        Per-step errors. Steps without active environments report ``nan``.

    Raises:
        ValueError: If the rollouts have different shapes.
    """
    if teacher.joint_q.shape != learned.joint_q.shape or teacher.joint_qd.shape != learned.joint_qd.shape:
        raise ValueError("Rollouts must have the same number of steps, environments, and joint coordinates.")
    active = teacher.active & learned.active
    count = active.sum(dim=-1)

    position_error = learned.joint_q - teacher.joint_q
    position_error = torch.where(
        teacher.revolute_q, torch.remainder(position_error + math.pi, 2.0 * math.pi) - math.pi, position_error
    )
    for start in teacher.quaternion_q_starts:
        same_sign = position_error[..., start : start + 4]
        opposite_sign = learned.joint_q[..., start : start + 4] + teacher.joint_q[..., start : start + 4]
        use_opposite = opposite_sign.norm(dim=-1, keepdim=True) < same_sign.norm(dim=-1, keepdim=True)
        position_error[..., start : start + 4] = torch.where(use_opposite, opposite_sign, same_sign)

    def rmse(error: torch.Tensor) -> torch.Tensor:
        squared = error.square().mean(dim=-1)
        return (torch.where(active, squared, 0.0).sum(dim=-1) / count).sqrt()

    return RolloutError(rmse(position_error), rmse(learned.joint_qd - teacher.joint_qd), count)


def _joint_q_layout(model: newton.Model) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return the revolute-angle mask and quaternion starts of one world's joint-position coordinates."""
    joints_per_world = model.joint_count // model.world_count
    joint_types = model.joint_type.numpy()[:joints_per_world]
    q_starts = model.joint_q_start.numpy()[: joints_per_world + 1]
    dof_q = int(q_starts[joints_per_world])
    revolute_q = torch.zeros(dof_q, dtype=torch.bool)
    quaternion_q_starts = []
    for joint_type, q_start in zip(joint_types, q_starts[:-1]):
        if joint_type == newton.JointType.REVOLUTE:
            revolute_q[q_start] = True
        elif joint_type == newton.JointType.FREE:
            quaternion_q_starts.append(int(q_start) + 3)
        elif joint_type == newton.JointType.BALL:
            quaternion_q_starts.append(int(q_start))
    return revolute_q, tuple(quaternion_q_starts)
