# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 the SuperMimic authors.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Task contracts and generalized/maximal coordinate representations for learned dynamics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab_rl.model_based.kinematics import (
    matrix_from_quat,
    matrix_to_quat,
    quat_apply,
    quat_apply_inverse,
    quat_integrate,
    quat_inv,
    quat_mul,
    quat_to_rot6d,
    rodrigues_apply,
    rodrigues_precompute,
    rot6d_to_matrix,
    small_matmul,
    subtract_frame_transforms,
)

from .kinematics import AnalyticIK, ForwardKinematics


def infer_num_joints_from_state_dim(state_dim: int) -> int:
    """Infers number of joints from full-state dimension.

    Full-state layout is:
    [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
    """
    if state_dim < 13 or (state_dim - 13) % 2 != 0:
        raise ValueError(f"Invalid full-state dim {state_dim}: expected 13 + 2 * num_joints")
    return (state_dim - 13) // 2


def rep_state_dim_from_num_joints(num_joints: int) -> int:
    return 10 + 2 * num_joints


def full_state_offsets(num_joints: int) -> tuple[int, int, int, int]:
    """Returns start offsets for [qj, root_lin_vel, root_ang_vel, vj] in full state."""
    idx_qj = 7
    idx_lin = idx_qj + num_joints
    idx_ang = idx_lin + 3
    idx_vj = idx_ang + 3
    return idx_qj, idx_lin, idx_ang, idx_vj


def rep_state_offsets(num_joints: int) -> tuple[int, int]:
    """Returns start offsets for [qj, vj] in rep state."""
    idx_qj = 10
    idx_vj = idx_qj + num_joints
    return idx_qj, idx_vj


def decompose_state(
    full_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes full state into canonical components."""
    num_joints = infer_num_joints_from_state_dim(full_state.shape[-1])

    idx_qj, idx_lin, idx_ang, idx_vj = full_state_offsets(num_joints)

    root_pos = full_state[..., 0:3]
    root_quat = full_state[..., 3:7]
    qj = full_state[..., idx_qj:idx_lin]
    root_lin_vel = full_state[..., idx_lin:idx_ang]
    root_ang_vel = full_state[..., idx_ang:idx_vj]
    vj = full_state[..., idx_vj : idx_vj + num_joints]
    return root_pos, root_quat, qj, root_lin_vel, root_ang_vel, vj


def decompose_rep_state(
    rep_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes gencoord rep state into canonical components."""
    # rep state layout: [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    if rep_state.shape[-1] < 10 or (rep_state.shape[-1] - 10) % 2 != 0:
        raise ValueError(f"Invalid rep-state dim {rep_state.shape[-1]}: expected 10 + 2 * num_joints")
    num_joints = (rep_state.shape[-1] - 10) // 2

    idx_qj, idx_vj = rep_state_offsets(num_joints)

    z = rep_state[..., 0:1]
    g_b = rep_state[..., 1:4]
    v_b = rep_state[..., 4:7]
    w_b = rep_state[..., 7:10]
    qj = rep_state[..., idx_qj:idx_vj]
    vj = rep_state[..., idx_vj : idx_vj + num_joints]
    return z, g_b, v_b, w_b, qj, vj


def compute_gencoord_state(full_state: torch.Tensor) -> torch.Tensor:
    """Converts raw full state to gencoord representation.

    Uses root body frame for velocities (no heading alignment).
    This avoids all singularities since we're just rotating to body frame.
    """
    (
        root_pos,
        root_quat,
        qj,
        root_lin_vel,
        root_ang_vel,
        vj,
    ) = decompose_state(full_state)

    z = root_pos[..., 2:3]
    world_down = torch.zeros_like(root_lin_vel)
    world_down[..., 2] = -1.0
    g_b = quat_apply_inverse(root_quat, world_down)
    v_b = quat_apply_inverse(root_quat, root_lin_vel)
    w_b = quat_apply_inverse(root_quat, root_ang_vel)

    return torch.cat([z, g_b, v_b, w_b, qj, vj], dim=-1)


def maxcoord_full_state_dim(num_bodies: int) -> int:
    return num_bodies * 13


def maxcoord_rep_state_dim(num_bodies: int) -> int:
    return num_bodies * 15 + 4


def maxcoord_prediction_dim(num_bodies: int) -> int:
    return num_bodies * 6


def _reshape_bodies(flat: torch.Tensor, num_bodies: int, feat: int) -> torch.Tensor:
    """(..., B*feat) -> (..., B, feat)"""
    return flat.unflatten(-1, (num_bodies, feat))


def decompose_maxcoord_state(
    full_state: torch.Tensor, num_bodies: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose flat maxcoord full state into per-body tensors.

    Returns:
      body_pos: (..., B, 3)
      body_quat: (..., B, 4)
      body_lin_vel: (..., B, 3)
      body_ang_vel: (..., B, 3)
    """
    B = num_bodies
    body_pos = _reshape_bodies(full_state[..., : B * 3], B, 3)
    body_quat = _reshape_bodies(full_state[..., B * 3 : B * 7], B, 4)
    body_lin_vel = _reshape_bodies(full_state[..., B * 7 : B * 10], B, 3)
    body_ang_vel = _reshape_bodies(full_state[..., B * 10 : B * 13], B, 3)
    return body_pos, body_quat, body_lin_vel, body_ang_vel


def build_maxcoord_full_state(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_lin_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
) -> torch.Tensor:
    """Build flat maxcoord full state from per-body tensors.

    Args:
      body_pos: (..., B, 3) body positions
      body_quat: (..., B, 4) body quaternions
      body_lin_vel: (..., B, 3) body linear velocities
      body_ang_vel: (..., B, 3) body angular velocities

    Returns:
      (..., B*13) flat maxcoord full state
    """
    return torch.cat(
        [
            body_pos.flatten(-2, -1),
            body_quat.flatten(-2, -1),
            body_lin_vel.flatten(-2, -1),
            body_ang_vel.flatten(-2, -1),
        ],
        dim=-1,
    )


def compute_maxcoord_rep_state(full_state: torch.Tensor, num_bodies: int) -> torch.Tensor:
    """Convert maxcoord full state to root-relative rep state.

    Layout:
      [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
       root_height(1), up_dir(3)]

    Args:
      full_state: (..., B*13) flat maxcoord state
      num_bodies: B

    Returns:
      (..., B*15 + 4) rep state
    """
    body_pos, body_quat, body_lin_vel, body_ang_vel = decompose_maxcoord_state(full_state, num_bodies)

    # Root frame
    root_pos = body_pos[..., 0:1, :]  # (..., 1, 3)
    root_quat = body_quat[..., 0:1, :]  # (..., 1, 4)
    root_quat_inv = quat_inv(root_quat.reshape(-1, 4)).reshape(root_quat.shape)

    # Local positions: relative to root, rotated into root frame
    rel_pos_w = body_pos - root_pos  # (..., B, 3)
    local_pos = quat_apply_inverse(
        root_quat.expand_as(body_quat),
        rel_pos_w,
    )  # (..., B, 3)

    # Local rotations as 6D representation (first two columns of rotation matrix)
    rel_quat = quat_mul(
        root_quat_inv.expand_as(body_quat),
        body_quat,
    )  # (..., B, 4)
    local_rot_6d = quat_to_rot6d(rel_quat)  # (..., B, 6)

    # Local velocities: rotated into root frame
    local_vel = quat_apply_inverse(
        root_quat.expand_as(body_quat),
        body_lin_vel,
    )  # (..., B, 3)
    local_avel = quat_apply_inverse(
        root_quat.expand_as(body_quat),
        body_ang_vel,
    )  # (..., B, 3)

    # Root height (Z coordinate in world frame)
    root_height = body_pos[..., 0, 2:3]  # (..., 1)

    # Up direction in root frame
    world_down = torch.zeros_like(root_pos[..., 0, :])  # (..., 3)
    world_down[..., 2] = -1.0
    up_dir = quat_apply_inverse(root_quat[..., 0, :], world_down)  # (..., 3)

    # Flatten per-body features and concatenate
    return torch.cat(
        [
            local_pos.flatten(-2, -1),  # B*3
            local_rot_6d.flatten(-2, -1),  # B*6
            local_vel.flatten(-2, -1),  # B*3
            local_avel.flatten(-2, -1),  # B*3
            root_height,  # 1
            up_dir,  # 3
        ],
        dim=-1,
    )


def decompose_maxcoord_rep_state(
    rep_state: torch.Tensor, num_bodies: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Decompose flat maxcoord rep state into per-component tensors.

    Returns:
      local_pos: (..., B, 3)
      local_rot_6d: (..., B, 6)
      local_vel: (..., B, 3)
      local_avel: (..., B, 3)
      root_height: (..., 1)
      up_dir: (..., 3)
    """
    B = num_bodies
    off = 0
    local_pos = _reshape_bodies(rep_state[..., off : off + B * 3], B, 3)
    off += B * 3
    local_rot_6d = _reshape_bodies(rep_state[..., off : off + B * 6], B, 6)
    off += B * 6
    local_vel = _reshape_bodies(rep_state[..., off : off + B * 3], B, 3)
    off += B * 3
    local_avel = _reshape_bodies(rep_state[..., off : off + B * 3], B, 3)
    off += B * 3
    root_height = rep_state[..., off : off + 1]
    off += 1
    up_dir = rep_state[..., off : off + 3]
    return local_pos, local_rot_6d, local_vel, local_avel, root_height, up_dir


def world_frame_bodies(
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    num_bodies: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover world-frame body kinematics from (aux_state, rep_state).

    Args:
      aux_state: (..., 7) [root_pos(3), root_quat(4)]
      rep_state: (..., B*15+4) rep state
      num_bodies: B

    Returns:
      body_pos: (..., B, 3)
      body_rot_mat: (..., B, 3, 3) world-frame rotation matrices
      body_lin_vel: (..., B, 3)
      body_ang_vel: (..., B, 3)
    """
    local_pos, local_rot_6d, local_vel, local_avel, _, _ = decompose_maxcoord_rep_state(rep_state, num_bodies)
    root_pos = aux_state[..., :3]
    root_mat = matrix_from_quat(aux_state[..., 3:7]).unsqueeze(-3)

    def rotate(vec: torch.Tensor) -> torch.Tensor:
        return small_matmul(root_mat, vec.unsqueeze(-1))[..., 0]

    body_pos = root_pos.unsqueeze(-2) + rotate(local_pos)
    body_rot_mat = small_matmul(root_mat, rot6d_to_matrix(local_rot_6d))
    body_lin_vel = rotate(local_vel)
    body_ang_vel = rotate(local_avel)
    return body_pos, body_rot_mat, body_lin_vel, body_ang_vel


def reconstruct_full_from_aux_and_rep(
    aux_state: torch.Tensor,
    rep_state: torch.Tensor,
    num_bodies: int,
) -> torch.Tensor:
    """Reconstruct maxcoord full state from aux_state + rep_state."""
    body_pos, body_rot_mat, body_lin_vel, body_ang_vel = world_frame_bodies(aux_state, rep_state, num_bodies)
    body_quat = matrix_to_quat(body_rot_mat)
    return build_maxcoord_full_state(body_pos, body_quat, body_lin_vel, body_ang_vel)


class MotionReference(Protocol):
    """Reference channels consumed by state integration and goal encoding."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    motion_starts: torch.Tensor
    motion_ends: torch.Tensor


class SimulationContext(Protocol):
    """Normalization layout shared by physical and imagined observations."""

    def find(self, name: str) -> slice | None: ...

    def denormalize(self, values: torch.Tensor, columns: slice) -> torch.Tensor: ...


ExtractFromMotion = Callable[[MotionReference, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TaskSpec:
    """Contract for one task slot of the multi-encoder actor."""

    name: str
    goal_dim: int
    extract_training_goal: ExtractFromMotion


def build_actor_obs_td(
    goals_by_name: dict[str, torch.Tensor],
    history: torch.Tensor,
    critic: torch.Tensor | None = None,
) -> TensorDict:
    """``history`` (B, H, obs_dim) (+ optional ``critic``) plus one ``goal_<name>``
    key per task. The newest frame ``history[:, -1]`` is the current obs."""
    obs: dict[str, torch.Tensor] = {"history": history}
    if critic is not None:
        obs["critic"] = critic
    obs.update({f"goal_{name}": goal for name, goal in goals_by_name.items()})
    return TensorDict(obs, batch_size=[history.shape[0]], device=history.device)


def create_representation(
    name: str,
    full_state_dim: int,
    num_actions: int,
    dt: float,
    decimation: int,
    default_joint_pos: torch.Tensor,
    actor_obs_mode: str,
) -> StateRepresentation:
    """Create a state representation wired up for training and reward compute."""
    if name == "gencoord":
        num_joints = infer_num_joints_from_state_dim(full_state_dim)
        rep: StateRepresentation = GenCoordRepresentation(num_joints, default_joint_pos)
    elif name == "maxcoord":
        if full_state_dim % 13 != 0:
            raise ValueError(f"Maxcoord full_state_dim ({full_state_dim}) must be divisible by 13")
        rep = MaxCoordRepresentation(full_state_dim // 13, default_joint_pos)
    else:
        raise ValueError(f"Unknown representation: {name}. Must be 'gencoord' or 'maxcoord'.")
    rep.actor_obs_mode = actor_obs_mode
    if actor_obs_mode == "proprio":
        rep.register_buffer(
            "_proprio_noise_bounds",
            rep.proprio_noise_bounds(num_actions, default_joint_pos.device),
            persistent=False,
        )
    rep.dt = dt
    # Semi-implicit composition of `decimation` substeps under constant accel.
    rep.substep_coeff = (decimation + 1) / (2 * decimation)
    return rep


# ---------------------------------------------------------------------------
# Goal layout constants
# ---------------------------------------------------------------------------
# Reference-motion frames exposed in the goal window: a strided future window of
# NUM_GOAL_FRAMES frames spaced GOAL_FRAME_STRIDE motion steps apart. Motion runs
# at the control rate (one frame per env step, ~50 Hz / 0.02 s), so stride 5
# ≈ 0.1 s spacing and a ~0.9 s lookahead with 10 frames.
NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5

# wm_goal  = [content, rel_pos_b(3*N), rel_ori_b(6*N)]
# actor_goal = [content, rel_ori_b(6*N)]
# content = NUM_GOAL_FRAMES joint targets: per frame [joint_pos, joint_vel].
GOAL_REL_POS_DIM = 3 * NUM_GOAL_FRAMES  # rel_pos_b: in wm_goal only
GOAL_REL_ORI_DIM = 6 * NUM_GOAL_FRAMES  # rel_ori_b: in both wm_goal and actor_goal


def actor_goal_dim(num_goals: int) -> int:
    """Number of goal dimensions seen by the actor (wm_goal minus rel_pos_b)."""
    return num_goals - GOAL_REL_POS_DIM


PROPRIO_NOISE_HALF_WIDTHS: dict[str, float] = {
    "projected_gravity": 0.05,
    "base_ang_vel": 0.2,
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "actions": 0.0,
}


def term_major_to_temporal(flat: torch.Tensor, term_widths: list[int], history_length: int) -> torch.Tensor:
    """Convert term-major flat obs to a ``(B, H, sum(w_i))`` frame-major window.

    Input layout ``[term0_t0..tH-1, term1_t0..tH-1, ...]`` is what the deployed
    actor obs group emits; the decoder consumes frame-major. Applied at the ONNX
    export boundary so deploy and training see the same window.
    """
    parts = []
    off = 0
    for w in term_widths:
        chunk = flat[..., off : off + history_length * w]
        parts.append(chunk.unflatten(-1, (history_length, w)))
        off += history_length * w
    return torch.cat(parts, dim=-1)


def extract_actor_goal(wm_goal: torch.Tensor) -> torch.Tensor:
    """Extract actor goal from full WM goal by removing rel_pos_b dims."""
    tail = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
    return torch.cat([wm_goal[..., :-tail], wm_goal[..., -GOAL_REL_ORI_DIM:]], dim=-1)


def _root_loss_blocks(aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """World root position (3) and rotation (9) as squared-error channels.

    The rotation is scaled by sqrt(1/2) so a plain squared difference is
    0.5*||dR||_F^2 = 2(1 - cos theta) = theta^2 + O(theta^4): a nominal in radians.
    """
    rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    return {"root_pos": aux_state[..., :3], "root_ori": rot * math.sqrt(0.5)}


def _assemble_goal(
    content: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``(wm_goal, actor_goal)``; only wm_goal carries the rel_pos dims."""
    rel_pos, rel_ori = assemble_anchor_window(root_pos, root_quat, target_pos, target_quat)
    wm_goal = torch.cat([content, rel_pos, rel_ori], dim=-1)
    actor_goal = torch.cat([content, rel_ori], dim=-1)
    return wm_goal, actor_goal


def goal_window_indices(motion: MotionReference, motion_times: torch.Tensor) -> torch.Tensor:
    """Clamped windowed frame indices ``[t, t+stride, ..., t+(N-1)*stride]``, ``(E, N)``.

    Future offsets are clamped to the last frame of each env's current clip so the
    window never bleeds into the next concatenated clip — matching how the env
    advances ``motion_times``.
    """
    clip = torch.searchsorted(motion.motion_ends, motion_times, right=True)
    # Past-the-end guard: searchsorted returns numel() at the final frame.
    clip = clip.clamp(max=motion.motion_ends.numel() - 1)
    last_valid = motion.motion_ends[clip] - 1
    offsets = torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
    return torch.minimum(motion_times[:, None] + offsets, last_valid[:, None])


def motion_goal_window(motion: MotionReference, motion_times: torch.Tensor) -> torch.Tensor:
    """Windowed joint targets as the goal content, ``(E, N * 2J)`` — per frame
    ``[joint_pos(J), joint_vel(J)]``.
    """
    idx = goal_window_indices(motion, motion_times)
    frames = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1)
    return frames.reshape(frames.shape[0], -1)


def motion_anchor_window(motion: MotionReference, motion_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Motion anchor (the root) world pose ``(pos (E, N, 3), quat (E, N, 4))`` at
    the same windowed indices as :func:`motion_goal_window`.
    """
    idx = goal_window_indices(motion, motion_times)
    return motion.root_pos_w[idx], motion.root_quat_w[idx]


def assemble_anchor_window(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame motion anchor expressed relative to the robot's current root.

    ``root_*``: ``(E, 3)`` / ``(E, 4)``; ``target_*``: ``(E, N, 3)`` / ``(E, N, 4)``.
    Returns flattened ``(rel_pos (E, N*3), rel_ori_rot6d (E, N*6))``. The world-frame
    origin offset cancels in the relative transform, so callers only need the root
    and target anchors expressed in a common frame.
    """
    num_envs, num_frames = target_pos.shape[:2]
    root_pos = root_pos[:, None, :].expand(num_envs, num_frames, 3).reshape(-1, 3)
    root_quat = root_quat[:, None, :].expand(num_envs, num_frames, 4).reshape(-1, 4)
    rel_pos, rel_ori_w = subtract_frame_transforms(
        root_pos, root_quat, target_pos.reshape(-1, 3), target_quat.reshape(-1, 4)
    )
    rel_pos = rel_pos.reshape(num_envs, -1)
    rel_ori = quat_to_rot6d(rel_ori_w).reshape(num_envs, -1)
    return rel_pos, rel_ori


_compiled_assemble_goal = torch.compile(_assemble_goal)


class StateRepresentation(nn.Module, ABC):
    """Abstract base class for state representations.

    Owns the state-semantics methods (``represent``, ``apply_prediction``,
    etc.) and the world-model reward.

    Subclasses must set these instance attributes in ``__init__``:
      full_state_dim, rep_state_dim, prediction_dim

    And extend this class attribute to cover every key ``loss_blocks`` returns:
      LOSS_WEIGHT_CONFIG: dict[str, str]
    """

    full_state_dim: int
    rep_state_dim: int
    prediction_dim: int
    num_joints: int
    actor_obs_mode: Literal["rep_state", "proprio"] = "rep_state"

    # Each channel group's nominal-error field on ``WorldModelLearnerCfg``. Every
    # representation scores the integrated root pose; subclasses add their own.
    LOSS_WEIGHT_CONFIG: dict[str, str] = {
        "root_pos": "nominal_error_root_pose_pos",
        "root_ori": "nominal_error_root_pose_ori",
    }

    substep_coeff: float = 1.0

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def represent(self, full_state: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract auxiliary state [root_pos(3), root_quat(4)] from full state."""
        ...

    @abstractmethod
    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        curr_rep_state: torch.Tensor,
        prediction: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply prediction to produce next aux_state and next rep_state.

        Returns:
          (next_aux_state, next_rep_state)
        """
        ...

    @abstractmethod
    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        """Reconstruct full state from aux_state + rep_state (on-demand, for viz)."""
        ...

    def bind(self, fk, soft_joint_pos_limits: torch.Tensor, sim_context: SimulationContext, reward_fn) -> None:
        """Resolve env-dependent state needed by the reward and the proprio sensor."""
        self._soft_joint_pos_limits = soft_joint_pos_limits
        self._reward_fn = reward_fn

    @abstractmethod
    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        """``[joint_pos, joint_vel]`` in ``qj`` column order."""
        ...

    @abstractmethod
    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize state-derived proprio from WM-predicted state."""
        ...

    def proprio_obs_dim(self, num_actions: int) -> int:
        """Dim of the deploy-shape sensor proprio."""
        return sum(self.proprio_term_widths(num_actions))

    def proprio_term_widths(self, num_actions: int) -> list[int]:
        """Widths of ``[projected_gravity, base_ang_vel, joint_pos, joint_vel, actions]``,
        the sensor proprio in the order ``_sensor_actor_group`` concatenates it."""
        return [3, 3, self.num_joints, self.num_joints, num_actions]

    def proprio_noise_bounds(self, num_actions: int, device: torch.device | str) -> torch.Tensor:
        """Per-dim uniform-noise half-widths matching the sensor proprio layout."""
        widths = self.proprio_term_widths(num_actions)
        halves = PROPRIO_NOISE_HALF_WIDTHS.values()
        return torch.cat([torch.full((w,), h, device=device) for w, h in zip(widths, halves, strict=True)])

    def actor_obs_dim(self, num_actions: int, num_goals: int, history_length: int) -> int:
        """Dim of the actor input — ``history_length`` sensor frames (proprio or
        rep_state) cat'd with the goal."""
        sensor_or_state = (
            self.rep_state_dim if self.actor_obs_mode == "rep_state" else self.proprio_obs_dim(num_actions)
        )
        return history_length * sensor_or_state + actor_goal_dim(num_goals)

    def add_proprio_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Add per-dim uniform proprio jitter — the single noise site shared by the
        synthesized sensor and the real-sim / imagination history rings."""
        return obs + ((torch.rand_like(obs) * 2 - 1) * self._proprio_noise_bounds).detach()

    def compute_actor_sensor(
        self,
        rep_state: torch.Tensor,
        sim_ctx: torch.Tensor,
        aux_state: torch.Tensor | None = None,
        joint_state: torch.Tensor | None = None,
        last_action: torch.Tensor | None = None,
        sim_actor_obs: torch.Tensor | None = None,
        noisy: bool = False,
    ) -> torch.Tensor:
        """Sensor-or-state half of the actor input (no goal).

        Returns ``rep_state`` in rep_state mode. In proprio mode, returns
        ``sim_actor_obs`` if provided, else
        synthesizes via :meth:`compute_wm_obs`; ``noisy=True`` adds per-dim
        uniform jitter scaled by the precomputed ``_proprio_noise_bounds``. ``aux_state``,
        ``joint_state`` and ``last_action`` are required only when synthesizing
        (proprio mode with no ``sim_actor_obs``).
        """
        if self.actor_obs_mode == "rep_state":
            return rep_state
        if sim_actor_obs is not None:
            obs = sim_actor_obs
        else:
            if aux_state is None or joint_state is None or last_action is None:
                raise ValueError(
                    "aux_state, joint_state and last_action are required to synthesize "
                    "proprio obs (no sim_actor_obs provided)"
                )
            obs = self.compute_wm_obs(rep_state, aux_state, joint_state, sim_ctx)
            obs = torch.cat([obs, last_action], dim=-1)
        if noisy:
            obs = self.add_proprio_noise(obs)
        return obs

    def build_actor_obs(self, sensor_history: torch.Tensor, actor_goal: torch.Tensor) -> torch.Tensor:
        """Flatten the ``(B, history_length, sensor_dim)`` window and cat the goal."""
        return torch.cat([sensor_history.flatten(start_dim=1), actor_goal], dim=-1)

    # ------------------------------------------------------------------
    # Default implementations (may be overridden)
    # ------------------------------------------------------------------

    def compute_wm_goal(
        self,
        motion: MotionReference,
        motion_times: torch.Tensor,
        aux_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(wm_goal, actor_goal)`` from the motion window at ``motion_times``.

        The single goal-build path shared by the WM env, the task goal extractor, and
        the viewer so every site produces an identical layout.
        """
        content = motion_goal_window(motion, motion_times)
        target_pos, target_quat = motion_anchor_window(motion, motion_times)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        return _compiled_assemble_goal(content, root_pos, root_quat, target_pos, target_quat)

    @abstractmethod
    def world_frame_bodies(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame ``(pos, rot_mat, lin_vel, ang_vel)`` of the tracked bodies."""
        ...

    def compute_wm_reward_from_reference(
        self,
        next_rep_state: torch.Tensor,
        next_aux_state: torch.Tensor,
        next_joint_state: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        target_body_pos_w: torch.Tensor,
        target_body_quat_w: torch.Tensor,
        target_body_lin_vel_w: torch.Tensor,
        target_body_ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        """Per-env scalar reward used for WM-env rollouts.

        ``action``/``last_action`` are raw policy outputs (pre-scale/offset), as the
        replay buffer stores them; :class:`isaaclab_rl.model_based.models.Dynamics` is the sole
        place the raw→proc transform is applied.
        """
        return self._reward_fn(
            next_aux_state,
            *self.world_frame_bodies(next_aux_state, next_rep_state),
            next_joint_state,
            action,
            last_action,
            target_body_pos_w,
            target_body_quat_w,
            target_body_lin_vel_w,
            target_body_ang_vel_w,
            self._soft_joint_pos_limits,
            self.dt,
        )

    @abstractmethod
    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """The channel groups the state loss scores, keyed as ``LOSS_WEIGHT_CONFIG``.

        Whatever the blocks leave out is excluded from the loss.
        """
        ...


class GenCoordRepresentation(StateRepresentation):
    """Generalized-coordinate (joint-space) representation.

    Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
    Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    Prediction: [Δv_b(3), Δw_b(3), Δvj(N)] — accel-level deltas in the current
    root frame; poses and derived channels are integrated analytically.
    """

    LOSS_WEIGHT_CONFIG = {
        **StateRepresentation.LOSS_WEIGHT_CONFIG,
        "joint_pos": "nominal_error_joint_pos",
        "joint_vel": "nominal_error_joint_vel",
    }

    def __init__(self, num_joints: int, default_joint_pos: torch.Tensor):
        super().__init__()
        self.num_joints = num_joints
        self.full_state_dim = 13 + 2 * num_joints
        self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
        self.prediction_dim = 6 + num_joints  # [Δv_b(3), Δw_b(3), Δvj(N)]
        self._default_joint_pos = default_joint_pos
        self.fk: ForwardKinematics  # set by bind
        self._encoder_bias: tuple[slice, SimulationContext] | None = None

    def bind(self, fk, soft_joint_pos_limits: torch.Tensor, sim_context: SimulationContext, reward_fn) -> None:
        super().bind(fk, soft_joint_pos_limits, sim_context, reward_fn)
        self.fk = fk
        columns = sim_context.find("encoder_bias")
        self._encoder_bias = None if columns is None else (columns, sim_context)

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        flat = full_state.reshape(-1, self.full_state_dim)
        rep = compute_gencoord_state(flat)
        return rep.reshape(*full_state.shape[:-1], -1)

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        return full_state[..., :7]

    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        curr_rep_state: torch.Tensor,
        prediction: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Semi-implicit joint-space integration of accel-level predictions.

        The network output is ``[Δv_b, Δw_b, Δvj]`` in the CURRENT root
        frame; ``qj`` integrates the joint velocities, the root pose
        integrates the twist, and ``z``/``g_b`` are derived from the
        integrated pose, so every prediction maps to a kinematically
        consistent state. Pose updates weight the delta by ``substep_coeff``
        to match how the sim composes its substeps; velocity states advance
        by the full delta.
        """
        N = self.num_joints
        c = self.substep_coeff
        _, _, v_b, w_b, qj, vj = decompose_rep_state(curr_rep_state)
        dv_b = prediction[..., 0:3]
        dw_b = prediction[..., 3:6]
        dvj = prediction[..., 6 : 6 + N]

        next_v_cur = v_b + dv_b  # next twist, expressed in the current root frame
        next_w_cur = w_b + dw_b
        next_vj = vj + dvj
        next_qj = qj + (vj + c * dvj) * dt

        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        v_global = quat_apply(root_quat, next_v_cur)
        w_global = quat_apply(root_quat, next_w_cur)
        v_pose = quat_apply(root_quat, v_b + c * dv_b)
        w_pose = quat_apply(root_quat, w_b + c * dw_b)
        next_root_pos = root_pos + v_pose * dt
        next_root_quat = quat_integrate(root_quat, w_pose, dt)

        next_v_b = quat_apply_inverse(next_root_quat, v_global)
        next_w_b = quat_apply_inverse(next_root_quat, w_global)
        world_down = torch.zeros_like(v_global)
        world_down[..., 2] = -1.0
        next_g_b = quat_apply_inverse(next_root_quat, world_down)

        next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
        next_rep = torch.cat(
            [next_root_pos[..., 2:3], next_g_b, next_v_b, next_w_b, next_qj, next_vj],
            dim=-1,
        )
        return next_aux, next_rep

    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        # Override root Z with rep_state's z
        root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
        v_global = quat_apply(root_quat, v_b)
        w_global = quat_apply(root_quat, w_b)
        parts = [root_pos, root_quat, qj, v_global, w_global, vj]
        return torch.cat(parts, dim=-1)

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # z, g_b and the root twist all follow from the integrated root pose.
        _, _, _, _, joint_pos, joint_vel = decompose_rep_state(rep_state)
        return {
            **_root_loss_blocks(aux_state),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def world_frame_bodies(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        pos, quat, lin_vel, ang_vel = self.fk(
            root_pos,
            root_quat,
            qj,
            quat_apply(root_quat, v_b),
            quat_apply(root_quat, w_b),
            vj,
        )
        return pos, matrix_from_quat(quat), lin_vel, ang_vel

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        _, _, _, _, qj, vj = decompose_rep_state(rep_state)
        return torch.cat([qj, vj], dim=-1)

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        del aux_state
        _, g_b, _, w_b, _, _ = decompose_rep_state(rep_state)
        qj, vj = joint_state.chunk(2, dim=-1)
        joint_pos_rel = qj - self._default_joint_pos
        if self._encoder_bias is not None:
            columns, sim_context = self._encoder_bias
            joint_pos_rel = joint_pos_rel + sim_context.denormalize(sim_ctx, columns)
        return torch.cat([g_b, w_b, joint_pos_rel, vj], dim=-1)


class MaxCoordRepresentation(StateRepresentation):
    """Maximal-coordinate per-body representation (following control_vae).

    Full state: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
    Rep state:  [local_pos(B*3), local_rot_6d(B*6), local_vel(B*3), local_avel(B*3),
                 root_height(1), up_dir(3)]
    Aux state:  [root_pos(3), root_quat(4)]
    Prediction: [vel_delta(B*3), avel_delta(B*3)] in root body frame
    """

    LOSS_WEIGHT_CONFIG = {
        **StateRepresentation.LOSS_WEIGHT_CONFIG,
        "body_pos": "nominal_error_body_pos",
        "body_rot": "nominal_error_body_rot",
        "body_vel": "nominal_error_body_vel",
        "body_avel": "nominal_error_body_avel",
    }

    def __init__(self, num_bodies: int, default_joint_pos: torch.Tensor):
        super().__init__()
        self.num_bodies = num_bodies
        self.full_state_dim = maxcoord_full_state_dim(num_bodies)
        self.rep_state_dim = maxcoord_rep_state_dim(num_bodies)
        self.prediction_dim = maxcoord_prediction_dim(num_bodies)
        self.num_joints = default_joint_pos.shape[0]
        self.register_buffer("_default_joint_pos", default_joint_pos, persistent=False)
        self.register_buffer("_gravity_w", torch.tensor([0.0, 0.0, -1.0]), persistent=False)
        self.analytic_ik: AnalyticIK  # set by bind
        self._encoder_bias: tuple[slice, SimulationContext] | None = None

    def bind(self, fk, soft_joint_pos_limits: torch.Tensor, sim_context: SimulationContext, reward_fn) -> None:
        super().bind(fk, soft_joint_pos_limits, sim_context, reward_fn)
        self.analytic_ik = fk.analytic_ik()
        columns = sim_context.find("encoder_bias")
        self._encoder_bias = None if columns is None else (columns, sim_context)

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        flat = full_state.reshape(-1, self.full_state_dim)
        rep = compute_maxcoord_rep_state(flat, self.num_bodies)
        return rep.reshape(*full_state.shape[:-1], -1)

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        B = self.num_bodies
        return torch.cat([full_state[..., :3], full_state[..., B * 3 : B * 3 + 4]], dim=-1)

    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        curr_rep_state: torch.Tensor,
        prediction: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate state by prediction deltas in root body frame.

        Integrates positions and rotations in the current root frame,
        then applies a frame-change rotation to the new root frame. Pose
        updates weight the delta by ``substep_coeff`` to match how the sim
        composes its substeps; velocity states advance by the full delta.
        """
        B = self.num_bodies
        c = self.substep_coeff
        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]

        local_pos, local_rot_6d, local_vel, local_avel, _, _ = decompose_maxcoord_rep_state(curr_rep_state, B)

        # Prediction deltas (in root body frame)
        dv_local = prediction[..., : B * 3].unflatten(-1, (B, 3))
        dw_local = prediction[..., B * 3 : B * 6].unflatten(-1, (B, 3))

        # --- Step 1: Apply deltas in local frame ---
        next_vel = local_vel + dv_local
        next_avel = local_avel + dw_local
        pose_vel = local_vel + c * dv_local
        pose_avel = local_avel + c * dw_local

        # --- Step 2: Integrate in current root frame ---
        # Positions (with drift correction: subtract root_pos_local)
        root_pos_local = local_pos[..., 0:1, :]
        rel_vel = pose_vel - pose_vel[..., 0:1, :]
        next_pos = (local_pos - root_pos_local) + rel_vel * dt
        # Rotations (per-body omega, shared trig for 2 rot6d cols)
        k_body, cos_body, sin_body = rodrigues_precompute(pose_avel * dt)
        cols = local_rot_6d.unflatten(-1, (3, 2))
        c1_mid = rodrigues_apply(cols[..., 0], k_body, cos_body, sin_body)
        c2_mid = rodrigues_apply(cols[..., 1], k_body, cos_body, sin_body)

        # --- Step 3: Root pose in world frame (with drift correction) ---
        next_root_pos = root_pos + quat_apply(
            root_quat,
            root_pos_local.squeeze(-2) + pose_vel[..., 0, :] * dt,
        )
        next_root_quat = quat_integrate(root_quat, quat_apply(root_quat, pose_avel[..., 0, :]), dt)

        # --- Step 4: Frame change (old root → new root) ---
        frame_omega = -pose_avel[..., 0:1, :] * dt
        k_f, cos_f, sin_f = rodrigues_precompute(frame_omega)

        next_local_pos = rodrigues_apply(next_pos, k_f, cos_f, sin_f)
        next_local_vel = rodrigues_apply(next_vel, k_f, cos_f, sin_f)
        next_local_avel = rodrigues_apply(next_avel, k_f, cos_f, sin_f)
        next_local_rot_6d = torch.stack(
            [
                rodrigues_apply(c1_mid, k_f, cos_f, sin_f),
                rodrigues_apply(c2_mid, k_f, cos_f, sin_f),
            ],
            dim=-1,
        ).flatten(-2, -1)

        # --- Step 5: Aux state & trailing fields ---
        world_down = torch.zeros_like(next_root_pos)
        world_down[..., 2] = -1.0
        next_up_dir = quat_apply_inverse(next_root_quat, world_down)

        next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
        next_rep = torch.cat(
            [
                next_local_pos.flatten(-2, -1),
                next_local_rot_6d.flatten(-2, -1),
                next_local_vel.flatten(-2, -1),
                next_local_avel.flatten(-2, -1),
                next_root_pos[..., 2:3],  # root_height
                next_up_dir,
            ],
            dim=-1,
        )
        return next_aux, next_rep

    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        return reconstruct_full_from_aux_and_rep(aux_state, rep_state, self.num_bodies)

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # Body 0 is the root: its pose is identically zero/identity and its twist is
        # the root twist. root_height/up_dir also follow from the root pose. All of
        # them are scored by root_pos/root_ori instead.
        pos, rot_6d, vel, avel, _, _ = decompose_maxcoord_rep_state(rep_state, self.num_bodies)
        return {
            **_root_loss_blocks(aux_state),
            "body_pos": pos[..., 1:, :].flatten(-2, -1),
            "body_rot": rot_6d[..., 1:, :].flatten(-2, -1),
            "body_vel": vel[..., 1:, :].flatten(-2, -1),
            "body_avel": avel[..., 1:, :].flatten(-2, -1),
        }

    def world_frame_bodies(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return world_frame_bodies(aux_state, rep_state, self.num_bodies)

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        _, rot6d, _, avel, _, _ = decompose_maxcoord_rep_state(rep_state, self.num_bodies)
        return torch.cat(self.analytic_ik(rot6d_to_matrix(rot6d), avel), dim=-1)

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        root_quat = aux_state[..., 3:7]
        g_w = self._gravity_w.expand(*aux_state.shape[:-1], 3)
        projected_gravity = quat_apply_inverse(root_quat, g_w)
        _, _, _, avel, _, _ = decompose_maxcoord_rep_state(rep_state, self.num_bodies)
        joint_pos, joint_vel = joint_state.chunk(2, dim=-1)
        joint_pos_rel = joint_pos - self._default_joint_pos
        if self._encoder_bias is not None:
            columns, sim_context = self._encoder_bias
            joint_pos_rel = joint_pos_rel + sim_context.denormalize(sim_ctx, columns)
        return torch.cat([projected_gravity, avel[..., 0, :], joint_pos_rel, joint_vel], -1)
