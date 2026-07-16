# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supported-lane physics contracts for the dVRK needle-pass task.

These tests require a fresh, load-qualified donor hold from the closed reset
state before exercising the runtime invariants and fixed hand-off traces. The
needle remains a free dynamic body throughout; no held state may rely on an
attachment or a post-reset state write.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from isaaclab.app import AppLauncher

# ``RecordVideo`` needs render products even when this module runs headless.
# Launch them only for an explicitly requested recording run so ordinary CUDA
# physics CI retains the renderer-free execution lane.
_RECORD_VIDEO = bool(os.environ.get("ISAACLAB_DVRK_NEEDLE_PASS_VIDEO_DIR"))
_app_launcher_kwargs = {"headless": True, "enable_cameras": _RECORD_VIDEO}
if _RECORD_VIDEO:
    # Fixed off-screen dimensions avoid capture/swapchain mismatches in
    # headless Isaac Sim, while RayTracedLighting is fast enough for this
    # short physics-verification trace.
    _video_renderer = os.environ.get("ISAACLAB_DVRK_NEEDLE_PASS_RENDERER", "RaytracedLighting")
    if _video_renderer not in {"RaytracedLighting", "PathTracing", "HydraStorm"}:
        raise ValueError("ISAACLAB_DVRK_NEEDLE_PASS_RENDERER must be RaytracedLighting, PathTracing, or HydraStorm")
    _app_launcher_kwargs.update(
        width=640,
        height=480,
        renderer=_video_renderer,
        rendering_mode="performance",
        anti_aliasing=0,
        denoiser=False,
    )
app_launcher = AppLauncher(**_app_launcher_kwargs)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import pytest
import torch
import warp as wp

import omni.usd
from pxr import PhysxSchema, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.utils.math as math_utils

if _RECORD_VIDEO:
    import carb

    _render_settings = carb.settings.get_settings()
    _render_settings.set_int("/rtx/post/tonemap/op", 4)
    _render_settings.set_float("/rtx/post/tonemap/filmIso", 200.0)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.needle_pass import assets
from isaaclab_tasks.contrib.needle_pass.config.dvrk.ik_abs_env_cfg import (
    DONOR_GRASP_JAW_POS,
    DVRK_HANDOFF_PHASE_CFG,
    DVRK_JAW_CHANNEL_T_T_C_POS_M,
    DVRK_JAW_CHANNEL_T_T_C_ROT_XYZW,
    ISAAC_GRASP_CANDIDATES_SHA256,
    LEFT_TOOL_HOME_POS_W,
    LEFT_TOOL_HOME_ROT_XYZW,
    RECEIVER_TOOL_TARGET_POS_W,
    RECEIVER_TOOL_TARGET_ROT_XYZW,
    RIGHT_TOOL_HOME_POS_W,
    RIGHT_TOOL_HOME_ROT_XYZW,
)
from isaaclab_tasks.contrib.needle_pass.mdp.terminations import (
    JAW_CONTACT_SENSOR_NAMES,
    HandoffMeasurements,
    HandoffPhase,
    get_handoff_phase_machine,
    jaw_needle_contact_measurements,
)
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_assets.robots.dvrk import (
    DVRK_PSM_ARM_JOINT_NAMES,
    DVRK_PSM_JAW_CLOSED_POS,
    DVRK_PSM_JAW_JOINT_NAMES,
    DVRK_PSM_JAW_OPEN_POS,
    DVRK_PSM_TOOL_TIP_BODY_NAME,
)

TASK_ID = "IsaacContrib-NeedlePass-dVRK-IK-Abs"
SEED = 42
TRACE_STEPS = 100
# Three seconds at the configured 240 Hz simulation rate. The normal CI lane
# remains short and renderer-free.
VIDEO_TRACE_STEPS = 720
NATIVE_RECEIVER_PREGRASP_CLEARANCE_M = 0.05
NATIVE_DONOR_HOLD_SETTLE_STEPS = 64
NATIVE_RECEIVER_APPROACH_SEGMENT_STEPS = 160
NATIVE_RECEIVER_CLOSE_SETTLE_STEPS = 480
NATIVE_DONOR_RELEASE_SETTLE_STEPS = 480
# The retained-lift motion is deliberately slower than the controller's 15 mm/s
# limit, while still requiring a 20 mm public-controller lift of the free body.
# The receiver also makes a 20 mm lateral escape from the static donor.  This
# keeps an already-released needle clear of the donor jaws; it is a smooth
# controller trajectory, not a modification of Isaac's generated grasp pose.
NATIVE_RECEIVER_LIFT_STEPS = 960
NATIVE_RECEIVER_LIFT_HEIGHT_M = 0.02
NATIVE_RECEIVER_TRANSFER_ESCAPE_POS_M = (0.02, 0.0, 0.0)
NATIVE_HANDOFF_TRACE_STEPS = (
    NATIVE_DONOR_HOLD_SETTLE_STEPS
    + 4 * NATIVE_RECEIVER_APPROACH_SEGMENT_STEPS
    + NATIVE_RECEIVER_CLOSE_SETTLE_STEPS
    + NATIVE_DONOR_RELEASE_SETTLE_STEPS
    + NATIVE_RECEIVER_LIFT_STEPS
)
RESET_FLOAT_ATOL = 1.0e-7

_ARTICULATION_STATE_WRITERS = (
    "write_root_pose_to_sim_index",
    "write_root_pose_to_sim_mask",
    "write_root_link_pose_to_sim_index",
    "write_root_link_pose_to_sim_mask",
    "write_root_com_pose_to_sim_index",
    "write_root_com_pose_to_sim_mask",
    "write_root_velocity_to_sim_index",
    "write_root_velocity_to_sim_mask",
    "write_root_com_velocity_to_sim_index",
    "write_root_com_velocity_to_sim_mask",
    "write_root_link_velocity_to_sim_index",
    "write_root_link_velocity_to_sim_mask",
    "write_joint_state_to_sim_index",
    "write_joint_state_to_sim_mask",
    "write_joint_position_to_sim_index",
    "write_joint_position_to_sim_mask",
    "write_joint_velocity_to_sim_index",
    "write_joint_velocity_to_sim_mask",
)
_RIGID_OBJECT_STATE_WRITERS = (
    "write_root_pose_to_sim_index",
    "write_root_pose_to_sim_mask",
    "write_root_link_pose_to_sim_index",
    "write_root_link_pose_to_sim_mask",
    "write_root_com_pose_to_sim_index",
    "write_root_com_pose_to_sim_mask",
    "write_root_velocity_to_sim_index",
    "write_root_velocity_to_sim_mask",
    "write_root_com_velocity_to_sim_index",
    "write_root_com_velocity_to_sim_mask",
    "write_root_link_velocity_to_sim_index",
    "write_root_link_velocity_to_sim_mask",
)
_ARTICULATION_PHYSX_STATE_SETTERS = (
    "set_root_transforms",
    "set_root_velocities",
    "set_dof_positions",
    "set_dof_velocities",
)
_ARTICULATION_PHYSX_CONTROL_SETTERS = (
    "set_dof_actuation_forces",
    "set_dof_position_targets",
    "set_dof_velocity_targets",
)
_RIGID_OBJECT_PHYSX_STATE_SETTERS = (
    "set_kinematic_targets",
    "set_transforms",
    "set_velocities",
)

_EXPECTED_RESET_STATE_WRITE_SEQUENCE = (
    ("asset", "left_psm", "write_joint_position_to_sim_index"),
    ("physx", "left_psm", "set_dof_positions"),
    ("asset", "left_psm", "write_joint_velocity_to_sim_index"),
    ("physx", "left_psm", "set_dof_velocities"),
    ("asset", "right_psm", "write_joint_position_to_sim_index"),
    ("physx", "right_psm", "set_dof_positions"),
    ("asset", "right_psm", "write_joint_velocity_to_sim_index"),
    ("physx", "right_psm", "set_dof_velocities"),
    ("asset", "needle", "write_root_pose_to_sim_index"),
    ("asset", "needle", "write_root_link_pose_to_sim_index"),
    ("physx", "needle", "set_transforms"),
    ("asset", "needle", "write_root_velocity_to_sim_index"),
    ("asset", "needle", "write_root_com_velocity_to_sim_index"),
    ("physx", "needle", "set_velocities"),
)
_RESET_STATE_WRITE_WHITELIST = frozenset(_EXPECTED_RESET_STATE_WRITE_SEQUENCE)


def _verification_video_dir() -> Path | None:
    """Return an opt-in directory for a recorded CUDA verification trace."""

    raw_path = os.environ.get("ISAACLAB_DVRK_NEEDLE_PASS_VIDEO_DIR")
    if not raw_path:
        return None
    video_dir = Path(raw_path).expanduser().resolve() / f"runtime-contract-{uuid4().hex}"
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


@contextmanager
def _task_env(
    num_envs: int,
    *,
    video_dir: Path | None = None,
    video_length: int = VIDEO_TRACE_STEPS,
    video_prefix: str = "dvrk-needle-pass-runtime-contract",
):
    """Construct one isolated headless task environment and always close it."""

    omni.usd.get_context().new_stage()
    env = None
    active_env = None
    try:
        env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=num_envs)
        env_cfg.seed = SEED
        if video_dir is None:
            env = gym.make(TASK_ID, cfg=env_cfg)
        else:
            env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
        env.unwrapped.sim._app_control_on_stop_handle = None
        active_env = env
        if video_dir is not None:
            active_env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                episode_trigger=lambda _episode_index: True,
                video_length=video_length,
                name_prefix=video_prefix,
                fps=round(1 / env.unwrapped.step_dt),
                disable_logger=True,
            )
        yield active_env if video_dir is not None else env.unwrapped
    finally:
        if env is not None:
            audit = getattr(env.unwrapped, "_needle_pass_direct_state_write_audit", None)
            if audit is not None:
                audit.uninstall()
            if active_env is not None:
                active_env.close()
            else:
                env.close()


def _held_start_action(env) -> torch.Tensor:
    """Hold the donor grasp and open receiver at their clone-local tool homes."""

    origins = env.scene.env_origins
    left_position_w = origins + torch.tensor(LEFT_TOOL_HOME_POS_W, device=env.device)
    right_position_w = origins + torch.tensor(RIGHT_TOOL_HOME_POS_W, device=env.device)
    left_orientation_xyzw = torch.tensor(LEFT_TOOL_HOME_ROT_XYZW, device=env.device).expand(env.num_envs, -1)
    right_orientation_xyzw = torch.tensor(RIGHT_TOOL_HOME_ROT_XYZW, device=env.device).expand(env.num_envs, -1)
    donor_held = torch.tensor(DONOR_GRASP_JAW_POS, device=env.device).expand(env.num_envs, -1)
    receiver_open = torch.tensor(DVRK_PSM_JAW_OPEN_POS, device=env.device).expand(env.num_envs, -1)
    action = torch.cat(
        (
            left_position_w,
            left_orientation_xyzw,
            donor_held,
            right_position_w,
            right_orientation_xyzw,
            receiver_open,
        ),
        dim=-1,
    )
    assert action.shape == (env.num_envs, 18)
    assert action.is_contiguous()
    assert torch.isfinite(action).all()
    return action


def _slerp_xyzw(start: torch.Tensor, end: torch.Tensor, fraction: float) -> torch.Tensor:
    """Interpolate orientations along the shortest physical rotation."""

    start = torch.nn.functional.normalize(start, dim=-1)
    end = torch.nn.functional.normalize(end, dim=-1)
    dot = torch.sum(start * end)
    # Quaternion signs encode the same orientation.  Flip the endpoint before
    # interpolation so the controller never crosses the near-zero quaternion
    # produced by a linear blend across a 171-degree rotation.
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = torch.clamp(dot, min=-1.0, max=1.0)
    angle = torch.arccos(dot)
    if float(angle) < 1.0e-6:
        return start
    sine = torch.sin(angle)
    weight_start = torch.sin((1.0 - fraction) * angle) / sine
    weight_end = torch.sin(fraction * angle) / sine
    return torch.nn.functional.normalize(weight_start * start + weight_end * end, dim=-1)


def _native_receiver_handoff_action(
    env,
    *,
    approach_fraction: float,
    receiver_jaw: tuple[float, float],
    donor_jaw: tuple[float, float] = DONOR_GRASP_JAW_POS,
    receiver_lift_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Command the fixed native receiver channel pose through the public ABI.

    ``RECEIVER_TOOL_TARGET_*`` is composed from the fixed Isaac grasp-generator
    candidate.  The only trajectory interpolation is the controller motion
    from the receiver home to that generated pose; no grasp pose is searched,
    perturbed, or written into the simulated state.
    """

    if not 0.0 <= approach_fraction <= 1.0:
        raise ValueError("approach_fraction must lie in [0, 1]")
    action = _held_start_action(env)
    fraction = torch.tensor(approach_fraction, device=env.device)
    home_position = torch.tensor(RIGHT_TOOL_HOME_POS_W, device=env.device)
    target_position = torch.tensor(RECEIVER_TOOL_TARGET_POS_W, device=env.device)
    target_position = target_position + torch.tensor(receiver_lift_offset_m, device=env.device)
    action[:, 9:12] = home_position + fraction * (target_position - home_position)

    home_orientation = torch.tensor(RIGHT_TOOL_HOME_ROT_XYZW, device=env.device)
    target_orientation = torch.tensor(RECEIVER_TOOL_TARGET_ROT_XYZW, device=env.device)
    action[:, 12:16] = _slerp_xyzw(home_orientation, target_orientation, approach_fraction)
    action[:, 7:9] = torch.tensor(donor_jaw, device=env.device)
    action[:, 16:18] = torch.tensor(receiver_jaw, device=env.device)
    return action


def _native_receiver_lift_offset(step: int) -> tuple[float, float, float]:
    """Return a zero-velocity-endpoint public-controller escape and lift."""

    if not 0 <= step < NATIVE_RECEIVER_LIFT_STEPS:
        raise ValueError("lift step is outside the configured trace")
    fraction = (step + 1) / NATIVE_RECEIVER_LIFT_STEPS
    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    return (
        NATIVE_RECEIVER_TRANSFER_ESCAPE_POS_M[0] * smooth_fraction,
        NATIVE_RECEIVER_TRANSFER_ESCAPE_POS_M[1] * smooth_fraction,
        NATIVE_RECEIVER_TRANSFER_ESCAPE_POS_M[2] * smooth_fraction + NATIVE_RECEIVER_LIFT_HEIGHT_M * smooth_fraction,
    )


def _native_donor_release_jaw(step: int) -> tuple[float, float]:
    """Open the donor through the public jaw action without a contact impulse.

    The receiver is already in a measured co-hold before this starts.  A
    monotonic controller ramp gives the physical recipient grasp time to take
    the full load; it does not alter either generated grasp pose or the needle
    state.
    """

    if not 0 <= step < NATIVE_DONOR_RELEASE_SETTLE_STEPS:
        raise ValueError("release step is outside the configured trace")
    fraction = (step + 1) / NATIVE_DONOR_RELEASE_SETTLE_STEPS
    return tuple(
        held + fraction * (opened - held)
        for held, opened in zip(DONOR_GRASP_JAW_POS, DVRK_PSM_JAW_OPEN_POS, strict=True)
    )


def _native_receiver_staged_approach_action(env, *, segment: int, fraction: float) -> torch.Tensor:
    """Move to a generated channel via a fixed collision-free pre-grasp path.

    The end pose is exactly ``RECEIVER_TOOL_TARGET_*`` reconstructed from the
    native candidate.  The elevated waypoints only keep the public controller
    out of the donor's occupied grasp volume while the recipient is open; they
    neither perturb the generated grasp nor write rigid-body state.
    """

    if segment not in range(4):
        raise ValueError("native receiver approach segment must be in [0, 3]")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("native receiver approach fraction must lie in [0, 1]")

    action = _held_start_action(env)
    home_position = torch.tensor(RIGHT_TOOL_HOME_POS_W, device=env.device)
    target_position = torch.tensor(RECEIVER_TOOL_TARGET_POS_W, device=env.device)
    clearance = torch.tensor((0.0, 0.0, NATIVE_RECEIVER_PREGRASP_CLEARANCE_M), device=env.device)
    home_high = home_position + clearance
    target_high = target_position + clearance
    home_orientation = torch.tensor(RIGHT_TOOL_HOME_ROT_XYZW, device=env.device)
    target_orientation = torch.tensor(RECEIVER_TOOL_TARGET_ROT_XYZW, device=env.device)

    if segment == 0:
        position = home_position + fraction * (home_high - home_position)
        orientation = home_orientation
    elif segment == 1:
        position = home_high
        orientation = _slerp_xyzw(home_orientation, target_orientation, fraction)
    elif segment == 2:
        position = home_high + fraction * (target_high - home_high)
        orientation = target_orientation
    else:
        position = target_high + fraction * (target_position - target_high)
        orientation = target_orientation

    action[:, 9:12] = position
    action[:, 12:16] = orientation
    return action


def _native_receiver_candidate_targets(env, candidate_poses_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose native ``T_N_C`` rows into physical world-frame tool targets."""

    if candidate_poses_n.shape != (env.num_envs, 7):
        raise ValueError("candidate poses must have shape (num_envs, 7)")
    needle = env.scene["needle"]
    needle_pos_w = needle.data.root_pos_w.torch
    needle_quat_w = needle.data.root_quat_w.torch
    channel_pos_t = torch.tensor(DVRK_JAW_CHANNEL_T_T_C_POS_M, device=env.device).expand(env.num_envs, -1)
    channel_quat_t = torch.tensor(DVRK_JAW_CHANNEL_T_T_C_ROT_XYZW, device=env.device).expand(env.num_envs, -1)
    channel_pos_n = candidate_poses_n[:, :3]
    channel_quat_n = candidate_poses_n[:, 3:]
    channel_quat_c_t = math_utils.quat_inv(channel_quat_t)
    channel_pos_c_t = -math_utils.quat_apply(channel_quat_c_t, channel_pos_t)
    tool_pos_n = channel_pos_n + math_utils.quat_apply(channel_quat_n, channel_pos_c_t)
    tool_quat_n = math_utils.quat_mul(channel_quat_n, channel_quat_c_t)
    tool_pos_w = needle_pos_w + math_utils.quat_apply(needle_quat_w, tool_pos_n)
    tool_quat_w = math_utils.quat_mul(needle_quat_w, tool_quat_n)
    return tool_pos_w, tool_quat_w


def _native_receiver_candidate_approach_action(
    env, receiver_pos_w: torch.Tensor, receiver_quat_xyzw: torch.Tensor, *, segment: int, fraction: float
) -> torch.Tensor:
    """Command a batched fixed pre-grasp path to unmodified native candidates."""

    if receiver_pos_w.shape != (env.num_envs, 3) or receiver_quat_xyzw.shape != (env.num_envs, 4):
        raise ValueError("batched receiver targets must match the environment count")
    if segment not in range(4) or not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid native receiver approach segment or fraction")
    action = _held_start_action(env)
    home_pos_w = env.scene.env_origins + torch.tensor(RIGHT_TOOL_HOME_POS_W, device=env.device)
    home_quat_xyzw = torch.tensor(RIGHT_TOOL_HOME_ROT_XYZW, device=env.device).expand(env.num_envs, -1)
    clearance = torch.tensor((0.0, 0.0, NATIVE_RECEIVER_PREGRASP_CLEARANCE_M), device=env.device)
    home_high = home_pos_w + clearance
    target_high = receiver_pos_w + clearance
    if segment == 0:
        position = home_pos_w + fraction * (home_high - home_pos_w)
        orientation = home_quat_xyzw
    elif segment == 1:
        position = home_high
        orientation = torch.stack(
            [_slerp_xyzw(home_quat_xyzw[index], receiver_quat_xyzw[index], fraction) for index in range(env.num_envs)]
        )
    elif segment == 2:
        position = home_high + fraction * (target_high - home_high)
        orientation = receiver_quat_xyzw
    else:
        position = target_high + fraction * (receiver_pos_w - target_high)
        orientation = receiver_quat_xyzw
    action[:, 9:12] = position
    action[:, 12:16] = orientation
    return action


def _native_receiver_candidate_handoff_action(
    env,
    receiver_pos_w: torch.Tensor,
    receiver_quat_xyzw: torch.Tensor,
    *,
    donor_jaw: tuple[float, float],
    receiver_jaw: tuple[float, float],
    receiver_lift_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Command exact generated channels for the batched physical screen."""

    if receiver_pos_w.shape != (env.num_envs, 3) or receiver_quat_xyzw.shape != (env.num_envs, 4):
        raise ValueError("batched receiver targets must match the environment count")
    action = _held_start_action(env)
    action[:, 9:12] = receiver_pos_w + torch.tensor(receiver_lift_offset_m, device=env.device)
    action[:, 12:16] = receiver_quat_xyzw
    action[:, 7:9] = torch.tensor(donor_jaw, device=env.device)
    action[:, 16:18] = torch.tensor(receiver_jaw, device=env.device)
    return action


def _pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return a column-vector homogeneous transform from a PhysX body pose."""

    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    transform[:3, 3] = position
    return transform


def _receiver_collision_channel_world(env, env_index: int) -> np.ndarray:
    """Measure the midpoint of the two live receiver collision shapes.

    USD intentionally retains authoring transforms while PhysX integrates the
    articulation.  We therefore compose the static shape-within-link transform
    with the current tensor body pose, rather than reading an authored world
    transform as though it described the live collision geometry.
    """

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    receiver = env.scene["right_psm"]
    body_ids, body_names = receiver.find_bodies(["psm_tool_gripper1_link", "psm_tool_gripper2_link"])
    assert len(body_ids) == 2, body_names
    centres = []
    for body_id, link_name in zip(body_ids, body_names, strict=True):
        link_path = f"{env.scene.env_prim_paths[env_index]}/RightPSM/{link_name}"
        collision_path = f"{link_path}/collisions_xform/collisions"
        authored_link_w = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(link_path)), dtype=np.float64).T
        authored_collision_w = np.asarray(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(collision_path)), dtype=np.float64
        ).T
        link_to_collision = np.linalg.inv(authored_link_w) @ authored_collision_w
        body_pose_w = _pose_matrix(
            receiver.data.body_pos_w.torch[env_index, body_id].detach().cpu().numpy(),
            receiver.data.body_quat_w.torch[env_index, body_id].detach().cpu().numpy(),
        )
        centres.append((body_pose_w @ link_to_collision)[:3, 3])
    return np.mean(np.stack(centres), axis=0)


def _native_probe_candidate_poses() -> tuple[tuple[int, ...], torch.Tensor]:
    """Load hash-pinned native rows as ``[xyz, qx, qy, qz, qw]`` poses."""

    candidate_path = os.environ.get("ISAACLAB_DVRK_NATIVE_CANDIDATES_CSV")
    indices_raw = os.environ.get("ISAACLAB_DVRK_NATIVE_RECEIVER_PROBE_INDICES")
    if not candidate_path or not indices_raw:
        pytest.skip("native receiver feasibility probe is opt-in")
    candidate_file = Path(candidate_path)
    digest = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    assert digest == ISAAC_GRASP_CANDIDATES_SHA256
    indices = tuple(int(value) for value in indices_raw.split(","))
    assert indices and len(indices) == len(set(indices))
    with candidate_file.open(encoding="utf-8", newline="") as candidate_stream:
        rows = {int(row["candidate_index"]): row for row in csv.DictReader(candidate_stream)}
    assert set(indices) <= rows.keys()
    # The pinned generator CSV stores each scalar-first quaternion in distinct
    # qw/qx/qy/qz columns. Assemble those named fields explicitly in Isaac
    # Lab's scalar-last xyzw order without altering the generated rotation.
    poses = torch.tensor(
        [
            [
                float(rows[index]["needle_channel_x_m"]),
                float(rows[index]["needle_channel_y_m"]),
                float(rows[index]["needle_channel_z_m"]),
                float(rows[index]["needle_channel_qx"]),
                float(rows[index]["needle_channel_qy"]),
                float(rows[index]["needle_channel_qz"]),
                float(rows[index]["needle_channel_qw"]),
            ]
            for index in indices
        ],
        dtype=torch.float32,
    )
    assert torch.isfinite(poses).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(poses[:, 3:], dim=-1),
        torch.ones(len(indices), dtype=poses.dtype),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    return indices, poses


def _handoff_diagnostics(env, phase_machine) -> dict[str, object]:
    """Return measured state when a physical hand-off assertion fails."""

    loads, reaction_normals_w, raw_forces_w = jaw_needle_contact_measurements(env)
    receiver = env.scene["right_psm"]
    body_ids, body_names = receiver.find_bodies(DVRK_PSM_TOOL_TIP_BODY_NAME)
    assert len(body_ids) == 1, body_names
    needle_pos_r, needle_quat_r = math_utils.subtract_frame_transforms(
        receiver.data.body_pos_w.torch[:, body_ids[0]],
        receiver.data.body_quat_w.torch[:, body_ids[0]],
        env.scene["needle"].data.root_pos_w.torch,
        env.scene["needle"].data.root_quat_w.torch,
    )
    target_pos_r = torch.tensor(DVRK_HANDOFF_PHASE_CFG.receiver_relative_position_target_m, device=env.device)
    target_quat_r = torch.tensor(
        DVRK_HANDOFF_PHASE_CFG.receiver_relative_orientation_target_xyzw, device=env.device
    ).expand_as(needle_quat_r)
    needle = env.scene["needle"]
    receiver_pose_w = torch.cat(
        (receiver.data.body_pos_w.torch[:, body_ids[0]], receiver.data.body_quat_w.torch[:, body_ids[0]]), dim=-1
    )
    measurements = HandoffMeasurements(
        normal_forces_n=loads,
        reaction_normals_w=reaction_normals_w,
        needle_pose_w=torch.cat((needle.data.root_pos_w.torch, needle.data.root_quat_w.torch), dim=-1),
        needle_velocity_w=torch.cat((needle.data.root_lin_vel_w.torch, needle.data.root_ang_vel_w.torch), dim=-1),
        receiver_pose_w=receiver_pose_w,
    )
    return {
        "phase": int(phase_machine.phase.item()),
        "loads_n": loads.detach().cpu().tolist(),
        "reaction_normals_w": reaction_normals_w.detach().cpu().tolist(),
        "raw_forces_w_n": raw_forces_w.detach().cpu().tolist(),
        "jaw_normal_dots": torch.sum(
            torch.nn.functional.normalize(reaction_normals_w[:, 0::2], dim=-1)
            * torch.nn.functional.normalize(reaction_normals_w[:, 1::2], dim=-1),
            dim=-1,
        )
        .detach()
        .cpu()
        .tolist(),
        "donor_bilateral": phase_machine._bilateral_contact(
            loads[:, :2], reaction_normals_w[:, :2], phase_machine._donor_engaged
        )
        .detach()
        .cpu()
        .tolist(),
        "receiver_bilateral": phase_machine._bilateral_contact(
            loads[:, 2:], reaction_normals_w[:, 2:], phase_machine._receiver_engaged
        )
        .detach()
        .cpu()
        .tolist(),
        "receiver_bounds": phase_machine._receiver_bounds(measurements).detach().cpu().tolist(),
        "receiver_tool_pos_w": receiver.data.body_pos_w.torch[:, body_ids[0]].detach().cpu().tolist(),
        "receiver_tool_quat_xyzw": receiver.data.body_quat_w.torch[:, body_ids[0]].detach().cpu().tolist(),
        "receiver_joint_pos": receiver.data.joint_pos.torch.detach().cpu().tolist(),
        "donor_joint_pos": env.scene["left_psm"].data.joint_pos.torch.detach().cpu().tolist(),
        "needle_pos_w": env.scene["needle"].data.root_pos_w.torch.detach().cpu().tolist(),
        "needle_quat_xyzw": env.scene["needle"].data.root_quat_w.torch.detach().cpu().tolist(),
        "needle_velocity_w": torch.cat(
            (env.scene["needle"].data.root_lin_vel_w.torch, env.scene["needle"].data.root_ang_vel_w.torch), dim=-1
        )
        .detach()
        .cpu()
        .tolist(),
        "receiver_relative_pos_m": needle_pos_r.detach().cpu().tolist(),
        "receiver_relative_position_error_m": torch.linalg.vector_norm(needle_pos_r - target_pos_r, dim=-1)
        .detach()
        .cpu()
        .tolist(),
        "receiver_relative_orientation_error_rad": math_utils.quat_error_magnitude(needle_quat_r, target_quat_r)
        .detach()
        .cpu()
        .tolist(),
    }


def _clone_tree(value: Any) -> Any:
    """Clone a nested tensor tree for a reset determinism comparison."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(child) for child in value)
    if isinstance(value, list):
        return [_clone_tree(child) for child in value]
    return value


def _assert_finite_tree(value: Any, path: str = "observations") -> None:
    """Require every floating-point tensor in a nested observation tree to be finite."""

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            assert torch.isfinite(value).all(), f"non-finite tensor at {path}"
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
        return
    raise AssertionError(f"unsupported observation leaf at {path}: {type(value).__name__}")


def _assert_same_tree(actual: Any, expected: Any, path: str = "state") -> None:
    """Compare two reset snapshots with one declared floating-point tolerance."""

    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor), f"type mismatch at {path}"
        if actual.is_floating_point() or actual.is_complex():
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=RESET_FLOAT_ATOL,
                msg=lambda message: f"{path}: {message}",
            )
        else:
            assert torch.equal(actual, expected), f"tensor mismatch at {path}"
        return
    if isinstance(actual, Mapping):
        assert isinstance(expected, Mapping), f"type mismatch at {path}"
        assert actual.keys() == expected.keys(), f"key mismatch at {path}"
        for key in actual:
            _assert_same_tree(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, (tuple, list)):
        assert isinstance(expected, type(actual)), f"type mismatch at {path}"
        assert len(actual) == len(expected), f"length mismatch at {path}"
        for index, (actual_child, expected_child) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_tree(actual_child, expected_child, f"{path}[{index}]")
        return
    assert actual == expected, f"value mismatch at {path}"


def _reset_snapshot(env, observations: Mapping[str, Any]) -> dict[str, Any]:
    """Capture reset-visible observations and all directly reset dynamic state."""

    return _clone_tree(
        {
            "observations": observations,
            "left_joint_pos": env.scene["left_psm"].data.joint_pos.torch,
            "left_joint_vel": env.scene["left_psm"].data.joint_vel.torch,
            "right_joint_pos": env.scene["right_psm"].data.joint_pos.torch,
            "right_joint_vel": env.scene["right_psm"].data.joint_vel.torch,
            "needle_root_link_pose_w": env.scene["needle"].data.root_link_pose_w.torch,
            "needle_root_com_velocity_w": env.scene["needle"].data.root_com_vel_w.torch,
        }
    )


class _DirectStateWriteAudit:
    """Reject high- and low-level state writes outside a reset event.

    Articulation effort and drive-target setters are recorded separately and
    deliberately remain usable: those are the physical control path exercised
    by every environment step, not instantaneous state mutation.
    """

    def __init__(self, env):
        self._env = env
        self._reset_depth = 0
        self._active_reset_calls: list[tuple[str, str, str]] | None = None
        self.reset_windows: list[tuple[tuple[str, str, str], ...]] = []
        self.asset_state_calls: list[tuple[str, str]] = []
        self.physx_state_calls: list[tuple[str, str]] = []
        self.physx_control_calls: list[tuple[str, str]] = []
        self.violations: list[tuple[str, str, str]] = []
        self.usd_mutations: list[tuple[str, str]] = []
        self._usd_notice_registration = None

    def install(self) -> None:
        """Wrap the event dispatcher and primitive state-writer methods."""

        if getattr(self._env, "_needle_pass_direct_state_write_audit", None) is not None:
            raise RuntimeError("only one direct-state audit may be installed per environment")
        self._env._needle_pass_direct_state_write_audit = self

        original_apply = self._env.event_manager.apply

        def audited_apply(mode: str, *args, **kwargs):
            if mode != "reset":
                return original_apply(mode, *args, **kwargs)
            assert self._reset_depth == 0, "nested reset event windows are not permitted"
            self._reset_depth += 1
            self._active_reset_calls = []
            try:
                result = original_apply(mode, *args, **kwargs)
            except BaseException:
                self._active_reset_calls = None
                raise
            finally:
                self._reset_depth -= 1
            reset_calls = tuple(self._active_reset_calls)
            self._active_reset_calls = None
            if reset_calls != _EXPECTED_RESET_STATE_WRITE_SEQUENCE:
                self.violations.append(("reset", "window", "unexpected_write_sequence"))
                raise AssertionError(
                    "reset state-write sequence differs from the documented whitelist:\n"
                    f"expected={_EXPECTED_RESET_STATE_WRITE_SEQUENCE!r}\nactual={reset_calls!r}"
                )
            self.reset_windows.append(reset_calls)
            return result

        self._env.event_manager.apply = audited_apply
        for asset_name in ("left_psm", "right_psm"):
            asset = self._env.scene[asset_name]
            self._wrap_guarded_methods(asset, "asset", asset_name, _ARTICULATION_STATE_WRITERS)
            self._wrap_guarded_methods(
                asset.root_view,
                "physx",
                asset_name,
                _ARTICULATION_PHYSX_STATE_SETTERS,
            )
            self._wrap_control_methods(asset.root_view, asset_name, _ARTICULATION_PHYSX_CONTROL_SETTERS)

        needle = self._env.scene["needle"]
        self._wrap_guarded_methods(needle, "asset", "needle", _RIGID_OBJECT_STATE_WRITERS)
        self._wrap_guarded_methods(
            needle.root_view,
            "physx",
            "needle",
            _RIGID_OBJECT_PHYSX_STATE_SETTERS,
        )
        stage = omni.usd.get_context().get_stage()
        self._usd_notice_registration = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._record_usd_changes, stage)

    def uninstall(self) -> None:
        """Revoke the stage notice before Isaac Lab tears the scene down."""

        if self._usd_notice_registration is not None:
            self._usd_notice_registration.Revoke()
            self._usd_notice_registration = None
        if getattr(self._env, "_needle_pass_direct_state_write_audit", None) is self:
            del self._env._needle_pass_direct_state_write_audit

    def _wrap_guarded_methods(
        self,
        owner: Any,
        layer: str,
        asset_name: str,
        method_names: tuple[str, ...],
    ) -> None:
        calls = self.asset_state_calls if layer == "asset" else self.physx_state_calls
        for method_name in method_names:
            original_method = getattr(owner, method_name)

            def guarded_writer(
                *args,
                _method=original_method,
                _label=(asset_name, method_name),
                _layer=layer,
                **kwargs,
            ):
                calls.append(_label)
                call = (_layer, *_label)
                if self._active_reset_calls is not None:
                    self._active_reset_calls.append(call)
                if self._reset_depth == 0 or call not in _RESET_STATE_WRITE_WHITELIST:
                    violation = call
                    self.violations.append(violation)
                    location = "outside reset" if self._reset_depth == 0 else "outside reset whitelist"
                    raise AssertionError(f"direct state write {location}: {_layer}:{_label[0]}.{_label[1]}")
                return _method(*args, **kwargs)

            setattr(owner, method_name, guarded_writer)

    def _wrap_control_methods(self, owner: Any, asset_name: str, method_names: tuple[str, ...]) -> None:
        for method_name in method_names:
            original_method = getattr(owner, method_name)

            def allowed_control(*args, _method=original_method, _label=(asset_name, method_name), **kwargs):
                self.physx_control_calls.append(_label)
                return _method(*args, **kwargs)

            setattr(owner, method_name, allowed_control)

    def _record_usd_changes(self, notice: Usd.Notice.ObjectsChanged, stage: Usd.Stage) -> None:
        """Record forbidden authored USD edits at any time after installation."""

        needle_root_path = _needle_root_path(self._env)
        for change_kind, paths in (
            ("resync", notice.GetResyncedPaths()),
            ("info", notice.GetChangedInfoOnlyPaths()),
        ):
            for path in paths:
                prim_path = path.GetPrimPath()
                prim = stage.GetPrimAtPath(prim_path)
                if prim_path.HasPrefix(Sdf.Path(needle_root_path)):
                    self.usd_mutations.append((change_kind, str(path)))
                    continue
                if not prim.IsValid():
                    continue
                if UsdPhysics.Joint(prim) or _prim_is_attachment(prim):
                    self.usd_mutations.append((change_kind, str(path)))
                    continue
                if PhysxSchema.PhysxPhysicsJointInstancer(prim) or any(
                    "Attachment" in schema_name for schema_name in prim.GetAppliedSchemas()
                ):
                    self.usd_mutations.append((change_kind, str(path)))
                    continue
                usd_property = stage.GetPropertyAtPath(path)
                if isinstance(usd_property, Usd.Relationship) and any(
                    _path_is_in_needle(target, prim, needle_root_path) for target in _relationship_targets(usd_property)
                ):
                    self.usd_mutations.append((change_kind, str(path)))


def _path_is_in_needle(target: Sdf.Path, owner_prim: Usd.Prim, needle_root_path: str) -> bool:
    """Return whether a relationship target addresses the needle subtree."""

    if not target.IsAbsolutePath():
        target = target.MakeAbsolutePath(owner_prim.GetPath())
    target_prim_path = target.GetPrimPath()
    needle_root = Sdf.Path(needle_root_path)
    return target_prim_path == needle_root or target_prim_path.HasPrefix(needle_root)


def _relationship_targets(relationship: Usd.Relationship) -> tuple[Sdf.Path, ...]:
    """Return unique direct and forwarded relationship targets."""

    targets: dict[str, Sdf.Path] = {}
    for target in (*relationship.GetTargets(), *relationship.GetForwardedTargets()):
        targets[str(target)] = target
    return tuple(targets.values())


def _needle_root_path(env, env_index: int = 0) -> str:
    """Resolve one cloned needle root from the configured scene prim path."""

    needle_prim_name = env.cfg.scene.needle.prim_path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    assert needle_prim_name and not any(token in needle_prim_name for token in ("*", "[", "]", "{", "}"))
    return f"{env.scene.env_prim_paths[env_index]}/{needle_prim_name}"


def _prim_is_in_needle(prim: Usd.Prim, needle_root_path: str) -> bool:
    prim_path = str(prim.GetPath())
    return prim_path == needle_root_path or prim_path.startswith(f"{needle_root_path}/")


def _prim_is_attachment(prim: Usd.Prim) -> bool:
    """Return whether a prim type or applied schema represents an attachment."""

    return "Attachment" in str(prim.GetTypeName()) or any(
        "Attachment" in schema_name for schema_name in prim.GetAppliedSchemas()
    )


def _assert_live_needle_topology(env) -> None:
    """Require one free dynamic body with no joint, attachment, or relationship constraint."""

    stage = omni.usd.get_context().get_stage()
    needle_root_path = _needle_root_path(env)
    needle_root = stage.GetPrimAtPath(needle_root_path)
    assert needle_root.IsValid()
    assert str(needle_root.GetParent().GetPath()) == env.scene.env_prim_paths[0]

    needle_prims = list(Usd.PrimRange(needle_root))
    rigid_prims = [prim for prim in needle_prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(rigid_prims) == 1
    assert not any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in needle_prims)

    rigid_body = UsdPhysics.RigidBodyAPI(rigid_prims[0])
    assert rigid_body.GetRigidBodyEnabledAttr().Get() is True
    assert rigid_body.GetKinematicEnabledAttr().Get() is False

    attached_joints: list[tuple[str, str, tuple[str, ...]]] = []
    attached_fixed_joints: list[str] = []
    physics_attachments: list[tuple[str, tuple[str, ...]]] = []
    physics_joint_instancers: list[tuple[str, tuple[str, ...]]] = []
    attachment_apis: list[tuple[str, str]] = []
    inbound_relationships: list[tuple[str, str, tuple[str, ...]]] = []
    for prim in stage.TraverseAll():
        joint = UsdPhysics.Joint(prim)
        if joint:
            targets = (
                *_relationship_targets(joint.GetBody0Rel()),
                *_relationship_targets(joint.GetBody1Rel()),
            )
            if _prim_is_in_needle(prim, needle_root_path) or any(
                _path_is_in_needle(target, prim, needle_root_path) for target in targets
            ):
                attached_joints.append(
                    (str(prim.GetPath()), str(prim.GetTypeName()), tuple(str(target) for target in targets))
                )
                if UsdPhysics.FixedJoint(prim):
                    attached_fixed_joints.append(str(prim.GetPath()))

        if _prim_is_attachment(prim):
            targets = tuple(
                target for relationship in prim.GetRelationships() for target in _relationship_targets(relationship)
            )
            if _prim_is_in_needle(prim, needle_root_path) or any(
                _path_is_in_needle(target, prim, needle_root_path) for target in targets
            ):
                physics_attachments.append((str(prim.GetPath()), tuple(str(target) for target in targets)))

        joint_instancer = PhysxSchema.PhysxPhysicsJointInstancer(prim)
        if joint_instancer:
            targets = (
                *_relationship_targets(joint_instancer.GetPhysicsBody0sRel()),
                *_relationship_targets(joint_instancer.GetPhysicsBody1sRel()),
            )
            if _prim_is_in_needle(prim, needle_root_path) or any(
                _path_is_in_needle(target, prim, needle_root_path) for target in targets
            ):
                physics_joint_instancers.append((str(prim.GetPath()), tuple(str(target) for target in targets)))

        if _prim_is_in_needle(prim, needle_root_path):
            for schema_name in prim.GetAppliedSchemas():
                if "Attachment" in schema_name:
                    attachment_apis.append((str(prim.GetPath()), schema_name))

        for relationship in prim.GetRelationships():
            targets = _relationship_targets(relationship)
            if not any(_path_is_in_needle(target, prim, needle_root_path) for target in targets):
                continue
            relationship_name = str(relationship.GetName())
            if _prim_is_in_needle(prim, needle_root_path) and relationship_name.startswith("material:binding"):
                continue
            inbound_relationships.append(
                (str(prim.GetPath()), relationship_name, tuple(str(target) for target in targets))
            )

    assert attached_joints == []
    assert attached_fixed_joints == []
    assert physics_attachments == []
    assert physics_joint_instancers == []
    assert attachment_apis == []
    assert inbound_relationships == []

    view_paths = tuple(str(path) for path in env.scene["needle"].root_view.prim_paths)
    assert len(view_paths) == 1
    assert view_paths[0] == str(rigid_prims[0].GetPath())


def _assert_live_needle_mass_and_material(env) -> None:
    """Check PhysX-resolved values and every collision's strong task binding."""

    needle = env.scene["needle"]
    root_view = needle.root_view
    coms = wp.to_torch(root_view.get_coms())
    assert coms.shape == (root_view.count, 7)
    expected_com_position = torch.tensor(
        assets.NEEDLE_CENTRE_OF_MASS_BODY_LOCAL_M,
        device=coms.device,
        dtype=coms.dtype,
    ).expand(coms.shape[0], -1)
    torch.testing.assert_close(coms[:, :3], expected_com_position, rtol=0.0, atol=5.0e-9)

    masses = wp.to_torch(root_view.get_masses())
    assert root_view.count == env.num_envs
    assert masses.shape == (root_view.count, 1)
    torch.testing.assert_close(
        masses,
        torch.full_like(masses, assets.NEEDLE_MASS_KG),
        rtol=1.0e-6,
        atol=1.0e-10,
    )

    live_materials = wp.to_torch(root_view.get_material_properties())
    assert live_materials.shape == (
        root_view.count,
        root_view.max_shapes,
        3,
    )
    assert root_view.max_shapes > 0
    expected_material = torch.tensor(
        (assets.NEEDLE_STATIC_FRICTION, assets.NEEDLE_DYNAMIC_FRICTION, assets.NEEDLE_RESTITUTION),
        device=live_materials.device,
        dtype=live_materials.dtype,
    )
    torch.testing.assert_close(
        live_materials,
        expected_material.expand_as(live_materials),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    stage = omni.usd.get_context().get_stage()
    needle_root_path = _needle_root_path(env)
    material_path = f"{needle_root_path}/physicsMaterial"
    material_prim = stage.GetPrimAtPath(material_path)
    assert material_prim.IsValid()
    assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(assets.NEEDLE_STATIC_FRICTION)
    assert material_prim.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(assets.NEEDLE_DYNAMIC_FRICTION)
    assert material_prim.GetAttribute("physics:restitution").Get() == pytest.approx(assets.NEEDLE_RESTITUTION)
    assert material_prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == assets.NEEDLE_FRICTION_COMBINE_MODE
    assert (
        material_prim.GetAttribute("physxMaterial:restitutionCombineMode").Get()
        == assets.NEEDLE_RESTITUTION_COMBINE_MODE
    )

    needle_root = stage.GetPrimAtPath(needle_root_path)
    collision_prims = [prim for prim in Usd.PrimRange(needle_root) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert collision_prims
    for collision_prim in collision_prims:
        binding_api = UsdShade.MaterialBindingAPI(collision_prim)
        binding = binding_api.GetDirectBinding("physics")
        assert binding.GetMaterialPath() == Sdf.Path(material_path)
        assert binding.GetMaterialPurpose() == "physics"
        bound_material, winning_relationship = binding_api.ComputeBoundMaterial("physics")
        assert bound_material.GetPath() == Sdf.Path(material_path)
        assert (
            UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(winning_relationship)
            == UsdShade.Tokens.strongerThanDescendants
        )


def _assert_contact_sensor_matrices(env) -> None:
    """Require all four filtered jaw sensors to expose finite ``N x 1 x 1 x 3`` forces."""

    assert len(JAW_CONTACT_SENSOR_NAMES) == 4
    for sensor_name in JAW_CONTACT_SENSOR_NAMES:
        sensor = env.scene.sensors[sensor_name]
        assert sensor.contact_view.filter_count == 1
        force_matrix_w = sensor.data.force_matrix_w.torch
        assert force_matrix_w is not None
        assert force_matrix_w.shape == (env.num_envs, 1, 1, 3)
        assert torch.isfinite(force_matrix_w).all(), f"non-finite contact matrix for {sensor_name}"


def _assert_live_action_and_joint_order(env) -> None:
    """Require the runtime action manager to preserve the public 18-D ABI."""

    expected_terms = ["left_arm_action", "left_jaw_action", "right_arm_action", "right_jaw_action"]
    assert env.action_manager.active_terms == expected_terms
    assert env.action_manager.action_term_dim == [7, 2, 7, 2]
    assert env.action_manager.total_action_dim == 18
    assert env.action_manager.action.shape == (env.num_envs, 18)

    for side in ("left", "right"):
        arm_term = env.action_manager._terms[f"{side}_arm_action"]
        jaw_term = env.action_manager._terms[f"{side}_jaw_action"]
        assert tuple(arm_term._joint_names) == tuple(DVRK_PSM_ARM_JOINT_NAMES)
        assert tuple(jaw_term._joint_names) == tuple(DVRK_PSM_JAW_JOINT_NAMES)


def _assert_live_psm_jaw_contract(env) -> None:
    """Check resolved jaw limits and the pinned jaw collision material."""

    stage = omni.usd.get_context().get_stage()
    expected_material_values = {
        "physics:staticFriction": 1.0,
        "physics:dynamicFriction": 10.0,
        "physics:restitution": 0.0,
    }
    for asset_name, prim_name in (("left_psm", "LeftPSM"), ("right_psm", "RightPSM")):
        articulation = env.scene[asset_name]
        jaw_ids, jaw_names = articulation.find_joints(list(DVRK_PSM_JAW_JOINT_NAMES), preserve_order=True)
        assert tuple(jaw_names) == tuple(DVRK_PSM_JAW_JOINT_NAMES)
        limits = articulation.data.joint_pos_limits.torch[:, jaw_ids, :]
        expected_limits = torch.tensor(
            ((-math.pi / 6.0, 0.0), (0.0, math.pi / 6.0)),
            device=limits.device,
            dtype=limits.dtype,
        ).expand(env.num_envs, -1, -1)
        torch.testing.assert_close(limits, expected_limits, rtol=0.0, atol=1.0e-6)
        for endpoint in (DVRK_PSM_JAW_OPEN_POS, DVRK_PSM_JAW_CLOSED_POS):
            endpoint_tensor = torch.tensor(endpoint, device=limits.device, dtype=limits.dtype)
            assert torch.all(endpoint_tensor >= limits[0, :, 0])
            assert torch.all(endpoint_tensor <= limits[0, :, 1])

        psm_root_path = f"{env.scene.env_prim_paths[0]}/{prim_name}"
        material_path = f"{psm_root_path}/Looks/PhysicsMaterial"
        material_prim = stage.GetPrimAtPath(material_path)
        assert material_prim.IsValid()
        for attribute_name, expected_value in expected_material_values.items():
            assert material_prim.GetAttribute(attribute_name).Get() == pytest.approx(expected_value)
        # The pinned PSM does not author a combine mode. PhysX therefore uses
        # its default average mode for the jaw material; the needle's explicit
        # min mode wins the pair and yields the declared resolved coefficients.
        assert not material_prim.GetAttribute("physxMaterial:frictionCombineMode").HasAuthoredValueOpinion()

        for jaw_link in ("psm_tool_gripper1_link", "psm_tool_gripper2_link"):
            collision_path = f"{psm_root_path}/{jaw_link}/collisions_xform/collisions"
            collision_prim = stage.GetPrimAtPath(collision_path)
            assert collision_prim.IsValid()
            assert collision_prim.HasAPI(UsdPhysics.CollisionAPI)
            binding = UsdShade.MaterialBindingAPI(collision_prim).GetDirectBinding("physics")
            assert binding.GetMaterialPath() == Sdf.Path(material_path)


def _assert_static_suture_pad(env) -> None:
    """Require the remote pad asset to remain a static collider outside the proof region."""

    stage = omni.usd.get_context().get_stage()
    pad_path = f"{env.scene.env_prim_paths[0]}/SuturePad"
    pad_root = stage.GetPrimAtPath(pad_path)
    assert pad_root.IsValid()
    pad_prims = list(Usd.PrimRange(pad_root))
    assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in pad_prims)
    assert not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in pad_prims)

    pad_position = env.cfg.scene.suture_pad.init_state.pos
    needle_position = env.cfg.scene.needle.init_state.pos
    assert pad_position[2] < needle_position[2]
    assert torch.linalg.vector_norm(torch.tensor(pad_position[:2]) - torch.tensor(needle_position[:2])) > 0.25


def _assert_configured_tool_homes(env) -> None:
    """Compare both simulated tool tips with their shared configuration constants."""

    for asset_name, position, orientation_xyzw in (
        ("left_psm", LEFT_TOOL_HOME_POS_W, LEFT_TOOL_HOME_ROT_XYZW),
        ("right_psm", RIGHT_TOOL_HOME_POS_W, RIGHT_TOOL_HOME_ROT_XYZW),
    ):
        articulation = env.scene[asset_name]
        body_ids, body_names = articulation.find_bodies(DVRK_PSM_TOOL_TIP_BODY_NAME)
        assert len(body_ids) == 1, body_names
        actual_position = articulation.data.body_pos_w.torch[:, body_ids[0], :]
        expected_position = env.scene.env_origins + torch.tensor(position, device=env.device)
        torch.testing.assert_close(actual_position, expected_position, rtol=0.0, atol=2.0e-6)

        expected_orientation = torch.tensor(orientation_xyzw, device=env.device).expand(env.num_envs, -1)
        actual_orientation = torch.nn.functional.normalize(
            articulation.data.body_quat_w.torch[:, body_ids[0], :],
            dim=-1,
        )
        quaternion_dot = torch.abs(torch.sum(actual_orientation * expected_orientation, dim=-1)).clamp(max=1.0)
        orientation_error = 2.0 * torch.acos(quaternion_dot)
        assert torch.all(orientation_error <= 1.0e-3), orientation_error


def _bounded_random_actions(env, steps: int, seed: int) -> torch.Tensor:
    """Return deterministic, finite random commands close to the two homes."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    actions = _held_start_action(env).cpu().repeat(steps, 1, 1)
    for position_start in (0, 9):
        actions[:, :, position_start : position_start + 3] += 0.004 * (
            torch.rand((steps, env.num_envs, 3), generator=generator) - 0.5
        )
    for quaternion_start in (3, 12):
        quaternion = actions[:, :, quaternion_start : quaternion_start + 4]
        quaternion += 0.01 * (torch.rand(quaternion.shape, generator=generator) - 0.5)
        actions[:, :, quaternion_start : quaternion_start + 4] = torch.nn.functional.normalize(
            quaternion,
            dim=-1,
        )
    for jaw_start in (7, 16):
        closedness = 0.25 * torch.rand((steps, env.num_envs, 1), generator=generator)
        jaw_open = torch.tensor(DVRK_PSM_JAW_OPEN_POS).reshape(1, 1, 2)
        actions[:, :, jaw_start : jaw_start + 2] = (1.0 - closedness) * jaw_open
    actions = actions.to(env.device)
    assert actions.shape == (steps, env.num_envs, 18)
    assert actions.is_contiguous()
    assert torch.isfinite(actions).all()
    return actions


def _run_finite_action_trace(env, actions: torch.Tensor) -> None:
    """Run a fixed action-only trace and check every transition tensor."""

    with torch.inference_mode():
        for step, action in enumerate(actions):
            assert torch.isfinite(action).all(), f"non-finite action at step {step}"
            observations, rewards, terminated, truncated, _ = env.step(action)
            _assert_finite_tree(observations)
            assert rewards.shape == (env.num_envs,)
            assert torch.isfinite(rewards).all(), f"non-finite reward at step {step}"
            assert terminated.shape == (env.num_envs,)
            assert truncated.shape == (env.num_envs,)
            assert terminated.dtype == torch.bool
            assert truncated.dtype == torch.bool


@pytest.mark.isaacsim_ci
def test_one_env_donor_held_reset_is_physically_retained():
    """Require the free needle to retain bilateral donor contact from reset."""

    with _task_env(num_envs=1) as env:
        env.reset(seed=SEED)
        initial_needle_position = env.scene["needle"].data.root_pos_w.torch.clone()
        initial_needle_orientation = env.scene["needle"].data.root_quat_w.torch.clone()
        action = _held_start_action(env)
        for _ in range(64):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated.any()
            assert not truncated.any()

        loads, _, raw_forces_w = jaw_needle_contact_measurements(env)
        machine = get_handoff_phase_machine(env, DVRK_HANDOFF_PHASE_CFG)
        assert torch.all(loads[:, :2] >= DVRK_HANDOFF_PHASE_CFG.engage_force_n), loads
        assert torch.equal(machine.phase, torch.full_like(machine.phase, int(HandoffPhase.DONOR_HOLD)))
        final_needle_position = env.scene["needle"].data.root_pos_w.torch
        position_drift = final_needle_position - initial_needle_position
        # Gravity-on seating is expected for a free rigid body between driven
        # jaws. Bilateral load and the measured DONOR_HOLD phase prove
        # retention; this sub-millimetre bound catches material slip or loss.
        assert torch.max(torch.abs(position_drift)) <= 5.0e-4, {
            "initial_needle_position_w": initial_needle_position.detach().cpu().tolist(),
            "final_needle_position_w": final_needle_position.detach().cpu().tolist(),
            "initial_needle_orientation_xyzw": initial_needle_orientation.detach().cpu().tolist(),
            "final_needle_orientation_xyzw": env.scene["needle"].data.root_quat_w.torch.detach().cpu().tolist(),
            "final_needle_velocity_w": torch.cat(
                (
                    env.scene["needle"].data.root_lin_vel_w.torch,
                    env.scene["needle"].data.root_ang_vel_w.torch,
                ),
                dim=-1,
            )
            .detach()
            .cpu()
            .tolist(),
            "position_drift_m": position_drift.detach().cpu().tolist(),
            "loads_n": loads.detach().cpu().tolist(),
            "raw_forces_w_n": raw_forces_w.detach().cpu().tolist(),
        }


def _run_native_grasp_handoff(runner, env) -> None:
    """Run the fixed native trace through measured hand-off and retained lift."""

    phase_machine = get_handoff_phase_machine(env, DVRK_HANDOFF_PHASE_CFG)
    debug_phase_transitions = bool(os.environ.get("ISAACLAB_DVRK_NEEDLE_PASS_DEBUG_PHASES"))
    previous_phase = int(phase_machine.phase.item())

    def report_phase_transition(segment: str, step: int) -> None:
        nonlocal previous_phase
        current_phase = int(phase_machine.phase.item())
        if debug_phase_transitions and current_phase != previous_phase:
            diagnostic = _handoff_diagnostics(env, phase_machine)
            print(
                "NATIVE_HANDOFF_PHASE_TRANSITION="
                + repr(
                    {
                        "segment": segment,
                        "step": step,
                        "from": previous_phase,
                        "to": current_phase,
                        "loads_n": diagnostic["loads_n"],
                        "donor_bilateral": diagnostic["donor_bilateral"],
                        "receiver_bilateral": diagnostic["receiver_bilateral"],
                        "receiver_bounds": diagnostic["receiver_bounds"],
                    }
                )
            )
        previous_phase = current_phase

    for _ in range(NATIVE_DONOR_HOLD_SETTLE_STEPS):
        _, _, terminated, truncated, _ = runner.step(_held_start_action(env))
        report_phase_transition("donor_hold", _)
        assert not terminated.any()
        assert not truncated.any()
    assert int(phase_machine.phase.item()) == int(HandoffPhase.DONOR_HOLD)

    for segment in range(4):
        for step in range(NATIVE_RECEIVER_APPROACH_SEGMENT_STEPS):
            _, _, terminated, truncated, _ = runner.step(
                _native_receiver_staged_approach_action(
                    env, segment=segment, fraction=(step + 1) / NATIVE_RECEIVER_APPROACH_SEGMENT_STEPS
                )
            )
            report_phase_transition(f"approach_{segment}", step)
            assert not terminated.any(), _handoff_diagnostics(env, phase_machine)
            assert not truncated.any(), _handoff_diagnostics(env, phase_machine)
            assert int(phase_machine.phase.item()) == int(HandoffPhase.DONOR_HOLD), _handoff_diagnostics(
                env, phase_machine
            )
    for _ in range(NATIVE_RECEIVER_CLOSE_SETTLE_STEPS):
        _, _, terminated, truncated, _ = runner.step(
            _native_receiver_handoff_action(env, approach_fraction=1.0, receiver_jaw=DVRK_PSM_JAW_CLOSED_POS)
        )
        report_phase_transition("receiver_close", _)
        assert not terminated.any(), _handoff_diagnostics(env, phase_machine)
        assert not truncated.any(), _handoff_diagnostics(env, phase_machine)
    assert int(phase_machine.phase.item()) == int(HandoffPhase.CO_HOLD), _handoff_diagnostics(env, phase_machine)

    # The public donor command opens only after the measured receiver co-hold.
    for step in range(NATIVE_DONOR_RELEASE_SETTLE_STEPS):
        _, _, terminated, truncated, _ = runner.step(
            _native_receiver_handoff_action(
                env,
                approach_fraction=1.0,
                donor_jaw=_native_donor_release_jaw(step),
                receiver_jaw=DVRK_PSM_JAW_CLOSED_POS,
            )
        )
        report_phase_transition("donor_release", step)
        assert not terminated.any(), _handoff_diagnostics(env, phase_machine)
        assert not truncated.any(), _handoff_diagnostics(env, phase_machine)
    if int(phase_machine.phase.item()) != int(HandoffPhase.RECEIVER_ONLY_HOLD):
        print("NATIVE_HANDOFF_RELEASE_FAILURE=" + repr(_handoff_diagnostics(env, phase_machine)))
    assert int(phase_machine.phase.item()) == int(HandoffPhase.RECEIVER_ONLY_HOLD), _handoff_diagnostics(
        env, phase_machine
    )

    for step in range(NATIVE_RECEIVER_LIFT_STEPS):
        before_step = _handoff_diagnostics(env, phase_machine)
        # A non-terminal lift frame must remain receiver-only.  This catches a
        # donor re-grasp immediately rather than accepting a later recovery.
        assert int(phase_machine.phase.item()) == int(HandoffPhase.RECEIVER_ONLY_HOLD), before_step
        loads_before_step, _, _ = jaw_needle_contact_measurements(env)
        assert torch.all(loads_before_step[:, :2] < DVRK_HANDOFF_PHASE_CFG.disengage_force_n), before_step
        assert torch.all(loads_before_step[:, 2:] >= DVRK_HANDOFF_PHASE_CFG.engage_force_n), before_step
        _, _, terminated, truncated, _ = runner.step(
            _native_receiver_handoff_action(
                env,
                approach_fraction=1.0,
                donor_jaw=DVRK_PSM_JAW_OPEN_POS,
                receiver_jaw=DVRK_PSM_JAW_CLOSED_POS,
                receiver_lift_offset_m=_native_receiver_lift_offset(step),
            )
        )
        report_phase_transition("receiver_lift", step)
        assert not truncated.any(), _handoff_diagnostics(env, phase_machine)
        if terminated.any():
            success = env.termination_manager.get_term("success")
            dropped = env.termination_manager.get_term("needle_dropped_or_out_of_bounds")
            if not torch.all(success) or torch.any(dropped):
                print(
                    "NATIVE_HANDOFF_LIFT_FAILURE="
                    + repr(
                        {
                            "step": step,
                            "before_step": before_step,
                            "success": success.detach().cpu().tolist(),
                            "dropped": dropped.detach().cpu().tolist(),
                            "after_reset": _handoff_diagnostics(env, phase_machine),
                        }
                    )
                )
            assert torch.all(success), _handoff_diagnostics(env, phase_machine)
            assert not torch.any(dropped), _handoff_diagnostics(env, phase_machine)
            # ManagerBasedRLEnv resets a terminal environment before returning
            # from ``step``.  ``success`` is therefore the authoritative
            # pre-reset retained-lift result; inspecting phase afterwards would
            # incorrectly observe the next episode's INITIAL state.
            return
        assert int(phase_machine.phase.item()) == int(HandoffPhase.RECEIVER_ONLY_HOLD), _handoff_diagnostics(
            env, phase_machine
        )
    raise AssertionError("native receiver never completed the measured retained-lift termination")


def _assert_native_handoff_audit(audit: _DirectStateWriteAudit) -> None:
    """Require only the explicit and terminal-reset writes, plus drives."""

    # The first window is the explicit seeded reset.  The second is Isaac
    # Lab's automatic reset after the measured success termination.  Both must
    # exactly match the documented reset whitelist; no transfer-step state
    # write is permitted.
    assert audit.reset_windows == [
        _EXPECTED_RESET_STATE_WRITE_SEQUENCE,
        _EXPECTED_RESET_STATE_WRITE_SEQUENCE,
    ]
    assert audit.violations == []
    assert audit.usd_mutations == []


@pytest.mark.isaacsim_ci
def test_one_env_native_grasp_generator_handoff_is_physically_qualified():
    """Qualify the full native transfer on CUDA PhysX without state mutation."""

    with _task_env(num_envs=1) as env:
        audit = _DirectStateWriteAudit(env)
        audit.install()
        env.reset(seed=SEED)
        _assert_live_needle_topology(env)
        _run_native_grasp_handoff(env, env)
        _assert_live_needle_topology(env)
        _assert_native_handoff_audit(audit)


@pytest.mark.isaacsim_ci
def test_one_env_native_grasp_generator_handoff_video():
    """Record the same full CUDA qualification trace when recording is requested."""

    video_dir = _verification_video_dir()
    if video_dir is None:
        pytest.skip("set ISAACLAB_DVRK_NEEDLE_PASS_VIDEO_DIR to record the qualified handoff")
    with _task_env(
        num_envs=1,
        video_dir=video_dir,
        video_length=NATIVE_HANDOFF_TRACE_STEPS,
        video_prefix="dvrk-needle-pass-native-handoff",
    ) as runner:
        env = runner.unwrapped
        audit = _DirectStateWriteAudit(env)
        audit.install()
        runner.reset(seed=SEED)
        _assert_live_needle_topology(env)
        _run_native_grasp_handoff(runner, env)
        _assert_live_needle_topology(env)
        _assert_native_handoff_audit(audit)

    videos = sorted(video_dir.glob("dvrk-needle-pass-native-handoff-episode-0*.mp4"))
    assert videos, f"RecordVideo did not write the qualified handoff video to {video_dir}"


@pytest.mark.isaacsim_ci
def test_native_receiver_candidates_complete_a_cuda_guarded_transfer():
    """Screen exact native rows through the guarded CUDA transfer and lift.

    The opt-in probe accepts exact rows from the hash-pinned generator output.
    It never creates pose neighbours, alters the free needle, or bypasses the
    donor release guard. Qualification requires the donor to be released, the
    recipient to retain bilateral force, and the free needle to lift 15 mm.
    """

    candidate_indices, candidate_poses_cpu = _native_probe_candidate_poses()
    with _task_env(num_envs=len(candidate_indices)) as env:
        env.reset(seed=SEED)
        initial_needle_z_w = env.scene["needle"].data.root_pos_w.torch[:, 2].clone()
        for _ in range(NATIVE_DONOR_HOLD_SETTLE_STEPS):
            _, _, terminated, truncated, _ = env.step(_held_start_action(env))
            assert not terminated.any()
            assert not truncated.any()

        candidate_poses = candidate_poses_cpu.to(env.device)
        receiver_pos_w, receiver_quat_xyzw = _native_receiver_candidate_targets(env, candidate_poses)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for segment in range(4):
            for step in range(NATIVE_RECEIVER_APPROACH_SEGMENT_STEPS):
                _, _, terminated, truncated, _ = env.step(
                    _native_receiver_candidate_approach_action(
                        env,
                        receiver_pos_w,
                        receiver_quat_xyzw,
                        segment=segment,
                        fraction=(step + 1) / NATIVE_RECEIVER_APPROACH_SEGMENT_STEPS,
                    )
                )
                alive &= ~(terminated | truncated)
        for _ in range(NATIVE_RECEIVER_CLOSE_SETTLE_STEPS):
            _, _, terminated, truncated, _ = env.step(
                _native_receiver_candidate_handoff_action(
                    env,
                    receiver_pos_w,
                    receiver_quat_xyzw,
                    donor_jaw=DONOR_GRASP_JAW_POS,
                    receiver_jaw=DVRK_PSM_JAW_CLOSED_POS,
                )
            )
            alive &= ~(terminated | truncated)

        phase_machine = get_handoff_phase_machine(env, DVRK_HANDOFF_PHASE_CFG)
        co_hold = phase_machine.phase == int(HandoffPhase.CO_HOLD)
        donor_released_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        donor_released_continuously = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        recipient_grasped_at_release = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        receiver_retained_continuously = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        release_receiver_loads = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        release_receiver_normal_dot = torch.full((env.num_envs,), torch.nan, device=env.device)
        for _ in range(NATIVE_DONOR_RELEASE_SETTLE_STEPS):
            active = alive.clone()
            _, _, terminated, truncated, _ = env.step(
                _native_receiver_candidate_handoff_action(
                    env,
                    receiver_pos_w,
                    receiver_quat_xyzw,
                    donor_jaw=DVRK_PSM_JAW_OPEN_POS,
                    receiver_jaw=DVRK_PSM_JAW_CLOSED_POS,
                )
            )
            valid_sample = active & ~(terminated | truncated)
            loads_during_release, normals_during_release, _ = jaw_needle_contact_measurements(env)
            donor_is_released = torch.all(
                loads_during_release[:, :2] < DVRK_HANDOFF_PHASE_CFG.disengage_force_n, dim=-1
            )
            receiver_is_bilateral = phase_machine._bilateral_contact(
                loads_during_release[:, 2:],
                normals_during_release[:, 2:],
                phase_machine._receiver_engaged,
            )
            receiver_unit_normals = torch.nn.functional.normalize(normals_during_release[:, 2:], dim=-1, eps=1.0e-12)
            receiver_normal_dot = torch.sum(receiver_unit_normals[:, 0] * receiver_unit_normals[:, 1], dim=-1)
            first_release = valid_sample & donor_is_released & ~donor_released_once
            release_receiver_loads[first_release] = loads_during_release[first_release, 2:]
            release_receiver_normal_dot[first_release] = receiver_normal_dot[first_release]
            recipient_grasped_at_release &= torch.where(first_release, receiver_is_bilateral, True)
            donor_released_continuously &= torch.where(valid_sample & donor_released_once, donor_is_released, True)
            receiver_retained_continuously &= torch.where(
                valid_sample & (donor_released_once | first_release), receiver_is_bilateral, True
            )
            donor_released_once |= valid_sample & donor_is_released
            alive &= valid_sample
        receiver = env.scene["right_psm"]
        body_ids, body_names = receiver.find_bodies(DVRK_PSM_TOOL_TIP_BODY_NAME)
        assert len(body_ids) == 1, body_names
        post_release_relative_pos, post_release_relative_quat = math_utils.subtract_frame_transforms(
            receiver.data.body_pos_w.torch[:, body_ids[0]],
            receiver.data.body_quat_w.torch[:, body_ids[0]],
            env.scene["needle"].data.root_pos_w.torch,
            env.scene["needle"].data.root_quat_w.torch,
        )
        post_release_velocity_w = torch.cat(
            (env.scene["needle"].data.root_lin_vel_w.torch, env.scene["needle"].data.root_ang_vel_w.torch), dim=-1
        ).clone()
        terminal_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        physical_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        retained_lift_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        required_retained_lift_steps = phase_machine._required_steps(DVRK_HANDOFF_PHASE_CFG.retained_lift_dwell_s)
        max_lift_delta_z_m = torch.full((env.num_envs,), -torch.inf, device=env.device)
        for step in range(NATIVE_RECEIVER_LIFT_STEPS):
            active = alive & ~terminal_success & ~physical_success
            loads_before_lift, normals_before_lift, _ = jaw_needle_contact_measurements(env)
            donor_is_released_before_lift = torch.all(
                loads_before_lift[:, :2] < DVRK_HANDOFF_PHASE_CFG.disengage_force_n, dim=-1
            )
            receiver_is_bilateral_before_lift = phase_machine._bilateral_contact(
                loads_before_lift[:, 2:],
                normals_before_lift[:, 2:],
                phase_machine._receiver_engaged,
            )
            tracking = active & donor_released_once
            donor_released_continuously &= torch.where(tracking, donor_is_released_before_lift, True)
            receiver_retained_continuously &= torch.where(tracking, receiver_is_bilateral_before_lift, True)
            lift_delta_z_m = env.scene["needle"].data.root_pos_w.torch[:, 2] - initial_needle_z_w
            max_lift_delta_z_m = torch.where(
                active, torch.maximum(max_lift_delta_z_m, lift_delta_z_m), max_lift_delta_z_m
            )
            retained_lift_condition = (
                tracking
                & donor_is_released_before_lift
                & receiver_is_bilateral_before_lift
                & (lift_delta_z_m >= DVRK_HANDOFF_PHASE_CFG.required_lift_delta_z_m)
            )
            updated_retained_lift_counter = torch.where(
                retained_lift_condition, retained_lift_counter + 1, torch.zeros_like(retained_lift_counter)
            )
            retained_lift_counter = torch.where(active, updated_retained_lift_counter, retained_lift_counter)
            physical_success |= (
                (retained_lift_counter >= required_retained_lift_steps)
                & donor_released_continuously
                & receiver_retained_continuously
            )
            active &= ~physical_success
            _, _, terminated, truncated, _ = env.step(
                _native_receiver_candidate_handoff_action(
                    env,
                    receiver_pos_w,
                    receiver_quat_xyzw,
                    donor_jaw=DVRK_PSM_JAW_OPEN_POS,
                    receiver_jaw=DVRK_PSM_JAW_CLOSED_POS,
                    receiver_lift_offset_m=_native_receiver_lift_offset(step),
                )
            )
            succeeded = env.termination_manager.get_term("success")
            terminal_success |= active & terminated & succeeded
            physical_success |= active & terminated & succeeded
            alive &= ~(active & (terminated | truncated))
            loads_during_lift, normals_during_lift, _ = jaw_needle_contact_measurements(env)
            donor_is_released = torch.all(loads_during_lift[:, :2] < DVRK_HANDOFF_PHASE_CFG.disengage_force_n, dim=-1)
            receiver_is_bilateral = phase_machine._bilateral_contact(
                loads_during_lift[:, 2:],
                normals_during_lift[:, 2:],
                phase_machine._receiver_engaged,
            )
            valid_post_step = active & alive & ~terminal_success & ~physical_success
            donor_released_continuously &= torch.where(valid_post_step & donor_released_once, donor_is_released, True)
            receiver_retained_continuously &= torch.where(
                valid_post_step & donor_released_once, receiver_is_bilateral, True
            )

        loads, _, raw_forces_w = jaw_needle_contact_measurements(env)
        donor_released = torch.all(loads[:, :2] < DVRK_HANDOFF_PHASE_CFG.disengage_force_n, dim=-1)
        lifted = env.scene["needle"].data.root_pos_w.torch[:, 2] - initial_needle_z_w >= 0.015
        qualified = (
            physical_success
            & co_hold
            & donor_released_once
            & donor_released_continuously
            & recipient_grasped_at_release
            & receiver_retained_continuously
        )
        actual_pos_w = receiver.data.body_pos_w.torch[:, body_ids[0]]
        actual_quat_w = receiver.data.body_quat_w.torch[:, body_ids[0]]
        jaw_body_ids, jaw_body_names = receiver.find_bodies(["psm_tool_gripper1_link", "psm_tool_gripper2_link"])
        assert len(jaw_body_ids) == 2, jaw_body_names
        receiver_jaw_body_pos_w = receiver.data.body_pos_w.torch[:, jaw_body_ids]
        orientation_dot = torch.abs(torch.sum(actual_quat_w * receiver_quat_xyzw, dim=-1)).clamp(max=1.0)
        receiver_jaw_sensor_pos_w = torch.stack(
            [env.scene.sensors[name].data.pos_w.torch[:, 0, :] for name in JAW_CONTACT_SENSOR_NAMES[2:]], dim=1
        )
        if env.num_envs == 1:
            live_channel_w = _receiver_collision_channel_world(env, 0)
            tool_channel_w = _pose_matrix(
                actual_pos_w[0].detach().cpu().numpy(), actual_quat_w[0].detach().cpu().numpy()
            ) @ _pose_matrix(
                np.asarray(DVRK_JAW_CHANNEL_T_T_C_POS_M, dtype=np.float64),
                np.asarray(DVRK_JAW_CHANNEL_T_T_C_ROT_XYZW, dtype=np.float64),
            )
            expected_channel_w = _pose_matrix(
                env.scene["needle"].data.root_pos_w.torch[0].detach().cpu().numpy(),
                env.scene["needle"].data.root_quat_w.torch[0].detach().cpu().numpy(),
            ) @ _pose_matrix(
                candidate_poses[0, :3].detach().cpu().numpy(), candidate_poses[0, 3:].detach().cpu().numpy()
            )
            print(
                "NATIVE_RECEIVER_LIVE_CHANNELS="
                + repr(
                    {
                        "collision_midpoint_w": live_channel_w.round(9).tolist(),
                        "tool_calibrated_channel_w": tool_channel_w[:3, 3].round(9).tolist(),
                        "native_expected_channel_w": expected_channel_w[:3, 3].round(9).tolist(),
                    }
                )
            )
        report = [
            {
                "candidate": candidate_indices[index],
                "alive": bool(alive[index].item()),
                "co_hold": bool(co_hold[index].item()),
                "success_termination": bool(terminal_success[index].item()),
                "physical_success": bool(physical_success[index].item()),
                "post_trace_donor_released": bool(donor_released[index].item()),
                "donor_released_continuously": bool(donor_released_continuously[index].item()),
                "recipient_grasped_at_release": bool(recipient_grasped_at_release[index].item()),
                "receiver_retained_continuously": bool(receiver_retained_continuously[index].item()),
                "release_receiver_loads_n": release_receiver_loads[index].detach().cpu().tolist(),
                "release_receiver_normal_dot": float(release_receiver_normal_dot[index].item()),
                "max_lift_delta_z_m": float(max_lift_delta_z_m[index].item()),
                "retained_lift_steps": int(retained_lift_counter[index].item()),
                "post_trace_lifted": bool(lifted[index].item()),
                "post_trace_receiver_loads_n": loads[index, 2:].detach().cpu().tolist(),
                "post_trace_receiver_sensor_forces_w_n": raw_forces_w[index, 2:].detach().cpu().tolist(),
                "receiver_target_pos_w": receiver_pos_w[index].detach().cpu().tolist(),
                "receiver_target_quat_xyzw": receiver_quat_xyzw[index].detach().cpu().tolist(),
                "tool_position_error_m": float(
                    torch.linalg.vector_norm(actual_pos_w[index] - receiver_pos_w[index]).item()
                ),
                "tool_orientation_error_rad": float((2.0 * torch.acos(orientation_dot[index])).item()),
                "receiver_jaws_rad": receiver.data.joint_pos.torch[index, -2:].detach().cpu().tolist(),
                "post_release_relative_pos_m": post_release_relative_pos[index].detach().cpu().tolist(),
                "post_release_relative_quat_xyzw": post_release_relative_quat[index].detach().cpu().tolist(),
                "post_release_velocity_w": post_release_velocity_w[index].detach().cpu().tolist(),
                "receiver_jaw_sensor_pos_w": receiver_jaw_sensor_pos_w[index].detach().cpu().tolist(),
                "receiver_jaw_body_pos_w": receiver_jaw_body_pos_w[index].detach().cpu().tolist(),
                "qualified": bool(qualified[index].item()),
            }
            for index in range(env.num_envs)
        ]
        print(f"NATIVE_RECEIVER_CUDA_PROBE={report!r}")
        assert qualified.any(), report


@pytest.mark.isaacsim_ci
def test_one_env_runtime_contracts_and_100_random_actions():
    """Audit the renderer-free CUDA runtime path with bounded random actions."""

    with _task_env(num_envs=1) as runner:
        env = runner.unwrapped
        audit = _DirectStateWriteAudit(env)
        audit.install()

        first_observations, _ = runner.reset(seed=SEED)
        first_snapshot = _reset_snapshot(env, first_observations)
        second_observations, _ = runner.reset(seed=SEED)
        second_snapshot = _reset_snapshot(env, second_observations)
        _assert_same_tree(second_snapshot, first_snapshot)

        expected_reset_writes = {
            (asset_name, method_name)
            for layer, asset_name, method_name in _EXPECTED_RESET_STATE_WRITE_SEQUENCE
            if layer == "asset"
        }
        expected_physx_reset_writes = {
            (asset_name, method_name)
            for layer, asset_name, method_name in _EXPECTED_RESET_STATE_WRITE_SEQUENCE
            if layer == "physx"
        }
        assert audit.reset_windows == [
            _EXPECTED_RESET_STATE_WRITE_SEQUENCE,
            _EXPECTED_RESET_STATE_WRITE_SEQUENCE,
        ]
        assert set(audit.asset_state_calls) == expected_reset_writes
        assert all(audit.asset_state_calls.count(write) == 2 for write in expected_reset_writes)
        assert set(audit.physx_state_calls) == expected_physx_reset_writes
        assert all(audit.physx_state_calls.count(write) == 2 for write in expected_physx_reset_writes)
        assert audit.violations == []

        _assert_live_needle_topology(env)
        _assert_live_needle_mass_and_material(env)
        _assert_contact_sensor_matrices(env)
        _assert_live_action_and_joint_order(env)
        _assert_live_psm_jaw_contract(env)
        _assert_static_suture_pad(env)
        _assert_configured_tool_homes(env)
        trace_steps = TRACE_STEPS
        _run_finite_action_trace(runner, _bounded_random_actions(env, trace_steps, SEED + 1))
        expected_control_setters = {
            (asset_name, method_name)
            for asset_name in ("left_psm", "right_psm")
            for method_name in _ARTICULATION_PHYSX_CONTROL_SETTERS
        }
        assert set(audit.physx_control_calls) == expected_control_setters
        assert all(audit.physx_control_calls.count(call) >= trace_steps for call in expected_control_setters)
        assert audit.violations == []
        assert audit.usd_mutations == []
        _assert_live_needle_topology(env)


@pytest.mark.isaacsim_ci
def test_32_env_100_random_actions_are_finite():
    """Exercise the batched absolute-world action ABI without asserting task success."""

    with _task_env(num_envs=32) as env:
        observations, _ = env.reset(seed=SEED)
        _assert_finite_tree(observations)
        _assert_contact_sensor_matrices(env)
        _assert_live_action_and_joint_order(env)
        _run_finite_action_trace(env, _bounded_random_actions(env, TRACE_STEPS, SEED + 32))
