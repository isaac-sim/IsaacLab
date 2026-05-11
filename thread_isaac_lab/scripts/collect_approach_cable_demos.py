#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect ApproachCable demonstrations for DAPG — multi-episode, cable-adaptive.

Executes the ApproachCable scripted trajectory (approach → descend → hold) in
Newton VBD, recording (obs, action) pairs matching NewtonApproachCableEnv format.
Fingers remain OPEN throughout — close/lift belong to Grip skill (env :1554-1562).

Success criterion matches env: clamp_pos_ok ∧ clamp_ori_ok sustained K_GRASP
RL steps (T_DIST_APPROACH=12mm, T_ALIGN=10°, K_GRASP=5).

Usage:
    source ~/env_isaaclab6/bin/activate
    python thread_isaac_lab/scripts/collect_approach_cable_demos.py \
        --num-episodes 20 --cable-noise-y 0.005 \
        --output thread_isaac_lab/data/bc_demos/approach_cable_demos_v6.npz \
        --device cuda:0
"""

import argparse
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import newton
import numpy as np
import warp as wp
from newton.ik import IKObjectiveJointLimit, IKObjectivePosition, IKObjectiveRotation, IKSolver
from newton.solvers import SolverVBD

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "configs"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "envs"))

from cable_orientation_utils import compute_cable_tangent, compute_hand_quat_for_cable
from task_config import (
    APPROACH_Z,
    CABLE_RADIUS,
    CABLE_SEG_LEN,
    CABLE_SEGMENTS,
    CLIP1_X,
    CLIP1_Y,
    CLIP1_Z,
    CLIP_BASE_HEIGHT,
    EE_TO_FINGERTIP,
    FINGER_OPEN_POS,
    GRASP_X,
    GRASP_Z,
    GRIP_HALF_SPAN,
    K_GRASP,
    NJMAX,
    SIM_SUBSTEPS,
    T_ALIGN,
    T_DIST,
    T_DIST_APPROACH,
    TABLE_HEIGHT,
    WIDE_LEFT_Y,
    WIDE_RIGHT_Y,
)
from test_newton_clip_routing import (
    EE_BODY_OFFSET,
    FRANKA_NUM_JOINTS,
    GRAVITY,
    add_cable_rod,
    add_kinematic_arm,
    build_fk_model,
    update_kinematic_bodies,
)

# v5 obs utilities (from env)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "envs"))
from newton_approach_cable_env import (
    _axis_angle_to_quat_xyzw,
    _compute_clamp_pos,
    _compute_ori_error_axis_angle,
    _normalize_quat_w_positive,
    _quat_distance,
    _quat_multiply_xyzw,
    _quat_rotate_vec,
    _temporal_quat_consistency,
)


def _quat_to_axis_angle(quat_xyzw):
    """Convert quaternion [qx, qy, qz, qw] to axis-angle rotation vector [3].

    Returns [3] rotation vector (direction = axis, magnitude = angle in rad).
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    # Ensure w > 0 for minimal angle representation
    if q[3] < 0:
        q = -q
    xyz = q[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / sin_half
    angle = 2.0 * math.atan2(sin_half, float(q[3]))
    return (axis * angle).astype(np.float32)


# Physics constants (match env)
DT = 1.0 / 480.0
VBD_ITERATIONS = 20
IK_ITERATIONS = 100
IK_STEP_SIZE = 1.0
ROBOT_BODY_COUNT = 2 * FRANKA_NUM_JOINTS  # 18

# v5 action scaling (match NewtonApproachCableEnv exactly)
POS_ACTION_SCALE = 0.015  # 15mm — must match env POS_ACTION_SCALE
ROT_ACTION_SCALE = 0.05
FINE_THRESHOLD = 0.050  # 50mm: adaptive scale kicks in below this
MIN_POS_SCALE = 0.0005  # 0.5mm: minimum adaptive scale

# Env-matched constants (SSOT: NewtonApproachCableEnv class attributes)
PHYSICS_STEPS_PER_RL = 10  # env :332
GRIP_SEG_WINDOW = 5  # env :336 — ±5 segment tolerance

# Clip C1 pose (constant for ApproachCable obs)
CLIP1_POS = np.array([CLIP1_X, CLIP1_Y, CLIP1_Z], dtype=np.float32)
CLIP1_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identity

# IK rotation (hand pointing down).
# Warp quats use the xyzw storage convention (see warp/_src/types.py
# "xyzw"), and Newton IK reads each target as wp.quat(vec4[0..3]).
# Pass quats here as (x, y, z, w) to stay consistent with the rest of
# this script (_quat_rotate_vec, _quat_distance, compute_hand_quat_for_cable).
# default_rot below therefore encodes (x=cos(pi/8), y=sin(pi/8), 0, 0),
# matching BASE_HAND_DOWN_QUAT in cable_orientation_utils.py.
_cos_pi8 = math.cos(math.pi / 8)
_sin_pi8 = math.sin(math.pi / 8)

# Grasp trajectory parameters
STEPS_PER_CM = 20  # differs from task_config (demo-specific: RL action rate)
MAX_MOVE_STEPS = 1600  # differs from task_config (demo-specific: safety limit)
FINGER_INTERP_STEPS = 240

# P1.5 precision refine: iterative proportional IK after descend. Bridges the
# gap between env contract threshold T_DIST_APPROACH=12mm (AC success) and
# T_DIST=2mm (Clamp success) so downstream BC demos include the same
# fine-alignment skill the Clamp env expects. Each rl_step = one 12D action at
# the DAPG record boundary; target is regenerated every step from the current
# clamp-to-cable error (proportional control, clipped to PRECISION_SCALE).
PRECISION_MAX_STEPS = 20  # RL steps budget (each = PHYSICS_STEPS_PER_RL)
PRECISION_GAIN = 0.6  # Proportional gain on clamp-to-nearest-cable error
PRECISION_SCALE = 0.003  # Per-step EE sub-target delta clamp [m] (3mm)


def solve_ik(fk_model, fk_state, target_left, target_right, device, rot_left=None, rot_right=None, rot_weight=0.5):
    """Solve IK for both arms with cable-adaptive rotation.

    Args:
        rot_left: wp.vec4 rotation target for left EE (x,y,z,w). None = hand-down default.
        rot_right: wp.vec4 rotation target for right EE (x,y,z,w). None = hand-down default.
        rot_weight: IKObjectiveRotation weight. 0.5 (default) gives the
            legacy approach/descend behavior. Precision phase passes
            rot_weight=1.0 so the back-out target_ee = target_clamp −
            R(target_quat)·[0,0,EE_TO_FINGERTIP] cancels the ΔR·offset
            drift; residual δθ directly becomes sin(δθ)·EE_TO_FINGERTIP
            at the fingertip, which needs to stay below T_DIST=2mm.
    """
    left_ee = EE_BODY_OFFSET
    right_ee = FRANKA_NUM_JOINTS + EE_BODY_OFFSET
    default_rot = wp.vec4(_cos_pi8, _sin_pi8, 0.0, 0.0)
    if rot_left is None:
        rot_left = default_rot
    if rot_right is None:
        rot_right = default_rot

    objectives = [
        IKObjectivePosition(
            link_index=left_ee,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.array([target_left], dtype=wp.vec3, device=device),
            weight=1.0,
        ),
        IKObjectivePosition(
            link_index=right_ee,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.array([target_right], dtype=wp.vec3, device=device),
            weight=1.0,
        ),
        IKObjectiveRotation(
            link_index=left_ee,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([rot_left], dtype=wp.vec4, device=device),
            weight=rot_weight,
        ),
        IKObjectiveRotation(
            link_index=right_ee,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([rot_right], dtype=wp.vec4, device=device),
            weight=rot_weight,
        ),
        IKObjectiveJointLimit(
            joint_limit_lower=fk_model.joint_limit_lower, joint_limit_upper=fk_model.joint_limit_upper, weight=10.0
        ),
    ]
    ik_solver = IKSolver(fk_model, n_problems=1, objectives=objectives)
    fk_jq = fk_state.joint_q.numpy()
    jq_in = wp.array(fk_jq.reshape(1, -1), dtype=float, device=device)
    jq_out = wp.zeros((1, fk_model.joint_coord_count), dtype=float, device=device)
    ik_solver.step(jq_in, jq_out, iterations=IK_ITERATIONS, step_size=IK_STEP_SIZE)
    return jq_out.numpy()[0]


def compute_obs(
    state,
    fk_jq,
    cable_bodies,
    target_seg_indices_r,
    target_seg_indices_l=None,
    *,
    prev_clamp_r_quat,
    prev_clamp_l_quat,
    prev_seg_r_quat,
    prev_seg_l_quat,
):
    """Compute 42D observation matching NewtonApproachCableEnv v5 format (biarm).

    Mirrors env._compute_obs_batch (newton_approach_cable_env.py L1339-1368):
    for each of the 4 quats, apply normalize -> temporal_consistency -> rebind
    BEFORE any downstream error/alias use. Any change here must be mirrored
    in the env.

    Per-arm independent nearest cable point (matching env per-arm target_seg fix).

    Args:
        state: Newton state with body_q.
        fk_jq: [joint_coord_count] FK joint positions.
        cable_bodies: list of cable body indices.
        target_seg_indices_r: list of target cable segment indices for right arm (+-1 window).
        target_seg_indices_l: list of target cable segment indices for left arm (+-1 window).
            If None, falls back to target_seg_indices_r (backward compat).
        prev_clamp_r_quat: Previous post-consistency clamp quat (R). Required
            keyword-only. Caller threads across compute_obs calls within the
            same episode; init to identity [0,0,0,1] on episode start.
        prev_clamp_l_quat: Previous post-consistency clamp quat (L). Required.
        prev_seg_r_quat: Previous post-consistency grasp target quat (R). Required.
        prev_seg_l_quat: Previous post-consistency grasp target quat (L). Required.
            grasp_target_quat_l does not appear directly in obs but feeds
            ori_error_aa_l (obs[36:39]); threading prev keeps sign-stability.

    Returns:
        Tuple of (obs, clamp_r_quat, clamp_l_quat, seg_quat, grasp_target_quat_l)
        where trailing 4 arrays are post-temporal copies for the caller to
        thread as prev_* on the next call. Defensive .copy() matches env
        L1341/1344/1359/1368.
    """
    if target_seg_indices_l is None:
        target_seg_indices_l = target_seg_indices_r

    bq = state.body_q.numpy()

    # Right EE body
    ee_r_pos = bq[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3]
    ee_r_quat = bq[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7]
    # uses raw quat; _compute_clamp_pos is rotation-sign-invariant, matches env L1335
    clamp_r_pos = _compute_clamp_pos(ee_r_pos, ee_r_quat)
    # Ordering: normalize -> temporal -> rebind. Mirrors env L1339-1341.
    clamp_r_quat = _normalize_quat_w_positive(ee_r_quat)
    clamp_r_quat = _temporal_quat_consistency(clamp_r_quat, prev_clamp_r_quat)

    # Left EE body
    ee_l_pos = bq[EE_BODY_OFFSET][:3]
    ee_l_quat = bq[EE_BODY_OFFSET][3:7]
    clamp_l_pos = _compute_clamp_pos(ee_l_pos, ee_l_quat)
    clamp_l_quat = _normalize_quat_w_positive(ee_l_quat)
    clamp_l_quat = _temporal_quat_consistency(clamp_l_quat, prev_clamp_l_quat)

    # Finger openings
    r_finger_opening = fk_jq[FRANKA_NUM_JOINTS + 7] + fk_jq[FRANKA_NUM_JOINTS + 8]
    l_finger_opening = fk_jq[7] + fk_jq[8]

    cable_bq = bq[cable_bodies]  # [n_cable, 7]
    cable_pos = cable_bq[:, :3]

    # Right arm: nearest cable point + target quat with temporal consistency.
    # MUST apply temporal BEFORE seg_quat alias (env L1358-1360 stores the
    # consistency-adjusted value and then aliases as seg_quat).
    seg_pos, seg_tangent, _ = _find_nearest_cable_point_demo(cable_pos, clamp_r_pos, target_seg_indices_r)
    grasp_target_quat = _normalize_quat_w_positive(compute_hand_quat_for_cable(seg_tangent))
    grasp_target_quat = _temporal_quat_consistency(grasp_target_quat, prev_seg_r_quat)
    seg_quat = grasp_target_quat  # obs[19:23] - post-temporal value

    # Left arm: independent nearest cable point + target quat
    seg_pos_l, seg_tangent_l, _ = _find_nearest_cable_point_demo(cable_pos, clamp_l_pos, target_seg_indices_l)
    grasp_target_quat_l = _normalize_quat_w_positive(compute_hand_quat_for_cable(seg_tangent_l))
    grasp_target_quat_l = _temporal_quat_consistency(grasp_target_quat_l, prev_seg_l_quat)

    # Right arm error signals - computed AFTER all 4 quats are post-temporal
    # so obs[3:7]/[11:15]/[19:23] quat channels stay sign-continuous across
    # time. ori_error_aa itself is sign-invariant via env L182-183 axis-angle
    # canonicalization, but the raw quat channels in obs require consistency.
    ori_error_aa = _compute_ori_error_axis_angle(clamp_r_quat, grasp_target_quat)
    pos_error = clamp_r_pos - seg_pos  # [3]

    # Left arm error signals (independent nearest cable point)
    ori_error_aa_l = _compute_ori_error_axis_angle(clamp_l_quat, grasp_target_quat_l)
    pos_error_l = clamp_l_pos - seg_pos_l  # [3]

    obs = np.array(
        [
            *clamp_r_pos,  # [0:3]
            *clamp_r_quat,  # [3:7]  post-temporal
            r_finger_opening,  # [7]
            *clamp_l_pos,  # [8:11]
            *clamp_l_quat,  # [11:15] post-temporal
            l_finger_opening,  # [15]
            *seg_pos,  # [16:19]
            *seg_quat,  # [19:23] post-temporal
            *CLIP1_POS,  # [23:26]
            *CLIP1_QUAT_XYZW,  # [26:30]
            *ori_error_aa,  # [30:33] uses post-temporal quats
            *pos_error,  # [33:36]
            *ori_error_aa_l,  # [36:39] uses post-temporal quats
            *pos_error_l,  # [39:42]
        ],
        dtype=np.float32,
    )

    # .copy() mirrors env L1341/1344/1359/1368 to prevent caller-held prev
    # state from aliasing compute_obs's locals.
    return (obs, clamp_r_quat.copy(), clamp_l_quat.copy(), grasp_target_quat.copy(), grasp_target_quat_l.copy())


def _find_nearest_cable_point_demo(cable_pos, hand_pos, search_indices):
    """Find nearest point on piecewise-linear cable to hand_pos.

    Mirrors env _find_nearest_cable_point (env :217-258): projects hand onto
    each edge within search_indices ±1 expansion and returns nearest projected
    point, tangent, and distance.
    """
    n_cable = len(cable_pos)
    best_dist_sq = float("inf")
    best_pos = cable_pos[search_indices[0]]
    best_tangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    idx_min = max(0, int(search_indices[0]) - 1)
    idx_max = min(n_cable - 1, int(search_indices[-1]) + 1)
    for i in range(idx_min, idx_max):
        a = cable_pos[i]
        b = cable_pos[i + 1]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-12:
            t = 0.0
        else:
            t = np.clip(np.dot(hand_pos - a, ab) / ab_len_sq, 0.0, 1.0)
        proj = a + t * ab
        d_sq = float(np.sum((hand_pos - proj) ** 2))
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_pos = proj
            norm = np.sqrt(ab_len_sq)
            best_tangent = ab / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return best_pos.copy().astype(np.float32), best_tangent.astype(np.float32), float(np.sqrt(best_dist_sq))


def _adaptive_pos_scale(clamp_pos, cable_pos, search_indices):
    """Compute distance-adaptive position action scale.

    Mirrors env :1700-1712 — uses piecewise-linear projection onto cable
    edges within search_indices (via _find_nearest_cable_point_demo) to
    compute dist. Scale shrinks linearly below FINE_THRESHOLD, clamped to
    MIN_POS_SCALE.
    """
    _, _, dist = _find_nearest_cable_point_demo(cable_pos, clamp_pos, search_indices)
    return max(MIN_POS_SCALE, POS_ACTION_SCALE * min(1.0, dist / FINE_THRESHOLD))


def compute_action(
    ee_delta_r, rot_delta_r, ee_delta_l, rot_delta_l, r_pos_scale=POS_ACTION_SCALE, l_pos_scale=POS_ACTION_SCALE
):
    """Compute 12D action matching NewtonApproachCableEnv v5 format (normalized).

    Args:
        ee_delta_r: [3] right EE position delta [m].
        rot_delta_r: [3] right EE rotation delta (axis-angle) [rad].
        ee_delta_l: [3] left EE position delta [m].
        rot_delta_l: [3] left EE rotation delta (axis-angle) [rad].
        r_pos_scale: Position action scale for right arm [m].
        l_pos_scale: Position action scale for left arm [m].
    """
    return np.array(
        [
            ee_delta_r[0] / r_pos_scale,
            ee_delta_r[1] / r_pos_scale,
            ee_delta_r[2] / r_pos_scale,
            rot_delta_r[0] / ROT_ACTION_SCALE,
            rot_delta_r[1] / ROT_ACTION_SCALE,
            rot_delta_r[2] / ROT_ACTION_SCALE,
            ee_delta_l[0] / l_pos_scale,
            ee_delta_l[1] / l_pos_scale,
            ee_delta_l[2] / l_pos_scale,
            rot_delta_l[0] / ROT_ACTION_SCALE,
            rot_delta_l[1] / ROT_ACTION_SCALE,
            rot_delta_l[2] / ROT_ACTION_SCALE,
        ],
        dtype=np.float32,
    )


def query_cable_y(state, cable_bodies, target_x):
    """Query cable center Y position after settle.

    The cable runs along Y, so all bodies have similar X. Return the
    mean Y of all cable bodies — this captures noise-induced Y shifts
    and gives the correct grip center for symmetric grasping.
    """
    bq = state.body_q.numpy()
    cable_pos = bq[cable_bodies, :3]
    return float(np.mean(cable_pos[:, 1]))


def build_physics_scene(fk_model, fk_state, device):
    """Build VBD scene: cable + kinematic arms + table (no clips needed for grasp)."""
    proto = newton.ModelBuilder()

    left_info = add_kinematic_arm(proto, fk_model, fk_state, arm_body_offset=0, label_prefix="left")
    right_info = add_kinematic_arm(proto, fk_model, fk_state, arm_body_offset=FRANKA_NUM_JOINTS, label_prefix="right")
    left_body_start, left_shape_start, left_shape_end, left_fv = left_info
    right_body_start, right_shape_start, right_shape_end, right_fv = right_info
    all_finger_visual = set(left_fv + right_fv)

    # Contact filtering
    for arm_ss, arm_se, arm_bs in [
        (left_shape_start, left_shape_end, left_body_start),
        (right_shape_start, right_shape_end, right_body_start),
    ]:
        for si in range(arm_ss, arm_se):
            local = proto.shape_body[si] - arm_bs
            if local < 7 or si in all_finger_visual:
                proto.shape_flags[si] = 1
            elif local in (7, 8):
                proto.shape_flags[si] = 0x6

    # Cable
    cable_half_len = CABLE_SEGMENTS * CABLE_SEG_LEN / 2
    cable_start = (GRASP_X, CLIP1_Y - cable_half_len, TABLE_HEIGHT + CABLE_RADIUS)
    cable_shape_start_idx = proto.shape_count
    cable_bodies, cable_joints = add_cable_rod(proto, start_pos=cable_start, direction=(0, 1, 0))
    cable_shape_end_idx = proto.shape_count

    # Cable-arm collision filter (arm 0-6)
    for cable_si in range(cable_shape_start_idx, cable_shape_end_idx):
        for arm_ss, arm_se, arm_bs in [
            (left_shape_start, left_shape_end, left_body_start),
            (right_shape_start, right_shape_end, right_body_start),
        ]:
            for arm_si in range(arm_ss, arm_se):
                local = proto.shape_body[arm_si] - arm_bs
                if local < 7:
                    proto.add_shape_collision_filter_pair(cable_si, arm_si)

    # Finger collision is now BOX primitives (no approximate_meshes needed)

    # Scene
    scene = newton.ModelBuilder(gravity=GRAVITY)
    scene.add_ground_plane()

    table_cfg = newton.ModelBuilder.ShapeConfig()
    table_cfg.ke = 500.0
    table_cfg.kd = 100.0
    table_cfg.mu = 1.0
    table_cfg.gap = 0.002
    table_xform = wp.transform((0.35, 0.0, TABLE_HEIGHT - 0.005), wp.quat_identity())
    scene.add_shape_box(body=-1, hx=0.50, hy=0.40, hz=0.005, xform=table_xform, cfg=table_cfg)

    scene.replicate(proto, world_count=1)
    scene.color()
    model = scene.finalize(device=device, requires_grad=False)

    # Zero inv_mass for robot bodies
    inv_mass = model.body_inv_mass.numpy()
    inv_inertia = model.body_inv_inertia.numpy()
    for bi in range(ROBOT_BODY_COUNT):
        inv_mass[bi] = 0.0
        inv_inertia[bi] = np.zeros(3, dtype=np.float32)
    model.body_inv_mass = wp.array(inv_mass, dtype=model.body_inv_mass.dtype, device=device)
    model.body_inv_inertia = wp.array(inv_inertia, dtype=model.body_inv_inertia.dtype, device=device)

    # Post-finalize: finger shape flags (BOX collision / MESH visual)
    model_sflags = model.shape_flags.numpy()
    model_stypes = model.shape_type.numpy()
    model_sbodies = model.shape_body.numpy()
    for si in range(len(model_stypes)):
        bi = model_sbodies[si]
        if bi < 0 or bi >= ROBOT_BODY_COUNT:
            continue
        local_l = bi
        local_r = bi - FRANKA_NUM_JOINTS
        if local_l in (7, 8) or local_r in (7, 8):
            if model_stypes[si] == 7:  # BOX = collision only
                model_sflags[si] = 0x6  # COLLIDE | BROADPHASE
            elif model_stypes[si] == 8:  # MESH = visual only
                model_sflags[si] = 0x1  # VISIBLE only
    model.shape_flags = wp.array(model_sflags, dtype=model.shape_flags.dtype, device=device)

    solver = SolverVBD(model, iterations=VBD_ITERATIONS)
    model.rigid_contact_max = NJMAX
    state = model.state()
    contacts = model.contacts()

    return model, state, solver, contacts, cable_bodies


def settle_cable(model, state, solver, contacts, fk_state, device, duration=2.0):
    """Settle cable under gravity."""
    control = model.control()
    state_buf = model.state()
    n_frames = int(duration / DT)
    for _ in range(n_frames):
        for _ in range(SIM_SUBSTEPS):
            update_kinematic_bodies(state, fk_state, ROBOT_BODY_COUNT)
            state.clear_forces()
            model.collide(state, contacts)
            solver.step(state, state_buf, control, contacts, DT / SIM_SUBSTEPS)
            state, state_buf = state_buf, state
    wp.synchronize()
    return state


def reset_state(model, state, solver, init_body_q, init_body_qd, device):
    """Reset physics state to initial configuration."""
    state.body_q.assign(init_body_q)
    state.body_qd.assign(init_body_qd)
    solver.body_q_prev.assign(init_body_q)
    wp.synchronize()
    return state


def _balanced_return(obs_list, act_list, success, final_dist_mm, precision_ok=False):
    """Drop trailing orphan obs so len(obs)==len(act) at abort.
    At record boundary we append obs before act (act requires ee_r_before),
    so a NaN mid-window can leave obs ahead by 1.

    Args:
        precision_ok: Keep-gate flag for the main() caller — True accepts
            the demo, False drops it from BC training. NOT a measurement
            of precision-phase success; it is True iff either the phase
            ran and sustained T_DIST for K_GRASP RL steps, or the phase
            did not execute (pre-precision abort from legacy callers).
            Defaults to False so every caller makes the keep decision
            explicit — missing it defaults to dropping the demo rather
            than silently accepting a partial trajectory (pre-F3 the
            default was True, which fails open on abort paths).
    """
    if len(obs_list) == len(act_list) + 1:
        obs_list.pop()
    return obs_list, act_list, success, final_dist_mm, precision_ok


def execute_approach_episode(  # noqa: C901 -- pre-existing complexity; refactor deferred to a separate task
    model,
    state,
    solver,
    contacts,
    fk_model,
    fk_state,
    cable_bodies,
    grip_y_right,
    device,
    record_every=PHYSICS_STEPS_PER_RL,
    grip_y_left=None,
    target_seg_indices_r=None,
    target_seg_indices_l=None,
    right_seg_index=None,
    left_seg_index=None,
    start_from_p0=False,
    demo_substeps=4,
):
    """Execute one approach episode, recording obs/action pairs.

    Approach trajectory: approach → descend → hold (fingers OPEN).
    Matches AC env contract (env :1554-1562): AC = approach only, no close/lift.
    Success = clamp_pos_ok ∧ clamp_ori_ok sustained for K_GRASP RL steps.

    Args:
        grip_y_right: Cable-adaptive Y position for right arm grip.
        grip_y_left: Cable-adaptive Y position for left arm grip (default: WIDE_LEFT_Y).
        record_every: Record obs/action every N physics steps. Must match
            PHYSICS_STEPS_PER_RL for correct DAPG action magnitude.
        target_seg_indices_r: list of cable segment indices for right arm obs.
        target_seg_indices_l: list of cable segment indices for left arm obs.
        right_seg_index: Cable segment index nearest to right grip (for tangent).
        left_seg_index: Cable segment index nearest to left grip (for tangent).
        start_from_p0: If True, skip approach phase (start from descend).
        demo_substeps: Substeps per physics step during recording (default: 4,
            matching RL_SIM_SUBSTEPS in training env).

    Returns:
        obs_list, act_list, success (bool), final_dist_mm (float),
        precision_ok (bool). precision_ok is True when the P1.5 precision
        phase sustained T_DIST for K_GRASP RL steps, or when the phase was
        not executed at all (fallback path); False when precision ran but
        aborted or timed out before convergence.
    """
    obs_list = []
    act_list = []
    sustain_counter = 0
    ever_succeeded = False
    final_dist_mm = float("inf")
    ran_precision = False
    precision_succeeded = False

    # Per-episode prev state for obs temporal consistency (F1).
    # MUST be at function-body scope (outside the phases loop) so state
    # persists across approach -> descend -> precision -> hold within one
    # episode. Re-init to identity on every call to execute_approach_episode
    # mirrors env per-world reset at newton_approach_cable_env.py L1250-1253.
    #
    # First-call semantics: compute_obs applies _normalize_quat_w_positive
    # BEFORE _temporal_quat_consistency (newton_approach_cable_env.py L125-126
    # enforces q[3] >= 0 post-normalize). Therefore dot(first_quat_normalized,
    # identity_prev=[0,0,0,1]) = first_quat_normalized[3] >= 0, and the
    # `dot < 0` flip condition is never true on the first call -- regardless
    # of warmup rotation perturbation (main L1263-1275) or BASE_HAND_DOWN_QUAT
    # specifics. Matches env's first-call semantics after per-world reset.
    # Do NOT hoist this init to main() scope (would carry prev across
    # episodes, violating env reset semantics).
    prev_clamp_r_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    prev_clamp_l_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    prev_seg_r_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    prev_seg_l_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Adaptive positions from cable center
    if grip_y_left is None:
        grip_y_left = WIDE_LEFT_Y
    finger_open = FINGER_OPEN_POS

    # Phases: (target_r, target_l, finger_r, desc)
    # approach → descend → hold (fingers OPEN throughout)
    phases = [
        # P0: approach — both arms at approach height
        ([GRASP_X, grip_y_right, APPROACH_Z], [GRASP_X, grip_y_left, APPROACH_Z], finger_open, "approach"),
        # P1: descend — both arms to grasp height
        ([GRASP_X, grip_y_right, GRASP_Z], [GRASP_X, grip_y_left, GRASP_Z], finger_open, "descend"),
        # P1.5: precision refine — iterative proportional IK toward cable
        # surface (dynamic target, recomputed per RL step). Bridges
        # T_DIST_APPROACH=12mm → T_DIST=2mm so BC demos include the
        # fine-alignment skill that downstream Clamp env expects.
        (None, None, finger_open, "precision"),
        # P2: hold — maintain position with fingers OPEN (AC env contract)
        ([GRASP_X, grip_y_right, GRASP_Z], [GRASP_X, grip_y_left, GRASP_Z], finger_open, "hold"),
    ]

    if start_from_p0:
        phases = phases[1:]  # skip approach — arms already at P0

    control = model.control()
    state_buf = model.state()

    for phase_idx, (target_r, target_l, target_finger_r, desc) in enumerate(phases):
        # Current EE positions
        bq = state.body_q.numpy()
        pos_r = bq[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3]
        pos_l = bq[EE_BODY_OFFSET][:3]
        coord_count = fk_model.joint_coord_count

        # Per-phase window state + action closure (shared across all phase
        # types, including P1.5 precision where jq_target is dynamic).
        ee_r_before = None
        ee_l_before = None
        r_scale_before = POS_ACTION_SCALE
        l_scale_before = POS_ACTION_SCALE

        def _compute_cumulative_action(state_now, r_scale, l_scale):
            """Compute action as cumulative EE delta since last record boundary."""
            wp.synchronize()
            bq_now = state_now.body_q.numpy()
            ee_r_now = bq_now[FRANKA_NUM_JOINTS + EE_BODY_OFFSET]
            ee_l_now = bq_now[EE_BODY_OFFSET]

            delta_r_pos = ee_r_now[:3] - ee_r_before[:3]
            delta_l_pos = ee_l_now[:3] - ee_l_before[:3]

            q_r_inv = ee_r_before[3:7].copy()
            q_r_inv[:3] = -q_r_inv[:3]
            delta_r_quat = _quat_multiply_xyzw(ee_r_now[3:7], q_r_inv)
            delta_r_rot = _quat_to_axis_angle(delta_r_quat)

            q_l_inv = ee_l_before[3:7].copy()
            q_l_inv[:3] = -q_l_inv[:3]
            delta_l_quat = _quat_multiply_xyzw(ee_l_now[3:7], q_l_inv)
            delta_l_rot = _quat_to_axis_angle(delta_l_quat)

            return compute_action(
                delta_r_pos, delta_r_rot, delta_l_pos, delta_l_rot, r_pos_scale=r_scale, l_pos_scale=l_scale
            )

        if desc == "precision":
            # === P1.5: precision refine via iterative proportional IK ===
            # Each rl_step = one 12D action at the DAPG record boundary; the
            # EE sub-target is regenerated every step from the current
            # clamp-to-cable error. Early-exit when max(dist_r, dist_l) <
            # T_DIST sustained for K_GRASP RL steps.
            ran_precision = True
            precision_sustain = 0
            # P3: reset AC-level sustain_counter on precision entry so any
            # stale count carried over from descend (which uses a looser
            # T_DIST_APPROACH=12mm criterion on a different target quat
            # source — see L1 back-out alignment) cannot spuriously satisfy
            # ever_succeeded before precision has a fair chance to run.
            sustain_counter = 0
            demo_dt = DT / demo_substeps
            rl_step = 0
            # C1 diagnostic: max observed IK rotation residual across the
            # precision phase (radians). Logged in the [precision] Done
            # summary so the back-out drift can be audited empirically.
            ik_res_max_observed = 0.0
            # F2 diagnostic: max observed IK position residual (meters,
            # EE-space against target_r_sub / target_l_sub). rot_weight=1.0
            # lets IK allocate residual to position under near-singular
            # configurations, so rot alone understates fingertip drift.
            ik_pos_res_max_observed = 0.0
            for rl_step in range(PRECISION_MAX_STEPS):
                # (1) Action for PREVIOUS window (uses OLD ee_r_before / scales)
                if ee_r_before is not None:
                    act_list.append(_compute_cumulative_action(state, r_scale_before, l_scale_before))

                # (2) Read state + tsi hysteresis update (BEFORE compute_obs)
                wp.synchronize()
                bq_before = state.body_q.numpy()
                cable_pos_now = bq_before[cable_bodies, :3]

                clamp_r_pos = _compute_clamp_pos(
                    bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3],
                    bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7],
                )
                new_center_r = int(np.argmin(np.linalg.norm(cable_pos_now - clamp_r_pos, axis=1)))
                prev_center_r = (
                    target_seg_indices_r[GRIP_SEG_WINDOW]
                    if len(target_seg_indices_r) > GRIP_SEG_WINDOW
                    else target_seg_indices_r[0]
                )
                if abs(new_center_r - prev_center_r) > 1:
                    target_seg_indices_r = list(
                        np.clip(
                            np.arange(new_center_r - GRIP_SEG_WINDOW, new_center_r + GRIP_SEG_WINDOW + 1),
                            0,
                            len(cable_pos_now) - 1,
                        ).astype(int)
                    )

                clamp_l_pos = _compute_clamp_pos(bq_before[EE_BODY_OFFSET][:3], bq_before[EE_BODY_OFFSET][3:7])
                new_center_l = int(np.argmin(np.linalg.norm(cable_pos_now - clamp_l_pos, axis=1)))
                prev_center_l = (
                    target_seg_indices_l[GRIP_SEG_WINDOW]
                    if len(target_seg_indices_l) > GRIP_SEG_WINDOW
                    else target_seg_indices_l[0]
                )
                if abs(new_center_l - prev_center_l) > 1:
                    target_seg_indices_l = list(
                        np.clip(
                            np.arange(new_center_l - GRIP_SEG_WINDOW, new_center_l + GRIP_SEG_WINDOW + 1),
                            0,
                            len(cable_pos_now) - 1,
                        ).astype(int)
                    )

                # (3) Compute obs + append. P7: guard against NaN leaking
                # into BC training data — upstream NaN checks cover cable
                # body_q and IK joint_q, but compute_obs mixes in FK joint
                # state and error metrics where a silent NaN would later
                # cause BC loss to diverge with no obvious source.
                fk_jq_now = fk_state.joint_q.numpy()
                obs, prev_clamp_r_quat, prev_clamp_l_quat, prev_seg_r_quat, prev_seg_l_quat = compute_obs(
                    state,
                    fk_jq_now,
                    cable_bodies,
                    target_seg_indices_r,
                    target_seg_indices_l,
                    prev_clamp_r_quat=prev_clamp_r_quat,
                    prev_clamp_l_quat=prev_clamp_l_quat,
                    prev_seg_r_quat=prev_seg_r_quat,
                    prev_seg_l_quat=prev_seg_l_quat,
                )
                # Source-level NaN guard — all 5 arrays checked, not redundant:
                # obs channels [3:7]/[11:15]/[19:23] mirror prev_clamp_r/l and
                # prev_seg_r directly, but prev_seg_l only reaches obs[36:39]
                # via _compute_ori_error_axis_angle, which propagates NaN
                # (not sanitized). Source-level checks prevent a NaN-poisoned
                # prev from surviving into the next rl_step.
                # Sequential checks log which array tripped for diagnostics.
                nan_source = None
                if np.any(~np.isfinite(obs)):
                    nan_source = "obs"
                elif np.any(~np.isfinite(prev_clamp_r_quat)):
                    nan_source = "prev_clamp_r_quat"
                elif np.any(~np.isfinite(prev_clamp_l_quat)):
                    nan_source = "prev_clamp_l_quat"
                elif np.any(~np.isfinite(prev_seg_r_quat)):
                    nan_source = "prev_seg_r_quat"
                elif np.any(~np.isfinite(prev_seg_l_quat)):
                    nan_source = "prev_seg_l_quat"
                if nan_source is not None:
                    print(f"  [precision] compute_obs NaN in {nan_source} at rl_step {rl_step} — aborting")
                    return _balanced_return(
                        obs_list, act_list, False, float("inf"), precision_ok=precision_succeeded or not ran_precision
                    )
                obs_list.append(obs)

                # (4) Capture EE state + adaptive scale for NEXT window
                ee_r_before = bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET].copy()
                ee_l_before = bq_before[EE_BODY_OFFSET].copy()
                r_scale_before = _adaptive_pos_scale(clamp_r_pos, cable_pos_now, target_seg_indices_r)
                l_scale_before = _adaptive_pos_scale(clamp_l_pos, cable_pos_now, target_seg_indices_l)

                # (5) Sustain check at T_DIST (2mm — Clamp env threshold)
                seg_pos_r, tan_r, dist_pos_r = _find_nearest_cable_point_demo(
                    cable_pos_now, clamp_r_pos, target_seg_indices_r
                )
                seg_pos_l, tan_l, dist_pos_l = _find_nearest_cable_point_demo(
                    cable_pos_now, clamp_l_pos, target_seg_indices_l
                )
                # P9/P10: canonicalize target quats with w>=0 so iter-to-iter
                # sign flips from compute_hand_quat_for_cable cannot masquerade
                # as a rotation jump (q and -q represent the same rotation but
                # dot(q,q')<0 can flip dist_ori classification / IK residual).
                target_quat_r = _normalize_quat_w_positive(compute_hand_quat_for_cable(tan_r))
                target_quat_l = _normalize_quat_w_positive(compute_hand_quat_for_cable(tan_l))
                ee_quat_r = bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7]
                ee_quat_l = bq_before[EE_BODY_OFFSET][3:7]
                dist_ori_r = _quat_distance(ee_quat_r, target_quat_r)
                dist_ori_l = _quat_distance(ee_quat_l, target_quat_l)
                pos_ok_precision = max(dist_pos_r, dist_pos_l) < T_DIST
                ori_ok_precision = max(dist_ori_r, dist_ori_l) < T_ALIGN
                if pos_ok_precision and ori_ok_precision:
                    precision_sustain += 1
                else:
                    precision_sustain = 0
                final_dist_mm = max(dist_pos_r, dist_pos_l) * 1000.0

                # AC env contract (env :1554-1562) uses T_DIST_APPROACH=12mm;
                # track that separately so the outer sustain/success flags
                # reflect the AC definition (precision is a bonus refinement).
                if max(dist_pos_r, dist_pos_l) < T_DIST_APPROACH and max(dist_ori_r, dist_ori_l) < T_ALIGN:
                    sustain_counter += 1
                else:
                    sustain_counter = 0
                if sustain_counter >= K_GRASP:
                    ever_succeeded = True

                # Early exit when T_DIST sustained
                if precision_sustain >= K_GRASP:
                    precision_succeeded = True
                    # C2: the fresh ee_r_before / ee_l_before captured at
                    # step (4) above has had no physics advance since, so
                    # _compute_cumulative_action would produce a ~0 delta
                    # amplified by MIN_POS_SCALE=5e-4 denominators up to
                    # ~2 in normalized action space. Record an explicit
                    # zero-action instead to keep the terminal transition
                    # consistent with its "no motion, task done" meaning
                    # (matches the for/else path which captures real
                    # physics delta across a full substep window).
                    act_list.append(np.zeros(12, dtype=np.float32))
                    print(f"  [precision] Sustained T_DIST at rl_step {rl_step}, dist={final_dist_mm:.2f}mm")
                    break

                # (6) Proportional sub-target at the clamp (fingertip), then
                # back out the EE target via the target rotation. Applying the
                # delta at the EE would leave a ΔR·EE_TO_FINGERTIP drift at
                # the clamp (sin(5°)·0.22m ≈ 19mm > PRECISION_SCALE=3mm), so
                # the IK must solve for an EE pose whose rotated fingertip
                # offset lands on the desired clamp target. P4: clamp the
                # raw proportional delta by its L2 norm (not per-axis) so
                # the step cap is a true 3mm sphere; per-axis np.clip would
                # allow sqrt(3)·3mm ≈ 5.2mm diagonals that disagree with
                # the sin(5°)·0.22m comparison above.
                err_r = (clamp_r_pos - seg_pos_r).astype(np.float32)
                err_l = (clamp_l_pos - seg_pos_l).astype(np.float32)
                delta_r = (-PRECISION_GAIN * err_r).astype(np.float32)
                delta_l = (-PRECISION_GAIN * err_l).astype(np.float32)
                norm_r = float(np.linalg.norm(delta_r))
                norm_l = float(np.linalg.norm(delta_l))
                if norm_r > PRECISION_SCALE:
                    delta_r = delta_r * (PRECISION_SCALE / norm_r)
                if norm_l > PRECISION_SCALE:
                    delta_l = delta_l * (PRECISION_SCALE / norm_l)
                target_clamp_r = (clamp_r_pos + delta_r).astype(np.float32)
                target_clamp_l = (clamp_l_pos + delta_l).astype(np.float32)

                # (7) Target rotation reuses the piecewise-linear tangent from
                # _find_nearest_cable_point_demo (via target_quat_r/l above),
                # which is the same tangent driving dist_ori in the sustain
                # check. compute_cable_tangent (central-difference at
                # new_center) would diverge from the edge tangent and cause
                # oscillation in ori error.
                rot_r_xyzw = target_quat_r
                rot_l_xyzw = target_quat_l
                fingertip_offset = np.array([0.0, 0.0, EE_TO_FINGERTIP], dtype=np.float32)
                ee_offset_r = _quat_rotate_vec(rot_r_xyzw, fingertip_offset)
                ee_offset_l = _quat_rotate_vec(rot_l_xyzw, fingertip_offset)
                target_r_sub = tuple((target_clamp_r - ee_offset_r).tolist())
                target_l_sub = tuple((target_clamp_l - ee_offset_l).tolist())

                rot_r_sub = wp.vec4(
                    float(rot_r_xyzw[0]), float(rot_r_xyzw[1]), float(rot_r_xyzw[2]), float(rot_r_xyzw[3])
                )
                rot_l_sub = wp.vec4(
                    float(rot_l_xyzw[0]), float(rot_l_xyzw[1]), float(rot_l_xyzw[2]), float(rot_l_xyzw[3])
                )

                # (8) Solve IK for sub-target. C1: rot_weight=1.0 so the
                # back-out cancellation above holds — at weight=0.5 the
                # 100-iter residual can exceed 0.5° which translates to
                # >2mm fingertip drift through the 0.22m EE→fingertip arm.
                jq_sub_target = solve_ik(
                    fk_model,
                    fk_state,
                    target_l_sub,
                    target_r_sub,
                    device,
                    rot_left=rot_l_sub,
                    rot_right=rot_r_sub,
                    rot_weight=1.0,
                )
                if np.any(~np.isfinite(jq_sub_target)):
                    print(f"  [precision] IK NaN at rl_step {rl_step} — aborting")
                    return _balanced_return(
                        obs_list, act_list, False, float("inf"), precision_ok=precision_succeeded or not ran_precision
                    )
                jq_sub_target[7] = FINGER_OPEN_POS
                jq_sub_target[8] = FINGER_OPEN_POS
                jq_sub_target[FRANKA_NUM_JOINTS + 7] = FINGER_OPEN_POS
                jq_sub_target[FRANKA_NUM_JOINTS + 8] = FINGER_OPEN_POS

                jq_sub_start = fk_state.joint_q.numpy().copy()

                # C1 diagnostic: measure IK rotation residual for the
                # solved target by forward-kinematicking jq_sub_target
                # through fk_state (non-destructive — we restore
                # jq_sub_start afterwards for the interpolation below).
                fk_state.joint_q.assign(jq_sub_target)
                newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)
                bq_ik = fk_state.body_q.numpy()
                # F2: pos residual (EE-space) before the existing rot residual.
                # Reads from the SAME bq_ik that the rot path uses - single FK
                # evaluation of jq_sub_target covers both. target_r_sub and
                # target_l_sub are EE targets (target_clamp - ee_offset).
                ik_pos_r = bq_ik[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3]
                ik_pos_l = bq_ik[EE_BODY_OFFSET][:3]
                ik_pos_res_r = float(np.linalg.norm(ik_pos_r - np.asarray(target_r_sub, dtype=np.float32)))
                ik_pos_res_l = float(np.linalg.norm(ik_pos_l - np.asarray(target_l_sub, dtype=np.float32)))
                ik_pos_res_max = max(ik_pos_res_r, ik_pos_res_l)
                ik_pos_res_max_observed = max(ik_pos_res_max_observed, ik_pos_res_max)
                # Existing rot residual path (unchanged):
                ik_quat_r = bq_ik[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7]
                ik_quat_l = bq_ik[EE_BODY_OFFSET][3:7]
                ik_res_r = _quat_distance(ik_quat_r, rot_r_xyzw)
                ik_res_l = _quat_distance(ik_quat_l, rot_l_xyzw)
                ik_res_max = max(ik_res_r, ik_res_l)
                ik_res_max_observed = max(ik_res_max_observed, ik_res_max)
                # Per-arm fingertip drift upper bound. Rotation-induced
                # displacement at radius r for angle θ is the chord length
                # 2·sin(θ/2)·r, which is the MAX displacement over all
                # rotation axes (tight when the axis is perpendicular to the
                # fingertip offset; strictly greater than actual displacement
                # when the axis parallels the offset, which gives zero).
                # Combined with EE-space pos residual via triangle inequality,
                # the sum upper-bounds per-arm fingertip deviation for any
                # ik_res ∈ [0, π] (IK NaN-guard at L882-885 covers out-of-range).
                # Note: plan v4.1 pseudocode used the linearization sin(θ)·r;
                # chord 2·sin(θ/2)·r is the correct geometric bound and
                # numerically identical (~1e-6 diff) at the 0.009 rad regime
                # the precision loop typically operates in.
                drift_r = 2.0 * math.sin(ik_res_r / 2.0) * EE_TO_FINGERTIP + ik_pos_res_r
                drift_l = 2.0 * math.sin(ik_res_l / 2.0) * EE_TO_FINGERTIP + ik_pos_res_l
                drift_max = max(drift_r, drift_l)
                # Threshold 1.8mm = T_DIST * 0.9 - early-warning headroom
                # below T_DIST=2mm. Honest note: this is an upper bound on
                # fingertip drift, so drift_max <= 1.8mm guarantees actual
                # fingertip drift <= 1.8mm < T_DIST. The bound can over-
                # trigger when pos_res and rot-induced displacement cancel
                # at the fingertip (false positive acceptable for diagnostic).
                if drift_max > T_DIST * 0.9:
                    print(
                        f"  [precision] rl_step {rl_step}: IK residual "
                        f"rot R={math.degrees(ik_res_r):.2f}° "
                        f"L={math.degrees(ik_res_l):.2f}° "
                        f"pos_ee R={ik_pos_res_r * 1000:.2f}mm "
                        f"L={ik_pos_res_l * 1000:.2f}mm "
                        f"(per-arm drift "
                        f"R={drift_r * 1000:.2f}mm L={drift_l * 1000:.2f}mm, "
                        f"max={drift_max * 1000:.2f}mm / "
                        f"T_DIST={T_DIST * 1000:.0f}mm)"
                    )
                # Restore fk_state to pre-IK joint_q for the interpolation
                # loop below — sub=0 expects fk_state == jq_sub_start.
                fk_state.joint_q.assign(jq_sub_start)
                newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)

                # (9) Execute PHYSICS_STEPS_PER_RL physics steps, linear interp
                nan_abort = False
                for sub in range(PHYSICS_STEPS_PER_RL):
                    t_interp = (sub + 1) / PHYSICS_STEPS_PER_RL
                    jq_interp = jq_sub_start.copy()
                    for d in range(coord_count):
                        jq_interp[d] = jq_sub_start[d] + (jq_sub_target[d] - jq_sub_start[d]) * t_interp
                    fk_state.joint_q.assign(jq_interp)
                    newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)
                    for _ in range(demo_substeps):
                        update_kinematic_bodies(state, fk_state, ROBOT_BODY_COUNT)
                        state.clear_forces()
                        model.collide(state, contacts)
                        solver.step(state, state_buf, control, contacts, demo_dt)
                        state, state_buf = state_buf, state
                    bq_check = state.body_q.numpy()
                    if np.any(~np.isfinite(bq_check[ROBOT_BODY_COUNT:])):
                        print(f"  [precision] Cable NaN at rl_step {rl_step}, sub {sub}")
                        nan_abort = True
                        break
                if nan_abort:
                    return _balanced_return(
                        obs_list, act_list, False, float("inf"), precision_ok=precision_succeeded or not ran_precision
                    )
            else:
                # Loop completed without early exit — append final action
                if ee_r_before is not None:
                    act_list.append(_compute_cumulative_action(state, r_scale_before, l_scale_before))

            # P11: disambiguate the exit reason so rl_step+1 is not
            # conflated between "sustain hit" and "budget exhausted".
            # NaN aborts return early and never reach this print.
            exit_reason = "sustain" if precision_succeeded else "timeout"
            print(
                f"  [precision] Done: exit={exit_reason}, "
                f"rl_steps={rl_step + 1}, "
                f"final_dist={final_dist_mm:.2f}mm, "
                f"sustain={precision_sustain}/{K_GRASP}, "
                f"ik_res_max={math.degrees(ik_res_max_observed):.2f}°, "
                f"ik_pos_res_ee_max={ik_pos_res_max_observed * 1000:.2f}mm, "
                f"succeeded={precision_succeeded}"
            )
            continue

        # Number of steps: max of both arm distances (target-driven phases)
        dist_r = np.linalg.norm(np.array(target_r) - pos_r)
        dist_l = np.linalg.norm(np.array(target_l) - pos_l)
        dist = max(dist_r, dist_l)
        n_arm_steps = max(int(dist * 100 * STEPS_PER_CM), 200)
        n_arm_steps = min(n_arm_steps, MAX_MOVE_STEPS)

        if desc == "hold":
            n_steps = max(K_GRASP * PHYSICS_STEPS_PER_RL * 3, 200)
            jq_start = fk_state.joint_q.numpy().copy()
            jq_target = jq_start.copy()
            jq_target[7] = FINGER_OPEN_POS
            jq_target[8] = FINGER_OPEN_POS
            jq_target[FRANKA_NUM_JOINTS + 7] = FINGER_OPEN_POS
            jq_target[FRANKA_NUM_JOINTS + 8] = FINGER_OPEN_POS
        else:
            # Finger interpolation
            fk_jq = fk_state.joint_q.numpy()
            current_finger_r = fk_jq[FRANKA_NUM_JOINTS + 7]
            finger_change = abs(target_finger_r - current_finger_r) > 0.001
            n_steps = max(n_arm_steps, FINGER_INTERP_STEPS) if finger_change else n_arm_steps

            # Compute cable-adaptive rotation from cable tangent.
            # L1: use the edge tangent returned by
            # _find_nearest_cable_point_demo (identical source to the
            # precision phase) so the descend→precision transition does
            # not inject a rotation delta the first precision iter has
            # to burn budget on. compute_cable_tangent's central-difference
            # at a body index can drift several degrees from the edge
            # tangent when the nearest projection lands mid-segment.
            cable_pos = bq[cable_bodies, :3]
            clamp_r_pos_phase = _compute_clamp_pos(
                bq[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3], bq[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7]
            )
            clamp_l_pos_phase = _compute_clamp_pos(bq[EE_BODY_OFFSET][:3], bq[EE_BODY_OFFSET][3:7])
            if right_seg_index is not None:
                _, tangent_r, _ = _find_nearest_cable_point_demo(cable_pos, clamp_r_pos_phase, target_seg_indices_r)
                rot_r_xyzw = _normalize_quat_w_positive(compute_hand_quat_for_cable(tangent_r))
                rot_r = wp.vec4(float(rot_r_xyzw[0]), float(rot_r_xyzw[1]), float(rot_r_xyzw[2]), float(rot_r_xyzw[3]))
            else:
                rot_r = None
            if left_seg_index is not None:
                _, tangent_l, _ = _find_nearest_cable_point_demo(cable_pos, clamp_l_pos_phase, target_seg_indices_l)
                rot_l_xyzw = _normalize_quat_w_positive(compute_hand_quat_for_cable(tangent_l))
                rot_l = wp.vec4(float(rot_l_xyzw[0]), float(rot_l_xyzw[1]), float(rot_l_xyzw[2]), float(rot_l_xyzw[3]))
            else:
                rot_l = None

            # Solve IK with cable-adaptive rotation
            jq_target = solve_ik(
                fk_model, fk_state, tuple(target_l), tuple(target_r), device, rot_left=rot_l, rot_right=rot_r
            )
            if np.any(~np.isfinite(jq_target)):
                print(f"  [{desc}] IK NaN — aborting episode")
                return _balanced_return(
                    obs_list, act_list, False, float("inf"), precision_ok=precision_succeeded or not ran_precision
                )

            # Set finger targets
            jq_target[7] = target_finger_r  # left: same as right (dual clamp)
            jq_target[8] = target_finger_r
            jq_target[FRANKA_NUM_JOINTS + 7] = target_finger_r
            jq_target[FRANKA_NUM_JOINTS + 8] = target_finger_r

            jq_start = fk_state.joint_q.numpy().copy()

        for step in range(n_steps):
            t = min((step + 1) / n_steps, 1.0)

            # P6: skip the record boundary in the hold phase — jq_target
            # is a copy of jq_start (zero motion) so every recorded pair
            # would be a ~null-action transition that biases BC toward
            # inaction. The hold phase still runs physics for the env
            # contract's sustain window, it just does not emit demos.
            do_record = (step % record_every == 0) and (desc != "hold")
            if do_record:
                wp.synchronize()

                # (1) Action for PREVIOUS window (uses OLD ee_r_before / scales)
                if ee_r_before is not None:
                    act_list.append(_compute_cumulative_action(state, r_scale_before, l_scale_before))

                # (2) Read current state + tsi hysteresis update (BEFORE compute_obs)
                bq_before = state.body_q.numpy()
                cable_pos_now = bq_before[cable_bodies, :3]

                clamp_r_pos = _compute_clamp_pos(
                    bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3],
                    bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7],
                )
                new_center_r = int(np.argmin(np.linalg.norm(cable_pos_now - clamp_r_pos, axis=1)))
                prev_center_r = (
                    target_seg_indices_r[GRIP_SEG_WINDOW]
                    if len(target_seg_indices_r) > GRIP_SEG_WINDOW
                    else target_seg_indices_r[0]
                )
                if abs(new_center_r - prev_center_r) > 1:
                    target_seg_indices_r = list(
                        np.clip(
                            np.arange(new_center_r - GRIP_SEG_WINDOW, new_center_r + GRIP_SEG_WINDOW + 1),
                            0,
                            len(cable_pos_now) - 1,
                        ).astype(int)
                    )

                clamp_l_pos = _compute_clamp_pos(bq_before[EE_BODY_OFFSET][:3], bq_before[EE_BODY_OFFSET][3:7])
                new_center_l = int(np.argmin(np.linalg.norm(cable_pos_now - clamp_l_pos, axis=1)))
                prev_center_l = (
                    target_seg_indices_l[GRIP_SEG_WINDOW]
                    if len(target_seg_indices_l) > GRIP_SEG_WINDOW
                    else target_seg_indices_l[0]
                )
                if abs(new_center_l - prev_center_l) > 1:
                    target_seg_indices_l = list(
                        np.clip(
                            np.arange(new_center_l - GRIP_SEG_WINDOW, new_center_l + GRIP_SEG_WINDOW + 1),
                            0,
                            len(cable_pos_now) - 1,
                        ).astype(int)
                    )

                # (3) Compute obs with fresh tsi. P7: NaN-guard mirrors
                # the precision-phase check so a numeric blow-up in FK or
                # error metrics cannot poison BC training silently.
                fk_jq_now = fk_state.joint_q.numpy()
                obs, prev_clamp_r_quat, prev_clamp_l_quat, prev_seg_r_quat, prev_seg_l_quat = compute_obs(
                    state,
                    fk_jq_now,
                    cable_bodies,
                    target_seg_indices_r,
                    target_seg_indices_l,
                    prev_clamp_r_quat=prev_clamp_r_quat,
                    prev_clamp_l_quat=prev_clamp_l_quat,
                    prev_seg_r_quat=prev_seg_r_quat,
                    prev_seg_l_quat=prev_seg_l_quat,
                )
                # Source-level NaN guard (see precision-phase rationale).
                # On pre-precision NaN (ran_precision=False), precision_ok
                # evaluates to True because "precision did not run" is not a
                # precision failure. success=False (passed below) still drops
                # the demo via main()'s keep-filter, so precision_ok=True here
                # means "precision was not the failing phase" rather than
                # "precision succeeded". Post-precision NaN inherits the true
                # precision_succeeded value.
                nan_source = None
                if np.any(~np.isfinite(obs)):
                    nan_source = "obs"
                elif np.any(~np.isfinite(prev_clamp_r_quat)):
                    nan_source = "prev_clamp_r_quat"
                elif np.any(~np.isfinite(prev_clamp_l_quat)):
                    nan_source = "prev_clamp_l_quat"
                elif np.any(~np.isfinite(prev_seg_r_quat)):
                    nan_source = "prev_seg_r_quat"
                elif np.any(~np.isfinite(prev_seg_l_quat)):
                    nan_source = "prev_seg_l_quat"
                if nan_source is not None:
                    print(f"  [{desc}] compute_obs NaN in {nan_source} at step {step} — aborting")
                    return _balanced_return(
                        obs_list, act_list, False, float("inf"), precision_ok=precision_succeeded or not ran_precision
                    )
                obs_list.append(obs)

                # (4) Capture EE state and adaptive scale for NEXT window
                ee_r_before = bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET].copy()
                ee_l_before = bq_before[EE_BODY_OFFSET].copy()
                # Use clamp (fingertip) position + piecewise-linear projection
                # for env :1700-1712 parity.
                r_scale_before = _adaptive_pos_scale(clamp_r_pos, cable_pos_now, target_seg_indices_r)
                l_scale_before = _adaptive_pos_scale(clamp_l_pos, cable_pos_now, target_seg_indices_l)

                # (5) Sustain check (env :1554-1562 criterion). P9/P10:
                # canonicalize target quats with w>=0 to match the
                # precision-phase sustain check and prevent sign-flip
                # jitter from spuriously zeroing sustain_counter.
                seg_pos_r, tan_r, dist_pos_r = _find_nearest_cable_point_demo(
                    cable_pos_now, clamp_r_pos, target_seg_indices_r
                )
                seg_pos_l, tan_l, dist_pos_l = _find_nearest_cable_point_demo(
                    cable_pos_now, clamp_l_pos, target_seg_indices_l
                )
                target_quat_r = _normalize_quat_w_positive(compute_hand_quat_for_cable(tan_r))
                target_quat_l = _normalize_quat_w_positive(compute_hand_quat_for_cable(tan_l))
                ee_quat_r = bq_before[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][3:7]
                ee_quat_l = bq_before[EE_BODY_OFFSET][3:7]
                dist_ori_r = _quat_distance(ee_quat_r, target_quat_r)
                dist_ori_l = _quat_distance(ee_quat_l, target_quat_l)
                pos_ok = max(dist_pos_r, dist_pos_l) < T_DIST_APPROACH
                ori_ok = max(dist_ori_r, dist_ori_l) < T_ALIGN
                if pos_ok and ori_ok:
                    sustain_counter += 1
                else:
                    sustain_counter = 0
                if sustain_counter >= K_GRASP:
                    ever_succeeded = True
                final_dist_mm = max(dist_pos_r, dist_pos_l) * 1000.0

            # Interpolate joint angles
            jq_interp = jq_start.copy()
            for d in range(coord_count):
                jq_interp[d] = jq_start[d] + (jq_target[d] - jq_start[d]) * t

            # Apply FK
            fk_state.joint_q.assign(jq_interp)
            newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)

            # Physics substeps (match RL_SIM_SUBSTEPS for action magnitude consistency)
            demo_dt = DT / demo_substeps
            for _ in range(demo_substeps):
                update_kinematic_bodies(state, fk_state, ROBOT_BODY_COUNT)
                state.clear_forces()
                model.collide(state, contacts)
                solver.step(state, state_buf, control, contacts, demo_dt)
                state, state_buf = state_buf, state

            # NaN check
            bq_check = state.body_q.numpy()
            if np.any(~np.isfinite(bq_check[ROBOT_BODY_COUNT:])):
                print(f"  [{desc}] Cable NaN at step {step}")
                return _balanced_return(
                    obs_list, act_list, False, float("inf"), precision_ok=precision_succeeded or not ran_precision
                )

        # Final action for last window
        if ee_r_before is not None:
            act_list.append(_compute_cumulative_action(state, r_scale_before, l_scale_before))

        print(f"  [{desc}] Done: {n_steps} steps")

    precision_ok = precision_succeeded or not ran_precision
    print(
        f"  [result] ever_succeeded={ever_succeeded}, "
        f"final_dist_mm={final_dist_mm:.1f}mm, "
        f"precision_ok={precision_ok} (ran={ran_precision}, "
        f"succeeded={precision_succeeded})"
    )
    return obs_list, act_list, ever_succeeded, final_dist_mm, precision_ok


def main():
    parser = argparse.ArgumentParser(description="Collect ApproachCable demos (multi-episode, cable-adaptive)")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--cable-noise-y", type=float, default=0.005, help="Cable start Y randomization [m]")
    parser.add_argument("--output", type=str, default="thread_isaac_lab/data/bc_demos/approach_cable_demos_v6.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--record-every",
        type=int,
        default=10,
        help="Record obs/action every N physics steps (must match "
        "NewtonApproachCableEnv.PHYSICS_STEPS_PER_RL for correct DAPG actions)",
    )
    parser.add_argument(
        "--start-from-p0",
        action="store_true",
        help="Start from P0 (env reset state, fingertip ~10mm above cable) "
        "instead of home. Produces shorter demos with env-matched obs range.",
    )
    parser.add_argument(
        "--demo-substeps",
        type=int,
        default=4,
        help="Physics substeps during demo recording (default: 4, "
        "matching RL_SIM_SUBSTEPS in training env). "
        "Settle/init phases always use SIM_SUBSTEPS=10.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100, help="IK interpolation steps for warmup perturbation (0=disable)"
    )
    parser.add_argument(
        "--warmup-pos-sigma", type=float, default=0.015, help="EE position perturbation sigma [m] (default: 15mm)"
    )
    parser.add_argument(
        "--warmup-rot-sigma", type=float, default=0.15, help="EE rotation perturbation sigma [rad] (default: ~8.6 deg)"
    )
    args = parser.parse_args()

    device = args.device
    os.environ["NEWTON_DEVICE"] = device
    import test_newton_clip_routing as _tncr

    _tncr.DEVICE = device

    # F-6: os.path.dirname("foo.npz") == "" → makedirs("") raises
    # FileNotFoundError. Guard with abspath + empty check.
    if not args.output.endswith(".npz"):
        parser.error(f"--output must end in .npz, got {args.output!r}")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[Demo] Episodes: {args.num_episodes}")
    print(f"[Demo] Cable noise Y: ±{args.cable_noise_y * 1000:.1f}mm")
    print(f"[Demo] Start from: {'P0 (env reset state)' if args.start_from_p0 else 'home'}")
    print(f"[Demo] Demo substeps: {args.demo_substeps} (settle/init: {SIM_SUBSTEPS})")
    print(f"[Demo] Output: {args.output}")
    print(f"[Demo] Device: {device}")
    if args.warmup_steps > 0:
        print(
            f"[Demo] Warmup: {args.warmup_steps} steps "
            f"(pos_σ={args.warmup_pos_sigma * 1000:.1f}mm, "
            f"rot_σ={args.warmup_rot_sigma * 180 / math.pi:.1f}°)"
        )

    # Build FK model
    print("[Demo] Building FK model...")
    fk_model = build_fk_model()
    fk_state = fk_model.state()

    # Init FK to home
    fk_jq = fk_state.joint_q.numpy()
    fk_tp = fk_model.joint_target_pos.numpy()
    fk_jq[:] = fk_tp[:]
    fk_jq[7] = FINGER_OPEN_POS
    fk_jq[8] = FINGER_OPEN_POS
    fk_jq[FRANKA_NUM_JOINTS + 7] = FINGER_OPEN_POS
    fk_jq[FRANKA_NUM_JOINTS + 8] = FINGER_OPEN_POS
    fk_state.joint_q.assign(fk_jq)
    newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)

    # Build scene
    print("[Demo] Building physics scene...")
    model, state, solver, contacts, cable_bodies = build_physics_scene(fk_model, fk_state, device)

    # Initial settle (baseline)
    print("[Demo] Initial cable settle (2s)...")
    state = settle_cable(model, state, solver, contacts, fk_state, device)

    # If --start-from-p0: move arms from home to P0 (env reset position), then re-settle
    if args.start_from_p0:
        p0_ee_z = (
            TABLE_HEIGHT + CLIP_BASE_HEIGHT + CABLE_RADIUS + 0.010 + EE_TO_FINGERTIP
        )  # 1.039m (matches env _setup_p0_precondition)
        print(f"[Demo] Moving to P0 (EE Z={p0_ee_z:.3f}m, fingertip ~10mm above cable top)...")

        jq_p0 = solve_ik(fk_model, fk_state, (GRASP_X, WIDE_LEFT_Y, p0_ee_z), (GRASP_X, WIDE_RIGHT_Y, p0_ee_z), device)
        jq_p0[7] = FINGER_OPEN_POS
        jq_p0[8] = FINGER_OPEN_POS
        jq_p0[FRANKA_NUM_JOINTS + 7] = FINGER_OPEN_POS
        jq_p0[FRANKA_NUM_JOINTS + 8] = FINGER_OPEN_POS

        # Interpolate home → P0 (not recorded)
        jq_start = fk_state.joint_q.numpy().copy()
        coord_count = fk_model.joint_coord_count
        control = model.control()
        state_buf = model.state()
        n_interp = 500
        for step in range(n_interp):
            t = (step + 1) / n_interp
            jq_interp = jq_start.copy()
            for d in range(coord_count):
                jq_interp[d] = jq_start[d] + (jq_p0[d] - jq_start[d]) * t
            fk_state.joint_q.assign(jq_interp)
            newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)
            for _ in range(SIM_SUBSTEPS):
                update_kinematic_bodies(state, fk_state, ROBOT_BODY_COUNT)
                state.clear_forces()
                model.collide(state, contacts)
                solver.step(state, state_buf, control, contacts, DT / SIM_SUBSTEPS)
                state, state_buf = state_buf, state

        # Re-settle cable at P0 position
        print("[Demo] Re-settling cable at P0 (1s)...")
        state = settle_cable(model, state, solver, contacts, fk_state, device, duration=1.0)

        # Verify P0 position
        wp.synchronize()
        bq_p0 = state.body_q.numpy()
        ee_z_l = bq_p0[EE_BODY_OFFSET][2]
        ee_z_r = bq_p0[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][2]
        print(f"[Demo] P0 reached: EE Z left={ee_z_l:.4f} right={ee_z_r:.4f} (target={p0_ee_z:.4f})")

    # Save initial state for reset
    init_body_q = state.body_q.numpy().copy()
    init_body_qd = state.body_qd.numpy().copy()
    init_fk_jq = fk_state.joint_q.numpy().copy()

    all_obs = []
    all_act = []
    episode_lengths = []
    success_count = 0

    t0 = time.perf_counter()

    for ep in range(args.num_episodes):
        print(f"\n{'=' * 50}")
        print(f"[Demo] Episode {ep + 1}/{args.num_episodes}")

        # Reset state
        state = reset_state(model, state, solver, init_body_q, init_body_qd, device)
        fk_state.joint_q.assign(init_fk_jq)
        newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)
        update_kinematic_bodies(state, fk_state, ROBOT_BODY_COUNT)

        # Randomize cable Y: shift cable bodies
        if args.cable_noise_y > 0:
            noise_y = np.random.uniform(-args.cable_noise_y, args.cable_noise_y)
            bq = state.body_q.numpy()
            for bi in cable_bodies:
                bq[bi, 1] += noise_y
            state.body_q.assign(bq)
            solver.body_q_prev.assign(bq)
            print(f"  Cable Y noise: {noise_y * 1000:+.1f}mm")

        # Re-settle with noise
        state = settle_cable(model, state, solver, contacts, fk_state, device, duration=1.0)

        # Query cable center Y for adaptive grip positioning
        cable_center_y = query_cable_y(state, cable_bodies, GRASP_X)
        grip_y_l = cable_center_y - GRIP_HALF_SPAN
        grip_y_r = cable_center_y + GRIP_HALF_SPAN
        print(f"  Cable center Y: {cable_center_y:.4f}, grip L={grip_y_l:.4f} R={grip_y_r:.4f}")

        # Compute per-arm target segment indices (±1 window, matching env per-arm logic)
        bq_cable = state.body_q.numpy()
        cable_pos = bq_cable[cable_bodies, :3]
        right_grip_pt = np.array([GRASP_X, grip_y_r, cable_pos[:, 2].mean()])
        left_grip_pt = np.array([GRASP_X, grip_y_l, cable_pos[:, 2].mean()])
        right_seg = int(np.argmin(np.linalg.norm(cable_pos - right_grip_pt, axis=1)))
        left_seg = int(np.argmin(np.linalg.norm(cable_pos - left_grip_pt, axis=1)))
        n_cable = len(cable_bodies)
        tsi_r = list(
            np.clip(np.arange(right_seg - GRIP_SEG_WINDOW, right_seg + GRIP_SEG_WINDOW + 1), 0, n_cable - 1).astype(int)
        )
        tsi_l = list(
            np.clip(np.arange(left_seg - GRIP_SEG_WINDOW, left_seg + GRIP_SEG_WINDOW + 1), 0, n_cable - 1).astype(int)
        )
        print(f"  Target seg R={right_seg} (indices {tsi_r}), L={left_seg} (indices {tsi_l})")

        # Warmup: perturb arm positions + rotations before recording
        if args.warmup_steps > 0:
            bq_wu = state.body_q.numpy()
            ee_r_pos = bq_wu[FRANKA_NUM_JOINTS + EE_BODY_OFFSET][:3]
            ee_l_pos = bq_wu[EE_BODY_OFFSET][:3]

            # Random position offsets (XY only — Z covered by grasp trajectory)
            offset_r = np.random.randn(3) * args.warmup_pos_sigma
            offset_l = np.random.randn(3) * args.warmup_pos_sigma
            offset_r[2] = 0.0
            offset_l[2] = 0.0
            target_r_wu = tuple(ee_r_pos + offset_r)
            target_l_wu = tuple(ee_l_pos + offset_l)

            # Random rotation perturbation (axis-angle → quat)
            rot_perturb_r = np.random.randn(3) * args.warmup_rot_sigma
            rot_perturb_l = np.random.randn(3) * args.warmup_rot_sigma
            # Apply perturbation to cable-adaptive rotation
            cable_pos_wu = bq_wu[cable_bodies, :3]
            tang_r = compute_cable_tangent(cable_pos_wu, right_seg)
            tang_l = compute_cable_tangent(cable_pos_wu, left_seg)
            base_quat_r = compute_hand_quat_for_cable(tang_r)  # xyzw
            base_quat_l = compute_hand_quat_for_cable(tang_l)
            perturb_quat_r = _axis_angle_to_quat_xyzw(rot_perturb_r)
            perturb_quat_l = _axis_angle_to_quat_xyzw(rot_perturb_l)
            wu_quat_r = _quat_multiply_xyzw(perturb_quat_r, base_quat_r)
            wu_quat_l = _quat_multiply_xyzw(perturb_quat_l, base_quat_l)
            rot_r_wp = wp.vec4(float(wu_quat_r[0]), float(wu_quat_r[1]), float(wu_quat_r[2]), float(wu_quat_r[3]))
            rot_l_wp = wp.vec4(float(wu_quat_l[0]), float(wu_quat_l[1]), float(wu_quat_l[2]), float(wu_quat_l[3]))

            # IK solve for perturbed targets
            jq_wu = solve_ik(
                fk_model, fk_state, target_l_wu, target_r_wu, device, rot_left=rot_l_wp, rot_right=rot_r_wp
            )
            if np.all(np.isfinite(jq_wu)):
                # Keep current finger state
                fk_jq_cur = fk_state.joint_q.numpy()
                jq_wu[7] = fk_jq_cur[7]
                jq_wu[8] = fk_jq_cur[8]
                jq_wu[FRANKA_NUM_JOINTS + 7] = fk_jq_cur[FRANKA_NUM_JOINTS + 7]
                jq_wu[FRANKA_NUM_JOINTS + 8] = fk_jq_cur[FRANKA_NUM_JOINTS + 8]

                # Interpolate to perturbed position (not recorded)
                jq_wu_start = fk_jq_cur.copy()
                coord_count = fk_model.joint_coord_count
                control_wu = model.control()
                state_buf_wu = model.state()
                for wu_step in range(args.warmup_steps):
                    t_wu = (wu_step + 1) / args.warmup_steps
                    jq_interp = jq_wu_start.copy()
                    for d in range(coord_count):
                        jq_interp[d] = jq_wu_start[d] + (jq_wu[d] - jq_wu_start[d]) * t_wu
                    fk_state.joint_q.assign(jq_interp)
                    newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)
                    for _ in range(SIM_SUBSTEPS):
                        update_kinematic_bodies(state, fk_state, ROBOT_BODY_COUNT)
                        state.clear_forces()
                        model.collide(state, contacts)
                        solver.step(state, state_buf_wu, control_wu, contacts, DT / SIM_SUBSTEPS)
                        state, state_buf_wu = state_buf_wu, state

                print(
                    f"  Warmup: pos_offset R={np.linalg.norm(offset_r) * 1000:.1f}mm "
                    f"L={np.linalg.norm(offset_l) * 1000:.1f}mm, "
                    f"rot_offset R={np.linalg.norm(rot_perturb_r) * 180 / math.pi:.1f}° "
                    f"L={np.linalg.norm(rot_perturb_l) * 180 / math.pi:.1f}°"
                )
            else:
                print("  Warmup: IK NaN, skipping perturbation")

        obs, act, success, final_dist_mm, precision_ok = execute_approach_episode(
            model,
            state,
            solver,
            contacts,
            fk_model,
            fk_state,
            cable_bodies,
            grip_y_left=grip_y_l,
            grip_y_right=grip_y_r,
            device=device,
            record_every=args.record_every,
            target_seg_indices_r=tsi_r,
            target_seg_indices_l=tsi_l,
            right_seg_index=right_seg,
            left_seg_index=left_seg,
            start_from_p0=args.start_from_p0,
            demo_substeps=args.demo_substeps,
        )

        if len(obs) != len(act):
            raise RuntimeError(
                f"obs/act length mismatch: obs={len(obs)}, act={len(act)} — "
                "check execute_approach_episode for missing append/record"
            )
        ep_len = len(obs)

        if success and precision_ok and ep_len > 0:
            all_obs.extend(obs)
            all_act.extend(act)
            episode_lengths.append(ep_len)
            success_count += 1
            print(f"  [KEPT] {ep_len} transitions (final_dist={final_dist_mm:.1f}mm)")
        else:
            print(
                f"  [DROPPED] success={success}, "
                f"precision_ok={precision_ok}, "
                f"transitions={ep_len}, dist={final_dist_mm:.1f}mm"
            )

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 50}")
    print(f"[Demo] Complete in {elapsed:.1f}s")
    print(f"[Demo] Success: {success_count}/{args.num_episodes}")
    print(f"[Demo] Total transitions: {len(all_obs)}")

    if len(all_obs) > 0:
        obs_arr = np.stack(all_obs)
        act_arr = np.stack(all_act)
        ep_len_arr = np.array(episode_lengths, dtype=np.int32)
        np.savez(args.output, obs=obs_arr, actions=act_arr, episode_lengths=ep_len_arr)
        print(f"[Demo] Saved: {args.output}")
        print(f"  obs: {obs_arr.shape}, actions: {act_arr.shape}, episodes: {ep_len_arr.shape[0]}")
    else:
        print("[Demo] No successful episodes — no output saved")


if __name__ == "__main__":
    main()
