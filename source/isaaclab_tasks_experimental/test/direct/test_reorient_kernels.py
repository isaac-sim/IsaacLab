# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity unit tests for the reorient family Warp kernels.

Kit-free: imports only torch/warp and the kernel module. Every kernel is
checked against an independent torch reference implementation on randomized
inputs, on CPU and CUDA.
"""

from __future__ import annotations

import pytest
import torch
import warp as wp
from isaaclab_tasks_experimental.direct.reorient.reorient_kernels import (
    ReorientRewardBuffers,
    compute_cube_keypoints,
    cube_keypoints_from_quat_kernel,
    ema_actuation_kernel,
    fingertip_pos_kernel,
    fingertip_quat_kernel,
    fingertip_vel_kernel,
    full_obs_kernel,
    out_of_reach_kernel,
    reduced_obs_kernel,
    reorient_progress_kernel,
    reorient_reward,
    reorient_success_kernel,
    rotation_distance_kernel,
)

wp.init()

N_ENVS = 33
N_JOINTS = 24
N_BODIES = 26
N_FINGERS = 5
N_ACTIONS = 20

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


# -- torch reference helpers (independent implementations, (x, y, z, w) layout) --


def _rand_unit_quat(n: int, device: str) -> torch.Tensor:
    q = torch.randn(n, 4, device=device)
    return torch.nn.functional.normalize(q, dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def _rotation_distance_ref(q_obj: torch.Tensor, q_goal: torch.Tensor) -> torch.Tensor:
    qe = _quat_mul(q_obj, _quat_conj(q_goal))
    return 2.0 * torch.asin(torch.clamp(qe[..., :3].norm(dim=-1), max=1.0))


def _wp2d(t: torch.Tensor, dtype) -> wp.array:
    return wp.from_torch(t.contiguous(), dtype=dtype)


# -- tests --


@pytest.mark.parametrize("device", DEVICES)
def test_rotation_distance_kernel_matches_reference(device):
    torch.manual_seed(0)
    q1, q2 = _rand_unit_quat(N_ENVS, device), _rand_unit_quat(N_ENVS, device)
    dist = torch.empty(N_ENVS, dtype=torch.float32, device=device)
    wp.launch(
        rotation_distance_kernel,
        dim=N_ENVS,
        inputs=[_wp2d(q1, wp.quatf), _wp2d(q2, wp.quatf)],
        outputs=[wp.from_torch(dist)],
        device=device,
    )
    torch.testing.assert_close(dist, _rotation_distance_ref(q1, q2), atol=1e-4, rtol=1e-4)
    # identical quaternions have zero distance
    wp.launch(
        rotation_distance_kernel,
        dim=N_ENVS,
        inputs=[_wp2d(q1, wp.quatf), _wp2d(q1.clone(), wp.quatf)],
        outputs=[wp.from_torch(dist)],
        device=device,
    )
    assert dist.abs().max().item() < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_success_kernel_flags_and_distance(device):
    torch.manual_seed(1)
    q1, q2 = _rand_unit_quat(N_ENVS, device), _rand_unit_quat(N_ENVS, device)
    tol = 0.4
    success = torch.empty(N_ENVS, dtype=torch.bool, device=device)
    dist = torch.empty(N_ENVS, dtype=torch.float32, device=device)
    wp.launch(
        reorient_success_kernel,
        dim=N_ENVS,
        inputs=[_wp2d(q1, wp.quatf), _wp2d(q2, wp.quatf), tol],
        outputs=[wp.from_torch(success), wp.from_torch(dist)],
        device=device,
    )
    ref_dist = _rotation_distance_ref(q1, q2)
    torch.testing.assert_close(dist, ref_dist, atol=1e-4, rtol=1e-4)
    assert torch.equal(success, dist <= tol)


@pytest.mark.parametrize("device", DEVICES)
def test_ema_actuation_kernel_matches_reference(device):
    torch.manual_seed(2)
    lower = -1.5 + 0.1 * torch.rand(N_ENVS, N_JOINTS, device=device)
    upper = 1.5 + 0.1 * torch.rand(N_ENVS, N_JOINTS, device=device)
    dof_ids = torch.randperm(N_JOINTS, device=device)[:N_ACTIONS].to(torch.int32)
    actions = 2.0 * torch.rand(N_ENVS, N_ACTIONS, device=device) - 1.0
    prev = torch.zeros(N_ENVS, N_JOINTS, device=device)
    prev_ref = prev.clone()
    cur = torch.zeros_like(prev)
    compact = torch.zeros(N_ENVS, N_ACTIONS, device=device)
    m = 0.7
    wp.launch(
        ema_actuation_kernel,
        dim=(N_ENVS, N_ACTIONS),
        inputs=[
            _wp2d(actions, wp.float32),
            _wp2d(lower, wp.float32),
            _wp2d(upper, wp.float32),
            wp.from_torch(dof_ids),
            m,
        ],
        outputs=[_wp2d(prev, wp.float32), _wp2d(cur, wp.float32), _wp2d(compact, wp.float32)],
        device=device,
    )
    ids = dof_ids.long()
    lo, hi = lower[:, ids], upper[:, ids]
    t = 0.5 * (actions + 1.0) * (hi - lo) + lo
    t = m * t + (1.0 - m) * prev_ref[:, ids]
    t = torch.clamp(t, lo, hi)
    torch.testing.assert_close(compact, t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cur[:, ids], t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prev[:, ids], t, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_fingertip_gather_kernels_match_reference(device):
    torch.manual_seed(3)
    body_pos = torch.randn(N_ENVS, N_BODIES, 3, device=device)
    body_quat = _rand_unit_quat(N_ENVS * N_BODIES, device).view(N_ENVS, N_BODIES, 4)
    body_vel = torch.randn(N_ENVS, N_BODIES, 6, device=device)
    origins = torch.randn(N_ENVS, 3, device=device)
    ids = torch.randperm(N_BODIES, device=device)[:N_FINGERS].to(torch.int32)
    out_pos = torch.empty(N_ENVS, 3 * N_FINGERS, device=device)
    out_quat = torch.empty(N_ENVS, 4 * N_FINGERS, device=device)
    out_vel = torch.empty(N_ENVS, 6 * N_FINGERS, device=device)
    wp.launch(
        fingertip_pos_kernel,
        dim=(N_ENVS, N_FINGERS),
        inputs=[_wp2d(body_pos, wp.vec3f), _wp2d(origins, wp.vec3f), wp.from_torch(ids)],
        outputs=[_wp2d(out_pos, wp.float32)],
        device=device,
    )
    wp.launch(
        fingertip_quat_kernel,
        dim=(N_ENVS, N_FINGERS),
        inputs=[_wp2d(body_quat, wp.quatf), wp.from_torch(ids)],
        outputs=[_wp2d(out_quat, wp.float32)],
        device=device,
    )
    wp.launch(
        fingertip_vel_kernel,
        dim=(N_ENVS, N_FINGERS),
        inputs=[_wp2d(body_vel, wp.spatial_vectorf), wp.from_torch(ids)],
        outputs=[_wp2d(out_vel, wp.float32)],
        device=device,
    )
    lids = ids.long()
    torch.testing.assert_close(out_pos, (body_pos[:, lids] - origins[:, None]).reshape(N_ENVS, -1))
    torch.testing.assert_close(out_quat, body_quat[:, lids].reshape(N_ENVS, -1))
    torch.testing.assert_close(out_vel, body_vel[:, lids].reshape(N_ENVS, -1))


@pytest.mark.parametrize("device", DEVICES)
def test_out_of_reach_kernel_matches_reference(device):
    torch.manual_seed(4)
    obj = torch.randn(N_ENVS, 3, device=device)
    origins = torch.randn(N_ENVS, 3, device=device)
    target = torch.randn(N_ENVS, 3, device=device)
    fall = 1.0
    out = torch.empty(N_ENVS, dtype=torch.bool, device=device)
    wp.launch(
        out_of_reach_kernel,
        dim=N_ENVS,
        inputs=[_wp2d(obj, wp.vec3f), _wp2d(origins, wp.vec3f), _wp2d(target, wp.vec3f), fall],
        outputs=[wp.from_torch(out)],
        device=device,
    )
    ref = (obj - origins - target).norm(dim=-1) >= fall
    assert torch.equal(out, ref)


@pytest.mark.parametrize("device", DEVICES)
def test_cube_keypoints_match_reference(device):
    torch.manual_seed(5)
    pose = torch.cat([torch.randn(N_ENVS, 3, device=device), _rand_unit_quat(N_ENVS, device)], dim=-1)
    out = torch.empty(N_ENVS, 8, 3, dtype=torch.float32, device=device)
    compute_cube_keypoints(_wp2d(pose, wp.float32), wp.from_torch(out, dtype=wp.vec3))
    half = torch.tensor([0.03, 0.03, 0.03], device=device)
    signs = torch.tensor(
        [[1 - 2 * ((c >> b) & 1) for b in range(3)] for c in range(8)], dtype=torch.float32, device=device
    )
    corners = signs * half  # (8, 3)
    ref = pose[:, None, :3] + _quat_rotate(pose[:, None, 3:7].expand(-1, 8, -1), corners.expand(N_ENVS, -1, -1))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    # quat-only variant: rotated offsets without the position term
    out2 = torch.empty(N_ENVS, 24, dtype=torch.float32, device=device)
    wp.launch(
        cube_keypoints_from_quat_kernel,
        dim=(N_ENVS, 8),
        inputs=[_wp2d(pose[:, 3:7], wp.quatf), wp.vec3(0.03, 0.03, 0.03)],
        outputs=[_wp2d(out2, wp.float32)],
        device=device,
    )
    ref2 = _quat_rotate(pose[:, None, 3:7].expand(-1, 8, -1), corners.expand(N_ENVS, -1, -1))
    torch.testing.assert_close(out2.view(N_ENVS, 8, 3), ref2, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reorient_reward_matches_reference(device):
    torch.manual_seed(6)
    obj = 0.2 * torch.randn(N_ENVS, 3, device=device)
    origins = torch.randn(N_ENVS, 3, device=device)
    target_pos = 0.2 * torch.randn(N_ENVS, 3, device=device)
    q_obj, q_goal = _rand_unit_quat(N_ENVS, device), _rand_unit_quat(N_ENVS, device)
    actions = 2.0 * torch.rand(N_ENVS, N_ACTIONS, device=device) - 1.0
    reset_buf = torch.rand(N_ENVS, device=device) < 0.2
    reset_goal_buf = torch.rand(N_ENVS, device=device) < 0.2
    successes_t = torch.randint(0, 5, (N_ENVS,), device=device).float()
    goal_resets = wp.from_torch(reset_goal_buf.clone())
    successes = wp.from_torch(successes_t.clone())
    consecutive_t = torch.tensor([1.5], device=device)
    consecutive = wp.array([1.5], dtype=wp.float32, device=device)
    params = dict(
        distance_scale=-10.0,
        rotation_scale=1.0,
        rotation_epsilon=0.1,
        action_penalty_scale=-0.0002,
        success_tolerance=0.6,
        success_bonus=250.0,
        fall_distance=0.24,
        fall_penalty=-50.0,
        averaging_factor=0.1,
    )
    buffers = ReorientRewardBuffers(N_ENVS, device)
    reward = reorient_reward(
        wp.from_torch(reset_buf),
        goal_resets,
        successes,
        consecutive,
        _wp2d(obj, wp.vec3f),
        _wp2d(origins, wp.vec3f),
        _wp2d(q_obj, wp.quatf),
        _wp2d(target_pos, wp.vec3f),
        _wp2d(q_goal, wp.quatf),
        _wp2d(actions, wp.float32),
        buffers=buffers,
        **params,
    )
    # torch reference
    goal_dist = (obj - origins - target_pos).norm(dim=-1)
    rot_dist = _rotation_distance_ref(q_obj, q_goal)
    goal_reset_ref = (rot_dist <= params["success_tolerance"]) | reset_goal_buf
    penalty = (actions**2).sum(dim=-1)
    value = (
        goal_dist * params["distance_scale"]
        + params["rotation_scale"] / (rot_dist + params["rotation_epsilon"])
        + penalty * params["action_penalty_scale"]
    )
    value = value + torch.where(goal_reset_ref, params["success_bonus"], 0.0)
    fell = goal_dist >= params["fall_distance"]
    value = value + torch.where(fell, params["fall_penalty"], 0.0)
    resets_ref = (fell | reset_buf).float()
    succ_ref = successes_t + goal_reset_ref.float()
    num_resets = resets_ref.sum()
    finished = (succ_ref * resets_ref).sum()
    consec_ref = torch.where(
        num_resets > 0, params["averaging_factor"] * finished / num_resets + 0.9 * consecutive_t, consecutive_t
    )
    torch.testing.assert_close(wp.to_torch(reward), value, atol=1e-3, rtol=1e-4)
    assert torch.equal(wp.to_torch(goal_resets), goal_reset_ref)
    torch.testing.assert_close(wp.to_torch(successes), succ_ref)
    torch.testing.assert_close(wp.to_torch(consecutive), consec_ref, atol=1e-4, rtol=1e-4)


def _obs_inputs(device):
    torch.manual_seed(7)
    data = dict(
        joint_pos=torch.rand(N_ENVS, N_JOINTS, device=device) - 0.5,
        joint_vel=torch.randn(N_ENVS, N_JOINTS, device=device),
        lower=-1.0 - torch.rand(N_ENVS, N_JOINTS, device=device),
        upper=1.0 + torch.rand(N_ENVS, N_JOINTS, device=device),
        obj=torch.randn(N_ENVS, 3, device=device),
        origins=torch.randn(N_ENVS, 3, device=device),
        q_obj=_rand_unit_quat(N_ENVS, device),
        lin_vel=torch.randn(N_ENVS, 3, device=device),
        ang_vel=torch.randn(N_ENVS, 3, device=device),
        in_hand=torch.randn(N_ENVS, 3, device=device),
        q_goal=_rand_unit_quat(N_ENVS, device),
        body_pos=torch.randn(N_ENVS, N_BODIES, 3, device=device),
        body_quat=_rand_unit_quat(N_ENVS * N_BODIES, device).view(N_ENVS, N_BODIES, 4),
        body_vel=torch.randn(N_ENVS, N_BODIES, 6, device=device),
        force=torch.randn(N_ENVS, N_BODIES, 3, device=device),
        torque=torch.randn(N_ENVS, N_BODIES, 3, device=device),
        actions=2.0 * torch.rand(N_ENVS, N_ACTIONS, device=device) - 1.0,
    )
    data["finger_ids"] = torch.randperm(N_BODIES, device=device)[:N_FINGERS].to(torch.int32)
    data["wrench_ids"] = torch.randperm(N_BODIES, device=device)[:N_FINGERS].to(torch.int32)
    return data


def _full_obs_ref(d, vel_scale, force_scale, with_forces):
    fids, wids = d["finger_ids"].long(), d["wrench_ids"].long()
    qe = _quat_mul(d["q_obj"], _quat_conj(d["q_goal"]))
    segs = [
        2.0 * (d["joint_pos"] - d["lower"]) / (d["upper"] - d["lower"]) - 1.0,
        vel_scale * d["joint_vel"],
        d["obj"] - d["origins"],
        d["q_obj"],
        d["lin_vel"],
        vel_scale * d["ang_vel"],
        d["in_hand"],
        d["q_goal"],
        qe,
        (d["body_pos"][:, fids] - d["origins"][:, None]).reshape(N_ENVS, -1),
        d["body_quat"][:, fids].reshape(N_ENVS, -1),
        d["body_vel"][:, fids].reshape(N_ENVS, -1),
    ]
    # with_forces == 0: observation has no wrench segment at all; any other value
    # reserves the segment — real data for 1, zero block otherwise (sensor not ready)
    if with_forces != 0:
        wrench = torch.cat([d["force"][:, wids, None, :], d["torque"][:, wids, None, :]], dim=2)
        wrench = force_scale * wrench.reshape(N_ENVS, -1)
        if with_forces != 1:
            wrench = torch.zeros_like(wrench)
        segs.append(wrench)
    segs.append(d["actions"])
    return torch.cat(segs, dim=-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_forces", [1, 0, 2])
def test_full_obs_kernel_matches_reference(device, with_forces):
    d = _obs_inputs(device)
    vel_scale, force_scale = 0.2, 0.05
    dim = 2 * N_JOINTS + 13 + 11 + 13 * N_FINGERS + N_ACTIONS
    if with_forces != 0:
        dim += 6 * N_FINGERS
    out = torch.empty(N_ENVS, dim, dtype=torch.float32, device=device)
    wp.launch(
        full_obs_kernel,
        dim=(N_ENVS, dim),
        inputs=[
            _wp2d(d["joint_pos"], wp.float32),
            _wp2d(d["joint_vel"], wp.float32),
            _wp2d(d["lower"], wp.float32),
            _wp2d(d["upper"], wp.float32),
            _wp2d(d["obj"], wp.vec3f),
            _wp2d(d["origins"], wp.vec3f),
            _wp2d(d["q_obj"], wp.quatf),
            _wp2d(d["lin_vel"], wp.vec3f),
            _wp2d(d["ang_vel"], wp.vec3f),
            _wp2d(d["in_hand"], wp.vec3f),
            _wp2d(d["q_goal"], wp.quatf),
            _wp2d(d["body_pos"], wp.vec3f),
            _wp2d(d["body_quat"], wp.quatf),
            _wp2d(d["body_vel"], wp.spatial_vectorf),
            wp.from_torch(d["finger_ids"]),
            _wp2d(d["force"], wp.vec3f),
            _wp2d(d["torque"], wp.vec3f),
            wp.from_torch(d["wrench_ids"]),
            _wp2d(d["actions"], wp.float32),
            vel_scale,
            force_scale,
            with_forces,
        ],
        outputs=[_wp2d(out, wp.float32)],
        device=device,
    )
    ref = _full_obs_ref(d, vel_scale, force_scale, with_forces)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_obs_kernel_matches_reference(device):
    d = _obs_inputs(device)
    fids = d["finger_ids"].long()
    dim = 3 * N_FINGERS + 7 + N_ACTIONS
    out = torch.empty(N_ENVS, dim, dtype=torch.float32, device=device)
    wp.launch(
        reduced_obs_kernel,
        dim=N_ENVS,
        inputs=[
            _wp2d(d["body_pos"], wp.vec3f),
            _wp2d(d["origins"], wp.vec3f),
            wp.from_torch(d["finger_ids"]),
            _wp2d(d["obj"], wp.vec3f),
            _wp2d(d["q_obj"], wp.quatf),
            _wp2d(d["q_goal"], wp.quatf),
            _wp2d(d["actions"], wp.float32),
        ],
        outputs=[_wp2d(out, wp.float32)],
        device=device,
    )
    qe = _quat_mul(d["q_obj"], _quat_conj(d["q_goal"]))
    ref = torch.cat(
        [
            (d["body_pos"][:, fids] - d["origins"][:, None]).reshape(N_ENVS, -1),
            d["obj"] - d["origins"],
            qe,
            d["actions"],
        ],
        dim=-1,
    )
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_consecutive", [50.0, 0.0])
def test_progress_kernel_matches_reference(device, max_consecutive):
    torch.manual_seed(8)
    success = torch.rand(N_ENVS, device=device) < 0.3
    distance = torch.rand(N_ENVS, device=device)
    distance[0] = float("inf")  # non-finite samples must not touch the tracking state
    successes = torch.randint(0, 100, (N_ENVS,), device=device).float()
    ep_len = torch.randint(1, 400, (N_ENVS,), device=device, dtype=torch.long)
    ep_len_ref = ep_len.clone()
    max_ep_len = 300  # below the episode-length range so both time-out outcomes occur
    time_out = torch.empty(N_ENVS, dtype=torch.bool, device=device)
    minimum = torch.full((N_ENVS,), 0.5, device=device)
    minimum_ref = minimum.clone()
    has_sample = torch.zeros(N_ENVS, dtype=torch.bool, device=device)
    wp.launch(
        reorient_progress_kernel,
        dim=N_ENVS,
        inputs=[
            wp.from_torch(success),
            wp.from_torch(distance),
            wp.from_torch(successes),
            max_consecutive,
            max_ep_len,
        ],
        outputs=[
            wp.from_torch(ep_len),
            wp.from_torch(minimum),
            wp.from_torch(time_out),
            wp.from_torch(has_sample),
        ],
        device=device,
    )
    if max_consecutive > 0.0:
        ep_ref = torch.where(success, torch.zeros_like(ep_len_ref), ep_len_ref)
        reached_ref = successes >= max_consecutive
    else:
        ep_ref = ep_len_ref
        reached_ref = torch.zeros(N_ENVS, dtype=torch.bool, device=device)
    time_out_ref = (ep_ref >= max_ep_len - 1) | reached_ref
    finite = torch.isfinite(distance)
    min_ref = torch.where(finite & (distance < minimum_ref), distance, minimum_ref)
    torch.testing.assert_close(ep_len, ep_ref)
    assert torch.equal(time_out, time_out_ref)
    torch.testing.assert_close(minimum, min_ref)
    assert torch.equal(has_sample, finite)
