# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test utilities for MDP warp-vs-stable parity tests.

Contains constants, assertion helpers, warp kernel runners, mock classes,
numpy math utilities, and mutation helpers used by the observation, reward,
termination, event, and action parity test files.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

# ---------------------------------------------------------------------------
# Constants (shared across all MDP parity test files)
# ---------------------------------------------------------------------------
NUM_ENVS = 64
NUM_JOINTS = 12
NUM_ACTIONS = 6
DEVICE = "cuda:0"
ATOL = 1e-5
RTOL = 1e-5

# Body/sensor-level defaults shared by observation, reward, and termination tests
NUM_BODIES = 4
NUM_HISTORY = 3
CMD_DIM = 3
BODY_IDS = [0, 2]

# Gravity direction constant (normalized, same as ArticulationData.GRAVITY_VEC_W)
GRAVITY_DIR_NP = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Numpy math utilities
# ---------------------------------------------------------------------------


def quat_rotate_inv_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply inverse quaternion rotation to vectors (numpy, batch).

    Equivalent to ``wp.quat_rotate_inv`` — rotates *v* by the conjugate of *q*.

    Args:
        q_xyzw: (N, 4) quaternion array in [x, y, z, w] order (warp convention).
        v: (N, 3) vector array.

    Returns:
        (N, 3) rotated vectors in float32.
    """
    qv = -q_xyzw[..., :3]  # conjugate xyz
    qw = q_xyzw[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return (v + qw * t + np.cross(qv, t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Warp / numpy utilities
# ---------------------------------------------------------------------------


def copy_np_to_wp(dest: wp.array, src_np: np.ndarray):
    """In-place overwrite of a warp array's contents from numpy (preserves pointer)."""
    tmp = wp.array(src_np, dtype=dest.dtype, device=str(dest.device))
    wp.copy(dest, tmp)


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------


def run_warp_obs(func, env, shape, device=DEVICE, **kwargs):
    """Run a warp observation function and return the result as a torch tensor."""
    out = wp.zeros(shape, dtype=wp.float32, device=device)
    func(env, out, **kwargs)
    return wp.to_torch(out).clone()


def run_warp_obs_captured(func, env, shape, device=DEVICE, **kwargs):
    """Run a warp observation function under CUDA graph capture and return the result."""
    out = wp.zeros(shape, dtype=wp.float32, device=device)
    func(env, out, **kwargs)  # warm-up
    with wp.ScopedCapture() as capture:
        func(env, out, **kwargs)
    wp.capture_launch(capture.graph)
    return wp.to_torch(out).clone()


def run_warp_rew(func, env, device=DEVICE, **kwargs):
    """Run a warp reward function and return the result as a torch tensor."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=device)
    func(env, out, **kwargs)
    return wp.to_torch(out).clone()


def run_warp_rew_captured(func, env, device=DEVICE, **kwargs):
    """Run a warp reward function under CUDA graph capture."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.float32, device=device)
    func(env, out, **kwargs)  # warm-up
    with wp.ScopedCapture() as capture:
        func(env, out, **kwargs)
    wp.capture_launch(capture.graph)
    return wp.to_torch(out).clone()


def run_warp_term(func, env, device=DEVICE, **kwargs):
    """Run a warp termination function and return the result as a torch tensor."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=device)
    func(env, out, **kwargs)
    return wp.to_torch(out).clone()


def run_warp_term_captured(func, env, device=DEVICE, **kwargs):
    """Run a warp termination function under CUDA graph capture."""
    out = wp.zeros((NUM_ENVS,), dtype=wp.bool, device=device)
    func(env, out, **kwargs)  # warm-up
    with wp.ScopedCapture() as capture:
        func(env, out, **kwargs)
    wp.capture_launch(capture.graph)
    return wp.to_torch(out).clone()


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = ATOL, rtol: float = RTOL):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor):
    assert torch.equal(actual, expected), f"Mismatch:\n  actual:   {actual}\n  expected: {expected}"


# ---------------------------------------------------------------------------
# Mock classes (shared across parity test files)
# ---------------------------------------------------------------------------


class MockArticulationData:
    """Mock articulation data backed by Warp arrays (same storage Newton uses).

    Args:
        num_envs: Number of environments.
        num_joints: Number of joints.
        device: Warp device string.
        seed: Random seed for reproducibility.
        num_bodies: Number of bodies. When > 0, generates body-level arrays
            (body_pose_w, body_lin_acc_w, body_com_pos_b) and multi-body
            projected_gravity_b. When 0, projected_gravity_b is root-level
            (derived from root quaternion).
    """

    def __init__(self, num_envs=NUM_ENVS, num_joints=NUM_JOINTS, device=DEVICE, seed=42, num_bodies=0):
        rng = np.random.RandomState(seed)

        # --- Joint state (float32 2D) ---
        self.joint_pos = wp.array(rng.randn(num_envs, num_joints).astype(np.float32), device=device)
        self.joint_vel = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 2.0, device=device)
        self.joint_acc = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 0.5, device=device)
        self.default_joint_pos = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 0.01, device=device)
        self.default_joint_vel = wp.array(np.zeros((num_envs, num_joints), dtype=np.float32), device=device)
        self.applied_torque = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 10.0, device=device)
        self.computed_torque = wp.array(rng.randn(num_envs, num_joints).astype(np.float32) * 10.0, device=device)

        # --- Soft joint limits ---
        limits_np = np.zeros((num_envs, num_joints, 2), dtype=np.float32)
        limits_np[:, :, 0] = -3.14
        limits_np[:, :, 1] = 3.14
        self.soft_joint_pos_limits = wp.array(limits_np, dtype=wp.vec2f, device=device)
        self.soft_joint_vel_limits = wp.array(np.full((num_envs, num_joints), 10.0, dtype=np.float32), device=device)

        # --- Root state ---
        root_pos_np = rng.randn(num_envs, 3).astype(np.float32)
        root_pos_np[:, 2] = np.abs(root_pos_np[:, 2]) + 0.1  # positive heights
        self.root_pos_w = wp.array(root_pos_np, dtype=wp.vec3f, device=device)

        # Unit quaternions
        quat_np = rng.randn(num_envs, 4).astype(np.float32)
        quat_np /= np.linalg.norm(quat_np, axis=1, keepdims=True)
        self.root_quat_w = wp.array(quat_np, dtype=wp.quatf, device=device)

        # Tier 1 compound: root_link_pose_w (transformf = pos + quat)
        pose_np = np.zeros((num_envs, 7), dtype=np.float32)
        pose_np[:, :3] = root_pos_np
        pose_np[:, 3:] = quat_np
        self.root_link_pose_w = wp.array(pose_np, dtype=wp.transformf, device=device)

        # World-frame velocities
        lin_vel_w_np = rng.randn(num_envs, 3).astype(np.float32)
        ang_vel_w_np = rng.randn(num_envs, 3).astype(np.float32)
        self.root_lin_vel_w = wp.array(lin_vel_w_np, dtype=wp.vec3f, device=device)
        self.root_ang_vel_w = wp.array(ang_vel_w_np, dtype=wp.vec3f, device=device)

        # Tier 1 compound: root_com_vel_w (spatial_vectorf: top=linear, bottom=angular)
        vel_np = np.zeros((num_envs, 6), dtype=np.float32)
        vel_np[:, :3] = lin_vel_w_np
        vel_np[:, 3:] = ang_vel_w_np
        self.root_com_vel_w = wp.array(vel_np, dtype=wp.spatial_vectorf, device=device)

        # Gravity direction constant
        self.GRAVITY_VEC_W = wp.vec3f(0.0, 0.0, -1.0)

        # Derived body-frame quantities (consistent with Tier 1 compounds)
        self.root_lin_vel_b = wp.array(quat_rotate_inv_np(quat_np, lin_vel_w_np), dtype=wp.vec3f, device=device)
        self.root_ang_vel_b = wp.array(quat_rotate_inv_np(quat_np, ang_vel_w_np), dtype=wp.vec3f, device=device)

        # --- projected_gravity_b and body-level data ---
        if num_bodies > 0:
            # Multi-body projected_gravity_b: (num_envs, num_bodies) vec3f
            grav_np = rng.randn(num_envs, num_bodies, 3).astype(np.float32)
            grav_np[:, :, 2] = -1.0
            grav_np /= np.linalg.norm(grav_np, axis=2, keepdims=True)
            self.projected_gravity_b = wp.array(grav_np, dtype=wp.vec3f, device=device)

            # body_pose_w: (num_envs, num_bodies) transformf
            bpose_np = np.zeros((num_envs, num_bodies, 7), dtype=np.float32)
            bpose_np[:, :, :3] = rng.randn(num_envs, num_bodies, 3).astype(np.float32)
            bpose_np[:, :, 3:7] = [0.0, 0.0, 0.0, 1.0]
            self.body_pose_w = wp.array(bpose_np, dtype=wp.transformf, device=device)

            # body_lin_acc_w: (num_envs, num_bodies) vec3f
            self.body_lin_acc_w = wp.array(
                rng.randn(num_envs, num_bodies, 3).astype(np.float32), dtype=wp.vec3f, device=device
            )

            # body_com_pos_b: (num_envs, num_bodies) vec3f
            self.body_com_pos_b = wp.array(
                rng.randn(num_envs, num_bodies, 3).astype(np.float32) * 0.01, dtype=wp.vec3f, device=device
            )
        else:
            # Root-level projected_gravity_b: (num_envs,) vec3f — derived from root quat
            self.projected_gravity_b = wp.array(
                quat_rotate_inv_np(quat_np, np.tile(GRAVITY_DIR_NP, (num_envs, 1))),
                dtype=wp.vec3f,
                device=device,
            )

        # --- Event-specific data ---
        self.root_vel_w = wp.array(rng.randn(num_envs, 6).astype(np.float32), dtype=wp.spatial_vectorf, device=device)

        default_pose_np = np.zeros((num_envs, 7), dtype=np.float32)
        default_pose_np[:, 0:3] = rng.randn(num_envs, 3).astype(np.float32) * 0.1
        default_pose_np[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        self.default_root_pose = wp.array(default_pose_np, dtype=wp.transformf, device=device)

        self.default_root_vel = wp.array(
            np.zeros((num_envs, 6), dtype=np.float32), dtype=wp.spatial_vectorf, device=device
        )

    def resolve_joint_mask(self, joint_ids=None):
        n = self.joint_pos.shape[1]
        mask = [False] * n
        if joint_ids is None or isinstance(joint_ids, slice):
            mask = [True] * n
        else:
            for j in joint_ids:
                mask[j] = True
        return wp.array(mask, dtype=wp.bool, device=str(self.joint_pos.device))


class MockArticulation:
    """Mock articulation asset with simulation write stubs.

    Provides both no-op write stubs (for event tests) and tracking write stubs
    (for action tests). The ``last_*_target`` attributes record the most recent
    values passed to ``set_joint_*_target``, enabling verification in action tests.
    """

    def __init__(self, data: MockArticulationData, num_bodies: int = 1, num_joints: int = NUM_JOINTS):
        self.data = data
        self.num_bodies = num_bodies
        self.num_joints = num_joints
        self.device = DEVICE
        self._joint_names = [f"joint_{i}" for i in range(num_joints)]
        # Tracking attributes for action tests
        self.last_pos_target = None
        self.last_vel_target = None
        self.last_effort_target = None
        self.last_joint_mask = None

    # -- Simulation write stubs (no-op, for event tests) --------------------

    def write_root_velocity_to_sim(self, *a, **kw):
        pass

    def write_root_pose_to_sim(self, *a, **kw):
        pass

    def write_joint_state_to_sim(self, *a, **kw):
        pass

    def set_external_force_and_torque(self, *a, **kw):
        pass

    # -- Action write stubs (tracking, for action tests) --------------------

    def set_joint_position_target(self, target, joint_ids=None, joint_mask=None):
        self.last_pos_target = target
        self.last_joint_mask = joint_mask

    def set_joint_velocity_target(self, target, joint_ids=None, joint_mask=None):
        self.last_vel_target = target
        self.last_joint_mask = joint_mask

    def set_joint_effort_target(self, target, joint_ids=None, joint_mask=None):
        self.last_effort_target = target
        self.last_joint_mask = joint_mask

    def set_joint_effort_target_index(self, target, joint_ids=None):
        self.last_effort_target = target

    # -- Query stubs --------------------------------------------------------

    def find_joints(self, names, preserve_order=False):
        if isinstance(names, list) and names == [".*"]:
            return list(range(self.num_joints)), list(self._joint_names)
        ids = []
        resolved = []
        for name in names if isinstance(names, list) else [names]:
            for i, jn in enumerate(self._joint_names):
                if (name in jn or name == jn or name == ".*") and i not in ids:
                    ids.append(i)
                    resolved.append(jn)
        if not ids:
            ids = list(range(self.num_joints))
            resolved = list(self._joint_names)
        return ids, resolved

    def find_bodies(self, name):
        return [0], [name]


class MockScene:
    """Mock scene with asset lookup, env origins, and optional sensors."""

    def __init__(self, assets: dict, env_origins, sensors=None):
        self._assets = assets
        self.env_origins = env_origins
        self.sensors = sensors or {}
        self.articulations = dict(assets)
        self.rigid_objects = {}
        self.num_envs = NUM_ENVS

    def __getitem__(self, name: str):
        return self._assets[name]


# ---------------------------------------------------------------------------
# Root-state mutation helper
# ---------------------------------------------------------------------------


def mutate_root_state(rng: np.random.RandomState, art_data: MockArticulationData, num_envs: int = NUM_ENVS):
    """Mutate root-level state arrays in-place (preserves buffer pointers).

    Updates root_pos_w, root_quat_w, root_link_pose_w, root_com_vel_w,
    root_lin_vel_w, root_ang_vel_w, root_lin_vel_b, root_ang_vel_b, and
    (when 1D) projected_gravity_b — all consistently derived from a fresh
    random quaternion and world-frame velocities.
    """
    root_pos_np = rng.randn(num_envs, 3).astype(np.float32)
    root_pos_np[:, 2] = np.abs(root_pos_np[:, 2]) + 0.05
    copy_np_to_wp(art_data.root_pos_w, root_pos_np)

    quat_np = rng.randn(num_envs, 4).astype(np.float32)
    quat_np /= np.linalg.norm(quat_np, axis=1, keepdims=True)
    copy_np_to_wp(art_data.root_quat_w, quat_np)

    pose_np = np.zeros((num_envs, 7), dtype=np.float32)
    pose_np[:, :3] = root_pos_np
    pose_np[:, 3:] = quat_np
    copy_np_to_wp(art_data.root_link_pose_w, pose_np)

    lin_vel_w_np = rng.randn(num_envs, 3).astype(np.float32)
    ang_vel_w_np = rng.randn(num_envs, 3).astype(np.float32)
    copy_np_to_wp(art_data.root_lin_vel_w, lin_vel_w_np)
    copy_np_to_wp(art_data.root_ang_vel_w, ang_vel_w_np)

    vel_np = np.zeros((num_envs, 6), dtype=np.float32)
    vel_np[:, :3] = lin_vel_w_np
    vel_np[:, 3:] = ang_vel_w_np
    copy_np_to_wp(art_data.root_com_vel_w, vel_np)

    copy_np_to_wp(art_data.root_lin_vel_b, quat_rotate_inv_np(quat_np, lin_vel_w_np))
    copy_np_to_wp(art_data.root_ang_vel_b, quat_rotate_inv_np(quat_np, ang_vel_w_np))

    # Root-level projected_gravity_b (1D) is derived from quat.
    # Multi-body (2D) is mutated separately by callers.
    if art_data.projected_gravity_b.ndim == 1:
        copy_np_to_wp(
            art_data.projected_gravity_b,
            quat_rotate_inv_np(quat_np, np.tile(GRAVITY_DIR_NP, (num_envs, 1))),
        )


class MockActionManagerWarp:
    """Returns warp arrays (for experimental functions)."""

    def __init__(self, action_wp: wp.array, prev_action_wp: wp.array):
        self._action = action_wp
        self._prev_action = prev_action_wp

    @property
    def action(self) -> wp.array:
        return self._action

    @property
    def prev_action(self) -> wp.array:
        return self._prev_action


class MockActionManagerTorch:
    """Returns torch tensors (for stable functions)."""

    def __init__(self, action_wp: wp.array, prev_action_wp: wp.array):
        self._action = wp.to_torch(action_wp)
        self._prev_action = wp.to_torch(prev_action_wp)

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action


# ---------------------------------------------------------------------------
# Shared mock classes (previously duplicated across test files)
# ---------------------------------------------------------------------------


class MockSceneEntityCfg:
    """Unified cfg that works for both stable (joint_ids) and experimental (joint_mask / joint_ids_wp)."""

    def __init__(self, name: str, joint_ids: list[int], num_joints: int, device: str):
        self.name = name
        self.joint_ids = joint_ids

        # Experimental extras
        mask = [False] * num_joints
        for idx in joint_ids:
            mask[idx] = True
        self.joint_mask = wp.array(mask, dtype=wp.bool, device=device)
        self.joint_ids_wp = wp.array(joint_ids, dtype=wp.int32, device=device)


class MockContactSensorData:
    """Mock contact sensor data with random force history.

    Stores ``net_forces_w_history`` as a warp ``vec3f`` 3D array of shape
    ``(num_envs, num_history, num_bodies)``.  Both warp kernels (which read
    the warp array directly) and stable functions (which call
    ``wp.to_torch``) work with this representation.
    """

    def __init__(self, num_envs=NUM_ENVS, num_history=NUM_HISTORY, num_bodies=NUM_BODIES, device=DEVICE, seed=77):
        rng = np.random.RandomState(seed)
        self.net_forces_w_history = wp.array(
            rng.randn(num_envs, num_history, num_bodies, 3).astype(np.float32),
            dtype=wp.vec3f,
            device=device,
        )


class MockContactSensor:
    """Mock contact sensor wrapping :class:`MockContactSensorData`."""

    def __init__(self, data: MockContactSensorData, num_bodies: int = NUM_BODIES):
        self.data = data
        self.num_bodies = num_bodies


class MockCommandTerm:
    """Mock command term with time_left and command_counter."""

    def __init__(self, num_envs=NUM_ENVS, device=DEVICE, seed=88):
        rng = np.random.RandomState(seed)
        self.time_left = torch.tensor(rng.rand(num_envs).astype(np.float32) * 0.05, device=device)
        self.command_counter = torch.tensor(rng.randint(0, 3, (num_envs,)), dtype=torch.float32, device=device)


class MockCommandManager:
    """Mock command manager returning a fixed command tensor and term."""

    def __init__(self, command_tensor: torch.Tensor, cmd_term: MockCommandTerm):
        self._cmd = command_tensor
        self._term = cmd_term

    def get_command(self, name: str) -> torch.Tensor:
        return self._cmd

    def get_term(self, name: str):
        return self._term


class MockBodyCfg:
    """SceneEntityCfg-like object for body-level reward/termination terms."""

    def __init__(self, name="robot", body_ids=None):
        self.name = name
        self.body_ids = body_ids if body_ids is not None else list(BODY_IDS)


class MockSensorCfg:
    """SceneEntityCfg-like object for contact sensor terms.

    Provides both ``body_ids`` (for stable functions) and ``body_ids_wp``
    (for experimental warp functions).
    """

    def __init__(self, name="contact_sensor", body_ids=None, device=DEVICE):
        self.name = name
        self.body_ids = body_ids if body_ids is not None else list(BODY_IDS)
        self.body_ids_wp = wp.array(self.body_ids, dtype=wp.int32, device=device)


class MockTerminationManager:
    """Mock termination manager providing both torch and warp terminated buffers."""

    def __init__(self, num_envs=NUM_ENVS, device=DEVICE):
        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.terminated_wp = wp.from_torch(self.terminated)


# ---------------------------------------------------------------------------
# Art-data mutation helpers (previously duplicated in obs/rew/term test files)
# ---------------------------------------------------------------------------


def mutate_art_data(
    art_data: MockArticulationData,
    warp_env,
    num_envs: int = NUM_ENVS,
    num_joints: int = NUM_JOINTS,
    num_actions: int = NUM_ACTIONS,
    rng_seed: int = 200,
):
    """Mutate every data array in-place so captured graphs see fresh values."""
    rng = np.random.RandomState(rng_seed)

    copy_np_to_wp(art_data.joint_pos, rng.randn(num_envs, num_joints).astype(np.float32) * 1.5)
    copy_np_to_wp(art_data.joint_vel, rng.randn(num_envs, num_joints).astype(np.float32) * 3.0)
    copy_np_to_wp(art_data.joint_acc, rng.randn(num_envs, num_joints).astype(np.float32) * 0.8)
    copy_np_to_wp(art_data.default_joint_pos, rng.randn(num_envs, num_joints).astype(np.float32) * 0.02)
    copy_np_to_wp(art_data.applied_torque, rng.randn(num_envs, num_joints).astype(np.float32) * 12.0)
    copy_np_to_wp(art_data.computed_torque, rng.randn(num_envs, num_joints).astype(np.float32) * 12.0)

    mutate_root_state(rng, art_data, num_envs)

    copy_np_to_wp(warp_env.action_manager._action, rng.randn(num_envs, num_actions).astype(np.float32))
    copy_np_to_wp(warp_env.action_manager._prev_action, rng.randn(num_envs, num_actions).astype(np.float32))

    warp_env.episode_length_buf[:] = torch.randint(0, 500, (num_envs,), dtype=torch.int64, device=DEVICE)

    wp.synchronize()


def mutate_body_data(
    art_data: MockArticulationData,
    num_envs: int = NUM_ENVS,
    num_bodies: int = NUM_BODIES,
    rng_seed: int = 200,
):
    """Mutate body-level and root-level data in-place so captured graphs see fresh values."""
    rng = np.random.RandomState(rng_seed)

    mutate_root_state(rng, art_data, num_envs)

    grav_np = rng.randn(num_envs, num_bodies, 3).astype(np.float32)
    grav_np[:, :, 2] = -1.0
    grav_np /= np.linalg.norm(grav_np, axis=2, keepdims=True)
    copy_np_to_wp(art_data.projected_gravity_b, grav_np)

    copy_np_to_wp(art_data.body_lin_acc_w, rng.randn(num_envs, num_bodies, 3).astype(np.float32))

    pose_np = np.zeros((num_envs, num_bodies, 7), dtype=np.float32)
    pose_np[:, :, :3] = rng.randn(num_envs, num_bodies, 3).astype(np.float32)
    pose_np[:, :, 3:7] = [0.0, 0.0, 0.0, 1.0]
    copy_np_to_wp(art_data.body_pose_w, pose_np)

    wp.synchronize()
