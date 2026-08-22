# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate terminations (experimental).

All functions in this file follow the Warp-compatible termination signature expected by
`isaaclab_experimental.managers.TerminationManager`:

- ``func(env, out, **params) -> None``

where ``out`` is a pre-allocated Warp array of shape ``(num_envs,)`` with boolean dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab_experimental.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.manager_term_cfg import TerminationTermCfg


"""
MDP terminations.
"""


@wp.kernel
def _time_out_kernel(
    episode_length: wp.array(dtype=wp.int64), max_episode_length: wp.int64, out: wp.array(dtype=wp.bool)
):
    i = wp.tid()
    out[i] = episode_length[i] >= max_episode_length


def time_out(env: ManagerBasedRLEnv, out) -> None:
    """Terminate the episode when episode length exceeds the maximum episode length."""
    wp.launch(
        kernel=_time_out_kernel,
        dim=env.num_envs,
        inputs=[env._episode_length_buf_wp, env.max_episode_length, out],
        device=env.device,
    )


@wp.kernel
def _pose_command_success_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    body_pos_w: wp.array(dtype=wp.vec3f, ndim=2),
    body_quat_w: wp.array(dtype=wp.quatf, ndim=2),
    cmd: wp.array(dtype=wp.float32, ndim=2),
    body_idx: int,
    check_position: bool,
    position_threshold: float,
    check_orientation: bool,
    orientation_threshold: float,
    succeeded: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.bool),
):
    """Flag envs whose body pose is within every configured threshold.

    Each threshold is an accept predicate, matching the stable term's ``error < threshold``: a
    non-finite error compares false and denies success, where a reject form would let it through.
    """
    i = wp.tid()
    success = bool(True)
    if check_position:
        des_b = wp.vec3f(cmd[i, 0], cmd[i, 1], cmd[i, 2])
        des_w = root_pos_w[i] + wp.quat_rotate(root_quat_w[i], des_b)
        cur_w = body_pos_w[i, body_idx]
        dx = cur_w[0] - des_w[0]
        dy = cur_w[1] - des_w[1]
        dz = cur_w[2] - des_w[2]
        success = success and (wp.sqrt(dx * dx + dy * dy + dz * dz) < position_threshold)
    if check_orientation:
        des_q_b = wp.quatf(cmd[i, 3], cmd[i, 4], cmd[i, 5], cmd[i, 6])
        des_q_w = root_quat_w[i] * des_q_b
        q_err = wp.quat_inverse(body_quat_w[i, body_idx]) * des_q_w
        # 2*atan2(|xyz|, |w|), matching the stable axis-angle magnitude. Both arguments scale
        # with the quaternion, so this is norm-invariant, where 2*acos(|w|) would understate
        # the error for a non-unit input. |w| takes the shortest path, as stable's sign flip does.
        axis = wp.vec3f(q_err[0], q_err[1], q_err[2])
        angle = 2.0 * wp.atan2(wp.length(axis), wp.abs(q_err[3]))
        success = success and (angle < orientation_threshold)
    out[i] = success
    # sticky per-episode tracker, mirroring the stable term's ``self._succeeded |= success``.
    # It must be written here rather than after the launch: the termination manager runs
    # graph-captured by default, and host-side work between launches is not replayed.
    if success:
        succeeded[i] = True


class pose_command_success(ManagerTermBase):
    """Terminate environments whose pose command satisfies all configured success thresholds.

    Warp-first override of :func:`isaaclab.envs.mdp.terminations.pose_command_success`.

    The command term, its thresholds, the tracked body index and the warp views of the command
    and success buffers are all resolved at init, leaving :meth:`__call__` a single kernel launch.

    Note:
        Thresholds are read once. A curriculum that mutates ``position_success_threshold`` or
        ``orientation_success_threshold`` at runtime is not honoured here: the termination manager
        runs graph-captured by default, and a kernel scalar is baked into the graph at capture.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        command_name = cfg.params["command_name"]
        command = env.command_manager.get_term(command_name)
        # carry "configured" as its own flag: a threshold is any float, so no value can encode absence
        position_threshold = command.cfg.position_success_threshold
        orientation_threshold = command.cfg.orientation_success_threshold
        self._check_position = position_threshold is not None
        self._check_orientation = orientation_threshold is not None
        self._position_threshold = float(position_threshold or 0.0)
        self._orientation_threshold = float(orientation_threshold or 0.0)
        # matches the stable term: with no threshold configured, no env ever succeeds
        self._any_threshold = self._check_position or self._check_orientation
        self._body_idx = command.body_idx
        self._asset: Articulation = command.robot
        cmd = env.command_manager.get_command(command_name)
        self._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)
        # zero-copy alias of the command's persistent sticky-success buffer (allocated once, never
        # reallocated), so the kernel's write lands in the tensor ``reset()`` reads and clears.
        self._succeeded_wp = wp.from_torch(command._succeeded) if self._any_threshold else None

    def __call__(self, env: ManagerBasedRLEnv, out: wp.array(dtype=wp.bool), command_name: str) -> None:
        if not self._any_threshold:
            out.zero_()
            return
        wp.launch(
            kernel=_pose_command_success_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_pos_w.warp,
                self._asset.data.root_quat_w.warp,
                self._asset.data.body_pos_w.warp,
                self._asset.data.body_quat_w.warp,
                self._cmd_wp,
                self._body_idx,
                self._check_position,
                self._position_threshold,
                self._check_orientation,
                self._orientation_threshold,
                self._succeeded_wp,
                out,
            ],
            device=env.device,
        )


"""
Root terminations.
"""


@wp.kernel
def _root_height_below_min_kernel(
    root_pos_w: wp.array(dtype=wp.vec3f),
    minimum_height: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    out[i] = root_pos_w[i][2] < minimum_height


def root_height_below_minimum(
    env: ManagerBasedRLEnv, out, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Terminate when the asset's root height is below the minimum height."""
    asset: Articulation = env.scene[asset_cfg.name]
    wp.launch(
        kernel=_root_height_below_min_kernel,
        dim=env.num_envs,
        inputs=[asset.data.root_pos_w.warp, minimum_height, out],
        device=env.device,
    )


"""
Joint terminations.
"""


@wp.kernel
def _joint_pos_out_of_manual_limit_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    lower: float,
    upper: float,
    out: wp.array(dtype=wp.bool),
):
    """2D kernel (num_envs, num_joints). ``out`` is pre-zeroed; only writes True."""
    i, j = wp.tid()
    if joint_mask[j]:
        v = joint_pos[i, j]
        if v < lower or v > upper:
            out[i] = True


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, out, bounds: tuple[float, float], asset_cfg: SceneEntityCfg
) -> None:
    """Terminate when joint positions are outside configured bounds. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_mask is None:
        raise ValueError(
            f"joint_pos_out_of_manual_limit requires SceneEntityCfg with resolved joint_mask, "
            f"but got None for asset '{asset_cfg.name}'."
        )
    if asset.data.joint_pos.warp.shape[1] != asset_cfg.joint_mask.shape[0]:
        raise ValueError(
            f"joint_mask length ({asset_cfg.joint_mask.shape[0]}) does not match "
            f"joint_pos dim ({asset.data.joint_pos.warp.shape[1]}) for asset '{asset_cfg.name}'."
        )
    wp.launch(
        kernel=_joint_pos_out_of_manual_limit_kernel,
        dim=(env.num_envs, asset.data.joint_pos.warp.shape[1]),
        inputs=[asset.data.joint_pos.warp, asset_cfg.joint_mask, bounds[0], bounds[1], out],
        device=env.device,
    )


"""
Contact sensor.
"""


@wp.kernel
def _illegal_contact_kernel(
    forces: wp.array(dtype=wp.vec3f, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    """Terminate when any selected body's max-over-history contact force norm exceeds threshold."""
    i = wp.tid()
    violated = bool(False)
    for k in range(body_ids.shape[0]):
        b = body_ids[k]
        for h in range(forces.shape[1]):
            f = forces[i, h, b]
            norm = wp.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
            if norm > threshold:
                violated = True
    out[i] = violated


def illegal_contact(env: ManagerBasedRLEnv, out, threshold: float, sensor_cfg: SceneEntityCfg) -> None:
    """Terminate when the contact force on the sensor exceeds the force threshold. Writes into ``out``.

    Warp-first override of :func:`isaaclab.envs.mdp.terminations.illegal_contact`.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    wp.launch(
        kernel=_illegal_contact_kernel,
        dim=env.num_envs,
        inputs=[contact_sensor.data.net_forces_w_history.warp, sensor_cfg.body_ids_wp, threshold, out],
        device=env.device,
    )
