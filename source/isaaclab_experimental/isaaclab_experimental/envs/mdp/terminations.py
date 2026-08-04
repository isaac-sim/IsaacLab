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
    position_threshold: float,
    orientation_threshold: float,
    out: wp.array(dtype=wp.bool),
):
    """Flag envs whose body pose is within every configured threshold. A negative threshold is unset."""
    i = wp.tid()
    success = bool(True)
    if position_threshold >= 0.0:
        des_b = wp.vec3f(cmd[i, 0], cmd[i, 1], cmd[i, 2])
        des_w = root_pos_w[i] + wp.quat_rotate(root_quat_w[i], des_b)
        cur_w = body_pos_w[i, body_idx]
        dx = cur_w[0] - des_w[0]
        dy = cur_w[1] - des_w[1]
        dz = cur_w[2] - des_w[2]
        if wp.sqrt(dx * dx + dy * dy + dz * dz) >= position_threshold:
            success = False
    if orientation_threshold >= 0.0:
        des_q_b = wp.quatf(cmd[i, 3], cmd[i, 4], cmd[i, 5], cmd[i, 6])
        des_q_w = root_quat_w[i] * des_q_b
        q_err = wp.quat_inverse(body_quat_w[i, body_idx]) * des_q_w
        if 2.0 * wp.acos(wp.clamp(wp.abs(q_err[3]), 0.0, 1.0)) >= orientation_threshold:
            success = False
    out[i] = success


class pose_command_success(ManagerTermBase):
    """Terminate environments whose pose command satisfies all configured success thresholds.

    Warp-first override of :func:`isaaclab.envs.mdp.terminations.pose_command_success`.

    The command term, its thresholds, the tracked body index and the warp view of the command
    buffer are all resolved at init, leaving :meth:`__call__` a single kernel launch.

    Note:
        The stable term also ORs the result into the command's per-episode success tracker. That
        tracker is already maintained by ``UniformPoseCommand._update_metrics`` on every step, so
        recomputing the thresholds here keeps ``Metrics/success_rate`` intact.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        command_name = cfg.params["command_name"]
        command = env.command_manager.get_term(command_name)
        # a negative threshold marks "not configured" for the kernel
        position_threshold = command.cfg.position_success_threshold
        orientation_threshold = command.cfg.orientation_success_threshold
        self._position_threshold = -1.0 if position_threshold is None else position_threshold
        self._orientation_threshold = -1.0 if orientation_threshold is None else orientation_threshold
        # matches the stable term: with no threshold configured, no env ever succeeds
        self._any_threshold = self._position_threshold >= 0.0 or self._orientation_threshold >= 0.0
        self._body_idx = command.body_idx
        self._asset: Articulation = command.robot
        cmd = env.command_manager.get_command(command_name)
        self._cmd_wp = cmd if isinstance(cmd, wp.array) else wp.from_torch(cmd)

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
                self._position_threshold,
                self._orientation_threshold,
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
