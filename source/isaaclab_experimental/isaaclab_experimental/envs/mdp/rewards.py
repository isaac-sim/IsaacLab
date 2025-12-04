# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import warp as wp
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.managers.manager_base import ManagerTermBase
from isaaclab_experimental.managers.manager_term_cfg import RewardTermCfg
from isaaclab_experimental.utils.warp.utils import resolve_asset_cfg
from isaaclab.utils.warp.math_ops import inplace_add
from isaaclab.utils.warp.math_ops import inplace_mul

if TYPE_CHECKING:
    from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

"""
General.
"""

@wp.kernel
def bool_mask_to_float_kernel(
    bool_mask: wp.array(dtype=bool),
    float_mask: wp.array(dtype=wp.float32),
    invert: bool = False,
) -> None:
    env_index = wp.tid()
    if invert:
        float_mask[env_index] =  0.0 if bool_mask[env_index] else 1.0
    else:
        float_mask[env_index] =  1.0 if bool_mask[env_index] else 0.0

class is_alive(ManagerTermBase):
    """Reward for being alive."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        self._alive_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            bool_mask_to_float_kernel,
            dim=env.num_envs,
            inputs=[
                env.termination_manager.terminated,
                self._alive_buffer,
                True,
            ],
        )
        return self._alive_buffer

class is_terminated(ManagerTermBase):
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        self._terminated_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            bool_mask_to_float_kernel,
            dim=env.num_envs,
            inputs=[
                env.termination_manager.terminated,
                self._terminated_buffer,
                False,
            ],
        )
        return self._terminated_buffer


@wp.kernel
def is_terminated_term_kernel(
    terminated: wp.array(dtype=bool),
    reset_buf: wp.array(dtype=wp.float32)
) -> None:
    env_index = wp.tid()
    reset_buf[env_index] = terminated[env_index]


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)
        self._reset_buf = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)
        self._time_out_buf = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, term_keys: str | list[str] = ".*") -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        # Return the unweighted reward for the termination terms
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            inplace_add(self._reset_buf, env.termination_manager.get_term(term))

        wp.launch(
            bool_mask_to_float_kernel,
            dim=env.num_envs,
            inputs=[
                env.termination_manager.time_outs,
                self._time_out_buf,
                True,
            ],
        )
        inplace_mul(self._reset_buf, self._time_out_buf)
        return self._reset_buf


"""
Root penalties.
"""
@wp.kernel
def lin_vel_z_l2_kernel(
    root_com_vel_b: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    env_index = wp.tid()
    reward[env_index] = root_com_vel_b[env_index, 2] * root_com_vel_b[env_index, 2]

class lin_vel_z_l2(ManagerTermBase):
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._lin_vel_z_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            lin_vel_z_l2_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_com_vel_b,
                self._lin_vel_z_buffer,
            ],
        )
        return self._lin_vel_z_buffer

@wp.kernel
def ang_vel_xy_l2_kernel(
    root_com_vel_b: wp.array(dtype=wp.spatial_vectorf),
    reward: wp.array(dtype=wp.float32)
) -> None:
    env_index = wp.tid()
    reward[env_index] = root_com_vel_b[env_index, 3] * root_com_vel_b[env_index, 3] + root_com_vel_b[env_index, 4] * root_com_vel_b[env_index, 4]

class ang_vel_xy_l2(ManagerTermBase):
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._ang_vel_xy_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            ang_vel_xy_l2_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_com_vel_b,
                self._ang_vel_xy_buffer,
            ],
        )
        return self._ang_vel_xy_buffer

@wp.kernel
def flat_orientation_l2_kernel(
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    reward: wp.array(dtype=wp.float32)
) -> None:
    env_index = wp.tid()
    reward[env_index] = projected_gravity_b[env_index, 0] * projected_gravity_b[env_index, 0] + projected_gravity_b[env_index, 1] * projected_gravity_b[env_index, 1]

class flat_orientation_l2(ManagerTermBase):
    """Penalize non-flat base orientation using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._flat_orientation_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            flat_orientation_l2_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.projected_gravity_b,
                self._flat_orientation_buffer,
            ],
        )
        return self._flat_orientation_buffer

@wp.kernel
def base_height_l2_kernel(
    root_pose_w: wp.array(dtype=wp.transformf),
    target_height: float,
    reward: wp.array(dtype=wp.float32)
) -> None:
    env_index = wp.tid()
    reward[env_index] = (root_pose_w[env_index, 2] - target_height) * (root_pose_w[env_index, 2] - target_height)

class base_height_l2(ManagerTermBase):
    """Penalize asset height from its target using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._sensor_cfg = cfg.params.get("sensor_cfg", None)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._base_height_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

        self._target_height = 0.0
        self.update_config(**cfg.params)

    def update_config(
        self,
        target_height: float,
        asset_cfg: SceneEntityCfg | None = None,
        sensor_cfg: SceneEntityCfg | None = None,
    ) -> None:
        self._target_height = target_height
        

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        if self._sensor_cfg is not None: # noqa: R506
            raise NotImplementedError("Height scan is not implemented in IsaacLab for Newton.")
        else:
            wp.launch(
                base_height_l2_kernel,
                dim=env.num_envs,
                inputs=[
                    self._asset.data.root_pose_w,
                    self._target_height,
                    self._base_height_buffer,
                ],
            )
        return self._base_height_buffer

@wp.func
def compute_lin_acc_norm(acc: wp.spatial_vectorf) -> float:
    return wp.normalize(wp.spatial_top(acc))

@wp.func
def aggregate_body_acc_norm(acc: wp.array(dtype=wp.spatial_vectorf), body_indices: wp.array(dtype=wp.int32)) -> float:
    cum_norm = 0.0
    for i in body_indices:
        cum_norm += compute_lin_acc_norm(acc[i])
    return cum_norm

@wp.kernel
def body_lin_acc_l2_kernel(
    body_lin_acc_w: wp.array2d(dtype=wp.spatial_vectorf),
    body_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_body_acc_norm(body_lin_acc_w[i], body_indices)

class body_lin_acc_l2(ManagerTermBase):
    """Penalize the linear acceleration of bodies using L2-kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._body_indices = wp.array(asset_cfg.body_ids, dtype=wp.int32, device=env.device)
        self._body_lin_acc_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            body_lin_acc_l2_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.body_lin_acc_w,
                self._body_indices,
                self._body_lin_acc_buffer,
            ],
        )
        return self._body_lin_acc_buffer


"""
Joint penalties.
"""

@wp.func
def square(x: float) -> float:
    return x * x

@wp.func
def aggregate_joint_square(data: wp.array(dtype=wp.float32), joint_indices: wp.array(dtype=wp.int32)) -> float:
    cum_square = 0.0
    for i in joint_indices:
        cum_square += square(data[i])
    return cum_square

@wp.kernel
def joint_torques_l2_kernel(
    joint_torques: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_joint_square(joint_torques[i], joint_indices)

class joint_torques_l2(ManagerTermBase):
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._joint_torques_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            joint_torques_l2_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.applied_torque,
                self._joint_indices,
                self._joint_torques_buffer,
            ],
        )
        return self._joint_torques_buffer

@wp.func
def aggregate_joint_l1(data: wp.array(dtype=wp.float32), joint_indices: wp.array(dtype=wp.int32)) -> float:
    cum_l1 = 0.0
    for i in joint_indices:
        cum_l1 += wp.abs(data[i])
    return cum_l1

@wp.kernel
def joint_vel_l1_kernel(
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_joint_l1(joint_vel[i], joint_indices)

class joint_vel_l1(ManagerTermBase):
    """Penalize joint velocities on the articulation using an L1-kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._joint_vel_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            joint_vel_l1_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.joint_vel,
                self._joint_indices,
                self._joint_vel_buffer,
            ],
        )
        return self._joint_vel_buffer

@wp.kernel
def joint_vel_l2_kernel(
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_joint_square(joint_vel[i], joint_indices)

class joint_vel_l2(ManagerTermBase):
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._joint_vel_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            joint_vel_l2_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.joint_vel,
                self._joint_indices,
                self._joint_vel_buffer,
            ],
        )
        return self._joint_vel_buffer


@wp.kernel
def joint_deviation_l1_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_joint_l1(joint_pos[i] - default_joint_pos[i], joint_indices)

class joint_deviation_l1(ManagerTermBase):
    """Penalize joint positions that deviate from the default one using L1-kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._joint_deviation_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            joint_deviation_l1_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.joint_pos,
                self._asset.data.default_joint_pos,
                self._joint_indices,
                self._joint_deviation_buffer,
            ],
        )
        return self._joint_deviation_buffer

@wp.func
def out_of_limits_lower(x: float) -> float:
    if x > 0.0:
        return 0.0
    else:
        return x

@wp.func
def out_of_limits_upper(x: float) -> float:
    if x < 0.0:
        return 0.0
    else:
        return -x

@wp.func
def aggregate_out_of_limits(
    data: wp.array(dtype=wp.float32),
    limits: wp.array(dtype=wp.vec2f),
    indices: wp.array(dtype=wp.int32)
) -> float:
    cum_out_of_limits = 0.0
    for i in indices:
        cum_out_of_limits += out_of_limits_lower(data[i] - limits[i, 0])
        cum_out_of_limits += out_of_limits_upper(data[i] - limits[i, 1])
    return cum_out_of_limits

@wp.kernel
def joint_pos_limits_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_out_of_limits(
        joint_pos[i],
        soft_joint_pos_limits[i],
        joint_indices,
        reward[i],
    )

class joint_pos_limits(ManagerTermBase):
    """Penalize joint positions if they cross the soft limits."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._joint_pos_limits_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            joint_pos_limits_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.joint_pos,
                self._asset.data.soft_joint_pos_limits,
                self._joint_indices,
                self._joint_pos_limits_buffer,
            ],
        )
        return self._joint_pos_limits_buffer

@wp.func
def out_of_limits_unified(x: float, limit: float, soft_ratio: float) -> float:
    return wp.abs(x) - limit * soft_ratio

@wp.func
def aggregate_out_of_limits_unified(
    data: wp.array(dtype=wp.float32),
    limits: wp.array(dtype=wp.vec2f),
    soft_ratio: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32)
) -> float:
    cum_out_of_limits = 0.0
    for i in indices:
        cum_out_of_limits += out_of_limits_unified(data[i], limits[i, 0], soft_ratio[i])
    return cum_out_of_limits

@wp.kernel
def joint_pos_limits_unified_kernel(
    joint_pos: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    soft_ratio: wp.array(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_out_of_limits_unified(
        joint_pos[i],
        soft_joint_pos_limits[i],
        soft_ratio[i],
        joint_indices,
        reward[i],
    )

class joint_pos_limits_unified(ManagerTermBase):
    """Penalize joint positions if they cross the soft limits using unified kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._soft_ratio = 0.0
        self._joint_pos_limits_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

        self.update_config(**cfg.params)

    def update_config(self, soft_ratio: float, asset_cfg: SceneEntityCfg | None = None) -> None:
        self._soft_ratio = soft_ratio

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            joint_pos_limits_unified_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.joint_pos,
                self._asset.data.soft_joint_pos_limits,
                self._soft_ratio,
                self._joint_indices,
                self._joint_pos_limits_buffer,
            ],
        )
        return self._joint_pos_limits_buffer


"""
Action penalties.
"""

@wp.func
def aggregate_applied_torque_limits(
    applied_torque: wp.array(dtype=wp.float32),
    computed_torque: wp.array(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32)
) -> float:
    cum_applied_torque_limits = 0.0
    for i in joint_indices:
        cum_applied_torque_limits += wp.abs(applied_torque[i] - computed_torque[i])
    return cum_applied_torque_limits

@wp.kernel
def applied_torque_limits_kernel(
    applied_torque: wp.array2d(dtype=wp.float32),
    computed_torque: wp.array2d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_applied_torque_limits(applied_torque[i], computed_torque[i], joint_indices)

class applied_torque_limits(ManagerTermBase):
    """Penalize applied torques if they cross the limits."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_indices = wp.array(asset_cfg.joint_ids, dtype=wp.int32, device=env.device)
        self._applied_torque_limits_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, asset_cfg: SceneEntityCfg | None = None) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            applied_torque_limits_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.applied_torque,
                self._asset.data.computed_torque,
                self._joint_indices,
                self._applied_torque_limits_buffer,
            ],
        )
        return self._applied_torque_limits_buffer

@wp.func
def aggregate_action_rate_l2(
    action: wp.array(dtype=wp.float32),
    prev_action: wp.array(dtype=wp.float32),
) -> float:
    cum_action_rate_l2 = 0.0
    for i in range(action.shape[0]):
        cum_action_rate_l2 += square(action[i] - prev_action[i])
    return cum_action_rate_l2

@wp.kernel
def action_rate_l2_kernel(
    action: wp.array2d(dtype=wp.float32),
    prev_action: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_action_rate_l2(action[i], prev_action[i])

class action_rate_l2(ManagerTermBase):
    """Penalize the rate of change of the actions using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        self._action_rate_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, **kwargs) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            action_rate_l2_kernel,
            dim=env.num_envs,
            inputs=[
                env.action_manager.action,
                env.action_manager.prev_action,
                self._action_rate_buffer,
            ],
        )
        return self._action_rate_buffer

@wp.func
def aggregate_action_l2(
    action: wp.array(dtype=wp.float32),
) -> float:
    cum_action_l2 = 0.0
    for i in range(action.shape[0]):
        cum_action_l2 += square(action[i])
    return cum_action_l2

@wp.kernel
def action_l2_kernel(
    action: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_action_l2(action[i])

class action_l2(ManagerTermBase):
    """Penalize the actions using L2 squared kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        self._action_l2_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

    def update_config(self, **kwargs) -> None:
        pass

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            action_l2_kernel,
            dim=env.num_envs,
            inputs=[
                env.action_manager.action,
                self._action_l2_buffer,
            ],
        )
        return self._action_l2_buffer


"""
Contact sensor.
"""


#def undesired_contacts(env: ManagerBasedRLEnvWarp, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#    """Penalize undesired contacts as the number of violations that are above a threshold."""
#    # extract the used quantities (to enable type-hinting)
#    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#    # check if contact force is above threshold
#    net_contact_forces = contact_sensor.data.net_forces_w_history
#    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
#    # sum over contacts for each environment
#    return torch.sum(is_contact, dim=1)
#
#
#def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
#    """Penalize if none of the desired contacts are present."""
#    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#    contacts = (
#        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
#    )
#    zero_contact = (~contacts).all(dim=1)
#    return 1.0 * zero_contact
#
#
#def contact_forces(env: ManagerBasedRLEnvWarp, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#    """Penalize contact forces as the amount of violations of the net contact force."""
#    # extract the used quantities (to enable type-hinting)
#    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#    net_contact_forces = contact_sensor.data.net_forces_w_history
#    # compute the violation
#    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
#    # compute the penalty
#    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""

@wp.func
def negative_exp(x: float, std: float) -> float:
    return wp.exp(-x / (std*std))

@wp.func
def aggregate_track_lin_vel_xy(
    vel_b: wp.array(dtype=wp.spatial_vectorf),
    command: wp.array2d(dtype=wp.float32),
    std: float,
) -> float:
    lin_vel = square(vel_b[0] - command[0]) + square(vel_b[1] - command[1])
    return negative_exp(lin_vel, std)

@wp.kernel
def track_lin_vel_xy_exp_kernel(
    vel_b: wp.array(dtype=wp.spatial_vectorf),
    command: wp.array3d(dtype=wp.float32),
    std: float,
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_track_lin_vel_xy(vel_b[i], command[i], std)

class track_lin_vel_xy_exp(ManagerTermBase):
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._std = 1.0
        self._command_name = "None"
        self._track_lin_vel_xy_exp_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

        self.update_config(**cfg.params)

    def update_config(self, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
        self._std = std
        self._command_name = command_name

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            track_lin_vel_xy_exp_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_com_vel_b,
                env.command_manager.get_command(self._command_name),
                self._std,
                self._track_lin_vel_xy_exp_buffer,
            ],
        )
        return self._track_lin_vel_xy_exp_buffer

@wp.func
def aggregate_track_ang_vel_z(
    ang_vel_b: wp.array(dtype=wp.spatial_vectorf),
    command: wp.array2d(dtype=wp.float32),
    std: float,
) -> float:
    ang_vel_error = square(ang_vel_b[2] - command[2])
    return negative_exp(ang_vel_error, std)

@wp.kernel
def track_ang_vel_z_exp_kernel(
    ang_vel_b: wp.array(dtype=wp.spatial_vectorf),
    command: wp.array3d(dtype=wp.float32),
    std: float,
    reward: wp.array(dtype=wp.float32)
) -> None:
    i = wp.tid()
    reward[i] = aggregate_track_ang_vel_z(ang_vel_b[i], command[i], std)

class track_ang_vel_z_exp(ManagerTermBase):
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnvWarp):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._std = 1.0
        self._command_name = "None"
        self._track_ang_vel_z_exp_buffer = wp.zeros((env.num_envs,), dtype=wp.float32, device=env.device)

        self.update_config(**cfg.params)

    def update_config(self, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
        self._std = std
        self._command_name = command_name

    def __call__(self, env: ManagerBasedRLEnvWarp, **kwargs) -> wp.array(dtype=wp.float32):
        wp.launch(
            track_ang_vel_z_exp_kernel,
            dim=env.num_envs,
            inputs=[
                self._asset.data.root_ang_vel_b,
                env.command_manager.get_command(self._command_name),
                self._std,
                self._track_ang_vel_z_exp_buffer,
            ],
        )
        return self._track_ang_vel_z_exp_buffer