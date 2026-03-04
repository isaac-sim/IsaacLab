# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation implementation backed by ovphysx TensorBindingsAPI."""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import warp as wp

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.physics import PhysicsManager

from .articulation_data import ArticulationData

from isaaclab_ovphysx import tensor_types as TT

if TYPE_CHECKING:
    import ovphysx

    from isaaclab.actuators import ActuatorBase
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.utils.wrench_composer import WrenchComposer

logger = logging.getLogger(__name__)


def _get_ovphysx():
    """Lazy import to avoid USD version conflicts at module load time."""
    import ovphysx
    return ovphysx


@wp.kernel
def _body_wrench_to_world(
    force_b: wp.array(dtype=wp.vec3f, ndim=2),
    torque_b: wp.array(dtype=wp.vec3f, ndim=2),
    poses: wp.array(dtype=wp.transformf, ndim=2),
    wrench_out: wp.array(dtype=wp.float32, ndim=3),
):
    """Rotate body-frame force/torque to world frame and pack into [N, L, 9]."""
    i, j = wp.tid()
    q = wp.transform_get_rotation(poses[i, j])
    f_w = wp.quat_rotate(q, force_b[i, j])
    t_w = wp.quat_rotate(q, torque_b[i, j])
    wrench_out[i, j, 0] = f_w[0]
    wrench_out[i, j, 1] = f_w[1]
    wrench_out[i, j, 2] = f_w[2]
    wrench_out[i, j, 3] = t_w[0]
    wrench_out[i, j, 4] = t_w[1]
    wrench_out[i, j, 5] = t_w[2]
    p_w = wp.transform_get_translation(poses[i, j])
    wrench_out[i, j, 6] = p_w[0]
    wrench_out[i, j, 7] = p_w[1]
    wrench_out[i, j, 8] = p_w[2]


class Articulation(BaseArticulation):
    """Articulation backed by the ovphysx TensorBindingsAPI.

    Reads and writes simulation state through ovphysx.TensorBinding objects created
    from the OvPhysxManager's PhysX instance.
    """

    __backend_name__ = "ovphysx"

    cfg: ArticulationCfg

    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> ArticulationData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def is_fixed_base(self) -> bool:
        return self._is_fixed_base

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def num_fixed_tendons(self) -> int:
        return getattr(self, "_num_fixed_tendons", 0)

    @property
    def num_spatial_tendons(self) -> int:
        return getattr(self, "_num_spatial_tendons", 0)

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        return getattr(self, "_fixed_tendon_names", [])

    @property
    def spatial_tendon_names(self) -> list[str]:
        return getattr(self, "_spatial_tendon_names", [])

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def root_view(self) -> Any:
        return None

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer | None:
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer | None:
        return self._permanent_wrench_composer

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset articulation state to defaults for the given environments.

        Writes default root pose, root velocity, joint positions, and joint
        velocities back into the simulation for the specified env_ids (or all
        environments if env_ids is None).
        """
        if env_ids is not None:
            self.write_root_pose_to_sim_index(root_pose=self._data.default_root_pose, env_ids=env_ids)
            self.write_root_velocity_to_sim_index(root_velocity=self._data.default_root_vel, env_ids=env_ids)
            self.write_joint_position_to_sim_index(position=self._data.default_joint_pos, env_ids=env_ids)
            self.write_joint_velocity_to_sim_index(velocity=self._data.default_joint_vel, env_ids=env_ids)
        elif env_mask is not None:
            self.write_root_pose_to_sim_mask(root_pose=self._data.default_root_pose, env_mask=env_mask)
            self.write_root_velocity_to_sim_mask(root_velocity=self._data.default_root_vel, env_mask=env_mask)
            self.write_joint_position_to_sim_mask(position=self._data.default_joint_pos, env_mask=env_mask)
            self.write_joint_velocity_to_sim_mask(velocity=self._data.default_joint_vel, env_mask=env_mask)
        else:
            self.write_root_pose_to_sim_index(root_pose=self._data.default_root_pose)
            self.write_root_velocity_to_sim_index(root_velocity=self._data.default_root_vel)
            self.write_joint_position_to_sim_index(position=self._data.default_joint_pos)
            self.write_joint_velocity_to_sim_index(velocity=self._data.default_joint_vel)

        # Zero out command buffers.
        self._data._joint_pos_target.zero_()
        self._data._joint_vel_target.zero_()
        self._data._joint_effort_target.zero_()
        self._data._computed_torque.zero_()
        self._data._applied_torque.zero_()

    def write_data_to_sim(self) -> None:
        """Apply external wrenches, actuator model, and write commands into the simulation."""
        # Apply external wrenches (before actuators, same as PhysX backend).
        self._apply_external_wrenches()

        T = _get_ovphysx()
        self._apply_actuator_model()
        # Write implicit targets
        for act in self.actuators.values():
            if act.computed_effort is None:
                if act.joint_indices is not None:
                    self._write_joint_subset(
                        T.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_TARGET_F32,
                        self._data.joint_pos_target, act.joint_indices,
                    )
                    self._write_joint_subset(
                        T.OVPHYSX_TENSOR_ARTICULATION_DOF_VELOCITY_TARGET_F32,
                        self._data.joint_vel_target, act.joint_indices,
                    )

        effort_binding = self._bindings.get(T.OVPHYSX_TENSOR_ARTICULATION_DOF_ACTUATION_FORCE_F32)
        if effort_binding is not None:
            np_buf = self._data.applied_torque.numpy()
            effort_binding.write(np_buf)

    def update(self, dt: float) -> None:
        self._data.update(dt)

    # ------------------------------------------------------------------
    # Finders
    # ------------------------------------------------------------------

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return self._find_names(self._body_names, name_keys, preserve_order)

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: list[int] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        names = [self._joint_names[i] for i in joint_subset] if joint_subset is not None else self._joint_names
        indices, matched = self._find_names(names, name_keys, preserve_order)
        if joint_subset is not None:
            indices = [joint_subset[i] for i in indices]
        return indices, matched

    def find_fixed_tendons(self, name_keys, tendon_subsets=None, preserve_order=False):
        names = self.fixed_tendon_names
        if not names:
            return [], []
        return self._find_names(names, name_keys, preserve_order)

    def find_spatial_tendons(self, name_keys, tendon_subsets=None, preserve_order=False):
        names = self.spatial_tendon_names
        if not names:
            return [], []
        return self._find_names(names, name_keys, preserve_order)

    # ------------------------------------------------------------------
    # Root state writers (with shape validation)
    # ------------------------------------------------------------------

    def _n_envs_index(self, env_ids):
        if env_ids is None:
            return self._num_instances
        if isinstance(env_ids, (list, tuple)):
            return len(env_ids)
        return env_ids.shape[0] if hasattr(env_ids, "shape") else len(env_ids)

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")
        self._write_root_transform(10, root_pose, env_ids)

    def write_root_pose_to_sim_mask(self, *, root_pose, env_mask=None) -> None:
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        self._write_root_transform(10, root_pose, mask=env_mask)

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")
        self._write_root_transform(10, root_pose, env_ids)

    def write_root_link_pose_to_sim_mask(self, *, root_pose, env_mask=None) -> None:
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        self._write_root_transform(10, root_pose, mask=env_mask)

    def write_root_com_pose_to_sim_index(self, *, root_pose, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_pose, (n,), wp.transformf, "root_pose")
        self._write_root_transform(10, root_pose, env_ids)

    def write_root_com_pose_to_sim_mask(self, *, root_pose, env_mask=None) -> None:
        self.assert_shape_and_dtype(root_pose, (self._num_instances,), wp.transformf, "root_pose")
        self._write_root_transform(10, root_pose, mask=env_mask)

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")
        self._write_root_spatial(11, root_velocity, env_ids)

    def write_root_velocity_to_sim_mask(self, *, root_velocity, env_mask=None) -> None:
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        self._write_root_spatial(11, root_velocity, mask=env_mask)

    def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")
        self._write_root_spatial(11, root_velocity, env_ids)

    def write_root_com_velocity_to_sim_mask(self, *, root_velocity, env_mask=None) -> None:
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        self._write_root_spatial(11, root_velocity, mask=env_mask)

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        self.assert_shape_and_dtype(root_velocity, (n,), wp.spatial_vectorf, "root_velocity")
        self._write_root_spatial(11, root_velocity, env_ids)

    def write_root_link_velocity_to_sim_mask(self, *, root_velocity, env_mask=None) -> None:
        self.assert_shape_and_dtype(root_velocity, (self._num_instances,), wp.spatial_vectorf, "root_velocity")
        self._write_root_spatial(11, root_velocity, mask=env_mask)

    # ------------------------------------------------------------------
    # Joint state writers (with shape validation)
    # ------------------------------------------------------------------

    def write_joint_state_to_sim_mask(self, joint_pos, joint_vel, env_mask=None, joint_mask=None) -> None:
        self.write_joint_position_to_sim_mask(position=joint_pos, env_mask=env_mask, joint_mask=joint_mask)
        self.write_joint_velocity_to_sim_mask(velocity=joint_vel, env_mask=env_mask, joint_mask=joint_mask)

    def write_joint_position_to_sim_index(self, *, position, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(position, (n, d), wp.float32, "position")
        self._write_flat_tensor(30, position, env_ids, joint_ids)

    def write_joint_position_to_sim_mask(self, *, position, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(position, (self._num_instances, self._num_joints), wp.float32, "position")
        self._write_flat_tensor_mask(30, position, env_mask, joint_mask)

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(velocity, (n, d), wp.float32, "velocity")
        self._write_flat_tensor(31, velocity, env_ids, joint_ids)

    def write_joint_velocity_to_sim_mask(self, *, velocity, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(velocity, (self._num_instances, self._num_joints), wp.float32, "velocity")
        self._write_flat_tensor_mask(31, velocity, env_mask, joint_mask)

    # ------------------------------------------------------------------
    # Joint property writers (with shape validation)
    # ------------------------------------------------------------------

    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(stiffness, (n, d), wp.float32, "stiffness")
        self._write_flat_tensor(35, stiffness, env_ids, joint_ids)

    def write_joint_stiffness_to_sim_mask(self, *, stiffness, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._num_joints), wp.float32, "stiffness")
        self._write_flat_tensor_mask(35, stiffness, env_mask, joint_mask)

    def write_joint_damping_to_sim_index(self, *, damping, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(damping, (n, d), wp.float32, "damping")
        self._write_flat_tensor(36, damping, env_ids, joint_ids)

    def write_joint_damping_to_sim_mask(self, *, damping, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(damping, (self._num_instances, self._num_joints), wp.float32, "damping")
        self._write_flat_tensor_mask(36, damping, env_mask, joint_mask)

    def write_joint_position_limit_to_sim_index(
        self, *, limits, joint_ids=None, env_ids=None, warn_limit_violation=True
    ) -> None:
        if isinstance(limits, (int, float)):
            raise ValueError("Float scalars are not supported for position limits (vec2f dtype)")
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(limits, (n, d), wp.vec2f, "limits")
        self._write_flat_tensor(TT.DOF_LIMIT, limits, env_ids, joint_ids)

    def write_joint_position_limit_to_sim_mask(
        self, *, limits, joint_mask=None, env_mask=None, warn_limit_violation=True
    ) -> None:
        if isinstance(limits, (int, float)):
            raise ValueError("Float scalars are not supported for position limits (vec2f dtype)")
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.vec2f, "limits")
        self._write_flat_tensor_mask(TT.DOF_LIMIT, limits, env_mask, joint_mask)

    def write_joint_velocity_limit_to_sim_index(self, *, limits, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(limits, (n, d), wp.float32, "limits")
        self._write_flat_tensor(TT.DOF_MAX_VELOCITY, limits, env_ids, joint_ids)

    def write_joint_velocity_limit_to_sim_mask(self, *, limits, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.float32, "limits")
        self._write_flat_tensor_mask(TT.DOF_MAX_VELOCITY, limits, env_mask, joint_mask)

    def write_joint_effort_limit_to_sim_index(self, *, limits, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(limits, (n, d), wp.float32, "limits")
        self._write_flat_tensor(TT.DOF_MAX_FORCE, limits, env_ids, joint_ids)

    def write_joint_effort_limit_to_sim_mask(self, *, limits, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(limits, (self._num_instances, self._num_joints), wp.float32, "limits")
        self._write_flat_tensor_mask(TT.DOF_MAX_FORCE, limits, env_mask, joint_mask)

    def write_joint_armature_to_sim_index(self, *, armature, joint_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(armature, (n, d), wp.float32, "armature")
        self._write_flat_tensor(40, armature, env_ids, joint_ids)

    def write_joint_armature_to_sim_mask(self, *, armature, joint_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(armature, (self._num_instances, self._num_joints), wp.float32, "armature")
        self._write_flat_tensor_mask(40, armature, env_mask, joint_mask)

    def write_joint_friction_coefficient_to_sim_index(
        self, *, joint_friction_coeff, joint_ids=None, env_ids=None
    ) -> None:
        n = self._n_envs_index(env_ids)
        d = len(joint_ids) if joint_ids is not None else self._num_joints
        self.assert_shape_and_dtype(joint_friction_coeff, (n, d), wp.float32, "joint_friction_coeff")
        self._write_friction_column(joint_friction_coeff, env_ids, joint_ids)

    def write_joint_friction_coefficient_to_sim_mask(
        self, *, joint_friction_coeff, joint_mask=None, env_mask=None
    ) -> None:
        self.assert_shape_and_dtype(
            joint_friction_coeff, (self._num_instances, self._num_joints), wp.float32, "joint_friction_coeff"
        )
        self._write_friction_column_mask(joint_friction_coeff, env_mask, joint_mask)

    # ------------------------------------------------------------------
    # Deprecated combined-state writers (required by ABC)
    # ------------------------------------------------------------------

    def write_root_state_to_sim(self, root_state, env_ids=None) -> None:
        self.write_root_pose_to_sim_index(root_pose=root_state, env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state, env_ids=None) -> None:
        self.write_root_com_pose_to_sim_index(root_pose=root_state, env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state, env_ids=None) -> None:
        self.write_root_link_pose_to_sim_index(root_pose=root_state, env_ids=env_ids)

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None) -> None:
        self.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Body setters
    # ------------------------------------------------------------------

    def set_masses_index(self, *, masses, body_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        b = len(body_ids) if body_ids is not None else self._num_bodies
        self.assert_shape_and_dtype(masses, (n, b), wp.float32, "masses")
        self._write_flat_tensor(TT.BODY_MASS, masses, env_ids, body_ids)

    def set_masses_mask(self, *, masses, body_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(masses, (self._num_instances, self._num_bodies), wp.float32, "masses")
        self._write_flat_tensor_mask(TT.BODY_MASS, masses, env_mask, body_mask)

    def set_coms_index(self, *, coms, body_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        b = len(body_ids) if body_ids is not None else self._num_bodies
        self.assert_shape_and_dtype(coms, (n, b), wp.transformf, "coms")
        self._write_flat_tensor(TT.BODY_COM_POSE, coms, env_ids, body_ids)

    def set_coms_mask(self, *, coms, body_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(coms, (self._num_instances, self._num_bodies), wp.transformf, "coms")
        self._write_flat_tensor_mask(TT.BODY_COM_POSE, coms, env_mask, body_mask)

    def set_inertias_index(self, *, inertias, body_ids=None, env_ids=None) -> None:
        n = self._n_envs_index(env_ids)
        b = len(body_ids) if body_ids is not None else self._num_bodies
        self.assert_shape_and_dtype(inertias, (n, b, 9), wp.float32, "inertias")
        self._write_flat_tensor(TT.BODY_INERTIA, inertias, env_ids, body_ids)

    def set_inertias_mask(self, *, inertias, body_mask=None, env_mask=None) -> None:
        self.assert_shape_and_dtype(inertias, (self._num_instances, self._num_bodies, 9), wp.float32, "inertias")
        self._write_flat_tensor_mask(TT.BODY_INERTIA, inertias, env_mask, body_mask)

    # ------------------------------------------------------------------
    # Joint target setters
    # ------------------------------------------------------------------

    def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None) -> None:
        self._set_target_into_buffer(self._data._joint_pos_target, target, env_ids, joint_ids)

    def set_joint_position_target_mask(self, *, target, joint_mask=None, env_mask=None) -> None:
        self._set_target_into_buffer_mask(self._data._joint_pos_target, target, env_mask, joint_mask)

    def set_joint_velocity_target_index(self, *, target, joint_ids=None, env_ids=None) -> None:
        self._set_target_into_buffer(self._data._joint_vel_target, target, env_ids, joint_ids)

    def set_joint_velocity_target_mask(self, *, target, joint_mask=None, env_mask=None) -> None:
        self._set_target_into_buffer_mask(self._data._joint_vel_target, target, env_mask, joint_mask)

    def set_joint_effort_target_index(self, *, target, joint_ids=None, env_ids=None) -> None:
        self._set_target_into_buffer(self._data._joint_effort_target, target, env_ids, joint_ids)

    def set_joint_effort_target_mask(self, *, target, joint_mask=None, env_mask=None) -> None:
        self._set_target_into_buffer_mask(self._data._joint_effort_target, target, env_mask, joint_mask)

    # ------------------------------------------------------------------
    # Tendon operations (stubs)
    # ------------------------------------------------------------------

    def _nft(self): return getattr(self, "_num_fixed_tendons", 0)
    def _nst(self): return getattr(self, "_num_spatial_tendons", 0)

    def set_fixed_tendon_stiffness_index(self, *, stiffness, fixed_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(stiffness, (n, t), wp.float32, "stiffness")
        if self._data._fixed_tendon_stiffness is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_stiffness, stiffness, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_stiffness_mask(self, *, stiffness, fixed_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._nft()), wp.float32, "stiffness")
        if self._data._fixed_tendon_stiffness is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_stiffness, stiffness, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_damping_index(self, *, damping, fixed_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(damping, (n, t), wp.float32, "damping")
        if self._data._fixed_tendon_damping is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_damping, damping, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_damping_mask(self, *, damping, fixed_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(damping, (self._num_instances, self._nft()), wp.float32, "damping")
        if self._data._fixed_tendon_damping is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_damping, damping, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_limit_stiffness_index(self, *, limit_stiffness, fixed_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(limit_stiffness, (n, t), wp.float32, "limit_stiffness")
        if self._data._fixed_tendon_limit_stiffness is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_limit_stiffness, limit_stiffness, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_limit_stiffness_mask(self, *, limit_stiffness, fixed_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(limit_stiffness, (self._num_instances, self._nft()), wp.float32, "limit_stiffness")
        if self._data._fixed_tendon_limit_stiffness is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_limit_stiffness, limit_stiffness, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_position_limit_index(self, *, limit, fixed_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(limit, (n, t), wp.vec2f, "limit")
        if self._data._fixed_tendon_pos_limits is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_pos_limits, limit, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_position_limit_mask(self, *, limit, fixed_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(limit, (self._num_instances, self._nft()), wp.vec2f, "limit")
        if self._data._fixed_tendon_pos_limits is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_pos_limits, limit, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_rest_length_index(self, *, rest_length, fixed_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(rest_length, (n, t), wp.float32, "rest_length")
        if self._data._fixed_tendon_rest_length is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_rest_length, rest_length, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_rest_length_mask(self, *, rest_length, fixed_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(rest_length, (self._num_instances, self._nft()), wp.float32, "rest_length")
        if self._data._fixed_tendon_rest_length is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_rest_length, rest_length, env_mask, fixed_tendon_mask)

    def set_fixed_tendon_offset_index(self, *, offset, fixed_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(fixed_tendon_ids) if fixed_tendon_ids else self._nft()
        self.assert_shape_and_dtype(offset, (n, t), wp.float32, "offset")
        if self._data._fixed_tendon_offset is not None:
            self._set_target_into_buffer(self._data._fixed_tendon_offset, offset, env_ids, fixed_tendon_ids)

    def set_fixed_tendon_offset_mask(self, *, offset, fixed_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(offset, (self._num_instances, self._nft()), wp.float32, "offset")
        if self._data._fixed_tendon_offset is not None:
            self._set_target_into_buffer_mask(self._data._fixed_tendon_offset, offset, env_mask, fixed_tendon_mask)

    def write_fixed_tendon_properties_to_sim_index(self, *, fixed_tendon_ids=None, env_ids=None):
        if self._nft() == 0:
            return
        for tt, buf in [
            (TT.FIXED_TENDON_STIFFNESS, self._data._fixed_tendon_stiffness),
            (TT.FIXED_TENDON_DAMPING, self._data._fixed_tendon_damping),
            (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._data._fixed_tendon_limit_stiffness),
            (TT.FIXED_TENDON_LIMIT, self._data._fixed_tendon_pos_limits),
            (TT.FIXED_TENDON_REST_LENGTH, self._data._fixed_tendon_rest_length),
            (TT.FIXED_TENDON_OFFSET, self._data._fixed_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor(tt, buf, env_ids, fixed_tendon_ids)

    def write_fixed_tendon_properties_to_sim_mask(self, *, fixed_tendon_mask=None, env_mask=None):
        if self._nft() == 0:
            return
        for tt, buf in [
            (TT.FIXED_TENDON_STIFFNESS, self._data._fixed_tendon_stiffness),
            (TT.FIXED_TENDON_DAMPING, self._data._fixed_tendon_damping),
            (TT.FIXED_TENDON_LIMIT_STIFFNESS, self._data._fixed_tendon_limit_stiffness),
            (TT.FIXED_TENDON_LIMIT, self._data._fixed_tendon_pos_limits),
            (TT.FIXED_TENDON_REST_LENGTH, self._data._fixed_tendon_rest_length),
            (TT.FIXED_TENDON_OFFSET, self._data._fixed_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor_mask(tt, buf, env_mask, fixed_tendon_mask)

    def set_spatial_tendon_stiffness_index(self, *, stiffness, spatial_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(stiffness, (n, t), wp.float32, "stiffness")
        if self._data._spatial_tendon_stiffness is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_stiffness, stiffness, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_stiffness_mask(self, *, stiffness, spatial_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(stiffness, (self._num_instances, self._nst()), wp.float32, "stiffness")
        if self._data._spatial_tendon_stiffness is not None:
            self._set_target_into_buffer_mask(self._data._spatial_tendon_stiffness, stiffness, env_mask, spatial_tendon_mask)

    def set_spatial_tendon_damping_index(self, *, damping, spatial_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(damping, (n, t), wp.float32, "damping")
        if self._data._spatial_tendon_damping is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_damping, damping, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_damping_mask(self, *, damping, spatial_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(damping, (self._num_instances, self._nst()), wp.float32, "damping")
        if self._data._spatial_tendon_damping is not None:
            self._set_target_into_buffer_mask(self._data._spatial_tendon_damping, damping, env_mask, spatial_tendon_mask)

    def set_spatial_tendon_limit_stiffness_index(self, *, limit_stiffness, spatial_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(limit_stiffness, (n, t), wp.float32, "limit_stiffness")
        if self._data._spatial_tendon_limit_stiffness is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_limit_stiffness, limit_stiffness, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_limit_stiffness_mask(self, *, limit_stiffness, spatial_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(limit_stiffness, (self._num_instances, self._nst()), wp.float32, "limit_stiffness")
        if self._data._spatial_tendon_limit_stiffness is not None:
            self._set_target_into_buffer_mask(self._data._spatial_tendon_limit_stiffness, limit_stiffness, env_mask, spatial_tendon_mask)

    def set_spatial_tendon_offset_index(self, *, offset, spatial_tendon_ids=None, env_ids=None):
        n = self._n_envs_index(env_ids); t = len(spatial_tendon_ids) if spatial_tendon_ids else self._nst()
        self.assert_shape_and_dtype(offset, (n, t), wp.float32, "offset")
        if self._data._spatial_tendon_offset is not None:
            self._set_target_into_buffer(self._data._spatial_tendon_offset, offset, env_ids, spatial_tendon_ids)

    def set_spatial_tendon_offset_mask(self, *, offset, spatial_tendon_mask=None, env_mask=None):
        self.assert_shape_and_dtype(offset, (self._num_instances, self._nst()), wp.float32, "offset")
        if self._data._spatial_tendon_offset is not None:
            self._set_target_into_buffer_mask(self._data._spatial_tendon_offset, offset, env_mask, spatial_tendon_mask)

    def write_spatial_tendon_properties_to_sim_index(self, *, spatial_tendon_ids=None, env_ids=None):
        if self._nst() == 0:
            return
        for tt, buf in [
            (TT.SPATIAL_TENDON_STIFFNESS, self._data._spatial_tendon_stiffness),
            (TT.SPATIAL_TENDON_DAMPING, self._data._spatial_tendon_damping),
            (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._data._spatial_tendon_limit_stiffness),
            (TT.SPATIAL_TENDON_OFFSET, self._data._spatial_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor(tt, buf, env_ids, spatial_tendon_ids)

    def write_spatial_tendon_properties_to_sim_mask(self, *, spatial_tendon_mask=None, env_mask=None):
        if self._nst() == 0:
            return
        for tt, buf in [
            (TT.SPATIAL_TENDON_STIFFNESS, self._data._spatial_tendon_stiffness),
            (TT.SPATIAL_TENDON_DAMPING, self._data._spatial_tendon_damping),
            (TT.SPATIAL_TENDON_LIMIT_STIFFNESS, self._data._spatial_tendon_limit_stiffness),
            (TT.SPATIAL_TENDON_OFFSET, self._data._spatial_tendon_offset),
        ]:
            if buf is not None:
                self._write_flat_tensor_mask(tt, buf, env_mask, spatial_tendon_mask)

    # ------------------------------------------------------------------
    # Internal: initialization
    # ------------------------------------------------------------------

    def _initialize_impl(self) -> None:
        from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxManager

        self._ovphysx = _get_ovphysx()
        physx_instance = OvPhysxManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")

        prim_path = self.cfg.prim_path
        # Replace {ENV_REGEX_NS} with a glob-like wildcard for ovphysx pattern matching.
        pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", prim_path)

        # Bindings are created lazily (on first access) to avoid allocating
        # handles for tensor types the user never queries.  Only the root-pose
        # binding is created eagerly because we need it to read articulation
        # metadata (joint count, body count, names, fixed-base flag).
        T = self._ovphysx
        self._bindings: dict[int, Any] = {}
        self._physx_instance = physx_instance
        self._binding_pattern = pattern

        # Eagerly create only the bindings we need for metadata + initial
        # property reads.  Everything else is created on demand via
        # _get_binding().
        eager_types = [
            T.OVPHYSX_TENSOR_ARTICULATION_ROOT_POSE_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_POSITION_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_STIFFNESS_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_DAMPING_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_LIMIT_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_MAX_VELOCITY_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_MAX_FORCE_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_ARMATURE_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_DOF_FRICTION_PROPERTIES_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_BODY_MASS_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_BODY_COM_POSE_F32,
            T.OVPHYSX_TENSOR_ARTICULATION_BODY_INERTIA_F32,
        ]
        for tt in eager_types:
            try:
                self._bindings[tt] = physx_instance.create_tensor_binding(
                    pattern=pattern, tensor_type=tt
                )
            except Exception:
                logger.debug("Could not create tensor binding for type %s on pattern %s", tt, pattern)

        sample = next(iter(self._bindings.values()))
        self._num_instances = sample.count
        self._num_joints = sample.dof_count
        self._num_bodies = sample.body_count
        self._is_fixed_base = sample.is_fixed_base
        self._joint_names = list(sample.dof_names)
        self._body_names = list(sample.body_names)

        # Create data container.
        self._data = ArticulationData(self._bindings, self._device, binding_getter=self._get_binding)

        # Discover tendon counts/names before buffer allocation so that
        # _create_buffers can size the tendon property arrays.
        self._process_tendons()

        self._create_buffers()

        self._process_cfg()
        self._process_actuators_cfg()
        self._validate_cfg()
        self._log_articulation_info()

    def _create_buffers(self) -> None:
        self._data._create_buffers()

        from isaaclab.utils.wrench_composer import WrenchComposer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)
        self._wrench_buf = wp.zeros(
            (self._num_instances, self._num_bodies, 9), dtype=wp.float32, device=self._device
        )

        # Joint-index arrays for each actuator (filled by _process_actuators_cfg).
        self._joint_ids_per_actuator: dict[str, list[int]] = {}

    def _process_cfg(self) -> None:
        """Process the articulation configuration (initial state, soft limits, etc.)."""
        cfg = self.cfg
        N = self._num_instances
        D = self._num_joints
        dev = self._device

        # Default root state from config.
        pos = cfg.init_state.pos
        rot = cfg.init_state.rot
        for i in range(N):
            self._data._default_root_pose.numpy()[i] = [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]]

        # Default joint positions / velocities from config patterns.
        self._resolve_joint_values(cfg.init_state.joint_pos, self._data._default_joint_pos)
        self._resolve_joint_values(cfg.init_state.joint_vel, self._data._default_joint_vel)

        # Soft joint position limits.
        # Joints without explicit USD limits report +/-FLT_MAX.  Clamp to a
        # large but finite range to avoid overflow in the midpoint / half-range
        # computation.
        factor = cfg.soft_joint_pos_limit_factor
        _LIMIT_CLAMP = 1e6
        lim_np = self._data.joint_pos_limits.numpy().reshape(N, D, 2)
        lim_np = np.clip(lim_np, -_LIMIT_CLAMP, _LIMIT_CLAMP)
        mid = 0.5 * (lim_np[..., 0] + lim_np[..., 1])
        half = 0.5 * (lim_np[..., 1] - lim_np[..., 0])
        soft = np.stack([mid - factor * half, mid + factor * half], axis=-1)
        self._data._soft_joint_pos_limits = wp.from_numpy(soft, dtype=wp.vec2f, device=dev)

    def _invalidate_initialize_callback(self, event) -> None:
        self._is_initialized = False

    def _process_actuators_cfg(self) -> None:
        """Build actuator instances from the config."""
        from isaaclab.actuators import ActuatorBaseCfg

        self.actuators: dict[str, ActuatorBase] = {}
        for name, act_cfg in self.cfg.actuators.items():
            joint_ids, joint_names = self.find_joints(act_cfg.joint_names_expr)
            if not joint_ids:
                logger.warning("Actuator '%s': no joints matched '%s'", name, act_cfg.joint_names_expr)
                continue
            act_cfg_copy = act_cfg.copy()
            act = act_cfg_copy.class_type(
                act_cfg_copy,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self._num_instances,
                device=self._device,
            )
            self.actuators[name] = act
            self._joint_ids_per_actuator[name] = joint_ids

    def _process_tendons(self) -> None:
        """Discover tendon counts from binding metadata and names from USD.

        Tendon counts come from the ovphysx binding (fixed_tendon_count /
        spatial_tendon_count). Tendon names come from walking the exported
        USD stage and checking for PhysxTendon applied schemas on joints,
        following the same logic as the PhysX backend.
        """
        self._fixed_tendon_names = []
        self._spatial_tendon_names = []

        sample = next(iter(self._bindings.values()))
        self._num_fixed_tendons = getattr(sample, "fixed_tendon_count", 0)
        self._num_spatial_tendons = getattr(sample, "spatial_tendon_count", 0)

        if self._num_fixed_tendons > 0 or self._num_spatial_tendons > 0:
            from isaaclab_ovphysx.physics.ovphysx_manager import OvPhysxManager
            stage_path = OvPhysxManager._stage_path
            if stage_path is not None:
                try:
                    from pxr import Usd, UsdPhysics
                    stage = Usd.Stage.Open(stage_path)
                    for prim in stage.Traverse():
                        if not prim.HasAPI(UsdPhysics.Joint):
                            continue
                        schemas_str = str(prim.GetAppliedSchemas())
                        name = prim.GetPath().name
                        if "PhysxTendonAxisRootAPI" in schemas_str:
                            self._fixed_tendon_names.append(name)
                        elif "PhysxTendonAttachmentRootAPI" in schemas_str or "PhysxTendonAttachmentLeafAPI" in schemas_str:
                            self._spatial_tendon_names.append(name)
                except Exception:
                    logger.debug("Could not parse USD stage for tendon names at %s", stage_path)

        self._data._num_fixed_tendons = self._num_fixed_tendons
        self._data._num_spatial_tendons = self._num_spatial_tendons
        self._data.fixed_tendon_names = self._fixed_tendon_names
        self._data.spatial_tendon_names = self._spatial_tendon_names

    def _apply_external_wrenches(self) -> None:
        """Compose and write external wrenches to the LINK_WRENCH binding.

        WrenchComposer accumulates forces/torques in body (link) frame.
        The LINK_WRENCH binding expects world-frame [fx,fy,fz,tx,ty,tz,px,py,pz].
        We rotate the body-frame vectors to world frame using the link quaternion
        and pack them into the [N, L, 9] tensor with application position = origin.
        """
        inst = self._instantaneous_wrench_composer
        perm = self._permanent_wrench_composer
        if not inst.active and not perm.active:
            return
        if inst.active:
            inst.add_forces_and_torques_index(
                forces=perm.composed_force,
                torques=perm.composed_torque,
                body_ids=list(range(self._num_bodies)),
                env_ids=list(range(self._num_instances)),
            )
            force_b = inst.composed_force
            torque_b = inst.composed_torque
        else:
            force_b = perm.composed_force
            torque_b = perm.composed_torque

        poses = self._data.body_link_pose_w
        wp.launch(
            _body_wrench_to_world,
            dim=(self._num_instances, self._num_bodies),
            inputs=[force_b, torque_b, poses],
            outputs=[self._wrench_buf],
            device=self._device,
        )
        wrench_binding = self._get_binding(TT.LINK_WRENCH)
        if wrench_binding is not None:
            wrench_binding.write(self._wrench_buf)
        inst.reset()

    def _apply_actuator_model(self) -> None:
        """Run the actuator model to compute torques from user targets.

        IsaacLab actuators are torch-based.  We convert warp -> torch via
        DLPack (zero-copy on GPU), run the actuator, then write results back.
        """
        from isaaclab.utils.types import ArticulationActions

        device = self._device

        for name, act in self.actuators.items():
            jids = act.joint_indices
            if jids is None:
                continue
            jids_t = jids if isinstance(jids, list) else list(jids)

            # warp -> torch (zero-copy on same device via DLPack)
            jp_target = wp.to_torch(self._data.joint_pos_target)[:, jids_t]
            jv_target = wp.to_torch(self._data.joint_vel_target)[:, jids_t]
            je_target = wp.to_torch(self._data.joint_effort_target)[:, jids_t]

            control_action = ArticulationActions(
                joint_positions=jp_target,
                joint_velocities=jv_target,
                joint_efforts=je_target,
            )

            jp_cur = wp.to_torch(self._data.joint_pos)[:, jids_t]
            jv_cur = wp.to_torch(self._data.joint_vel)[:, jids_t]

            control_action = act.compute(control_action, jp_cur, jv_cur)

            if act.computed_effort is not None:
                ct = wp.to_torch(self._data._computed_torque)
                at = wp.to_torch(self._data._applied_torque)
                ct[:, jids_t] = act.computed_effort
                at[:, jids_t] = act.applied_effort

    def _validate_cfg(self) -> None:
        pass

    def _log_articulation_info(self) -> None:
        logger.info(
            "OvPhysX Articulation: instances=%d joints=%d bodies=%d fixed_base=%s",
            self._num_instances, self._num_joints, self._num_bodies, self._is_fixed_base,
        )

    # ------------------------------------------------------------------
    # Internal: lazy binding creation
    # ------------------------------------------------------------------

    def _get_binding(self, tensor_type: int):
        """Return a cached TensorBinding, creating it on first access.

        Bindings are lightweight handles (a pointer + shape metadata into
        PhysX's shared GPU buffer).  Creating one does NOT allocate new GPU
        memory -- the underlying simulation buffers are allocated once by PhysX
        regardless of how many bindings point into them.  Still, we defer
        creation so that tensor types the user never queries are never looked up.
        """
        binding = self._bindings.get(tensor_type)
        if binding is not None:
            return binding
        try:
            binding = self._physx_instance.create_tensor_binding(
                pattern=self._binding_pattern, tensor_type=tensor_type
            )
            self._bindings[tensor_type] = binding
            return binding
        except Exception:
            logger.debug("Could not create tensor binding for type %s", tensor_type)
            return None

    # ------------------------------------------------------------------
    # Internal: write helpers (GPU-native via DLPack)
    #
    # ovphysx binding.write() accepts any DLPack-compatible tensor (warp,
    # torch, numpy).  We keep data on the simulation device whenever
    # possible to avoid GPU->CPU->GPU copies.
    # ------------------------------------------------------------------

    def _to_flat_f32(self, data, target_shape: tuple[int, ...] | None = None) -> wp.array | np.ndarray:
        """Ensure data is a contiguous float32 tensor suitable for binding I/O.

        State tensor bindings (positions, velocities, poses) live on the
        simulation device (GPU in GPU mode).  We always return data on
        self._device so the binding device check passes.
        """
        dev = self._device
        if isinstance(data, wp.array):
            if str(data.device) != dev:
                data = wp.clone(data, device=dev)
            if data.dtype != wp.float32:
                data = wp.array(data.numpy(), dtype=wp.float32, device=dev)
            return data
        elif isinstance(data, torch.Tensor):
            np_data = data.detach().cpu().numpy().astype(np.float32)
            return wp.from_numpy(np_data, dtype=wp.float32, device=dev)
        elif isinstance(data, np.ndarray):
            return wp.from_numpy(data.astype(np.float32), dtype=wp.float32, device=dev)
        elif isinstance(data, (int, float)):
            return wp.from_numpy(np.array(data, dtype=np.float32), dtype=wp.float32, device=dev)
        return wp.from_numpy(np.asarray(data, dtype=np.float32), dtype=wp.float32, device=dev)

    def _write_root_transform(self, tensor_type: int, data, env_ids=None, mask=None) -> None:
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        flat = self._to_flat_f32(data)
        if env_ids is not None:
            idx = self._to_cpu_numpy(env_ids).astype(np.int32)
            binding.write(flat, indices=idx)
        elif mask is not None:
            binding.write(flat, mask=self._to_flat_f32(mask))
        else:
            binding.write(flat)

    def _write_root_spatial(self, tensor_type: int, data, env_ids=None, mask=None) -> None:
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        flat = self._to_flat_f32(data)
        if env_ids is not None:
            idx = self._to_cpu_numpy(env_ids).astype(np.int32)
            binding.write(flat, indices=idx)
        elif mask is not None:
            binding.write(flat, mask=self._to_flat_f32(mask))
        else:
            binding.write(flat)

    def _write_flat_tensor(self, tensor_type: int, data, env_ids=None, joint_ids=None) -> None:
        if isinstance(data, (int, float)):
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        np_data = self._to_cpu_numpy(data)
        if joint_ids is not None:
            # Subset of joints: read-modify-write to scatter into the correct columns.
            full = np.zeros(binding.shape, dtype=np.float32)
            binding.read(full)
            jids = np.asarray(joint_ids, dtype=np.intp)
            if env_ids is not None:
                eids = self._to_cpu_numpy(env_ids).astype(np.intp)
                full[np.ix_(eids, jids)] = np_data.reshape(len(eids), len(jids), *np_data.shape[2:])
            else:
                full[:, jids] = np_data.reshape(full.shape[0], len(jids), *np_data.shape[2:])
            binding.write(full)
        elif env_ids is not None:
            idx = self._to_cpu_numpy(env_ids).astype(np.int32)
            # When the caller passes a full-size buffer (e.g., tendon flush)
            # but only a subset of envs should be written, extract the rows.
            if np_data.shape[0] != len(idx) and np_data.shape[0] == binding.shape[0]:
                np_data = np_data[idx.astype(np.intp)]
            flat = wp.from_numpy(np_data.astype(np.float32), dtype=wp.float32, device=self._device)
            binding.write(flat, indices=idx)
        else:
            flat = self._to_flat_f32(data)
            binding.write(flat)

    def _write_flat_tensor_mask(self, tensor_type: int, data, env_mask=None, joint_mask=None) -> None:
        if isinstance(data, (int, float)):
            return
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        flat = self._to_flat_f32(data)
        if env_mask is not None:
            binding.write(flat, mask=self._to_flat_f32(env_mask))
        else:
            binding.write(flat)

    def _write_friction_column(self, data, env_ids=None, joint_ids=None) -> None:
        """Write static friction coefficient into column 0 of DOF_FRICTION_PROPERTIES [N,D,3]."""
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        if binding is None:
            return
        full = np.zeros(binding.shape, dtype=np.float32)
        binding.read(full)
        if isinstance(data, (int, float)):
            # Broadcast scalar to the targeted slice
            if env_ids is not None and joint_ids is not None:
                eids = self._to_cpu_numpy(env_ids).astype(np.intp)
                jids = np.asarray(joint_ids, dtype=np.intp)
                full[np.ix_(eids, jids, [0])] = data
            elif env_ids is not None:
                eids = self._to_cpu_numpy(env_ids).astype(np.intp)
                full[eids, :, 0] = data
            elif joint_ids is not None:
                jids = np.asarray(joint_ids, dtype=np.intp)
                full[:, jids, 0] = data
            else:
                full[..., 0] = data
            binding.write(full)
            return
        np_data = self._to_cpu_numpy(data)
        if env_ids is not None and joint_ids is not None:
            eids = self._to_cpu_numpy(env_ids).astype(np.intp)
            jids = np.asarray(joint_ids, dtype=np.intp)
            full[np.ix_(eids, jids, [0])] = np_data.reshape(len(eids), len(jids), 1)
        elif env_ids is not None:
            eids = self._to_cpu_numpy(env_ids).astype(np.intp)
            full[eids, :, 0] = np_data.reshape(len(eids), -1)
        elif joint_ids is not None:
            jids = np.asarray(joint_ids, dtype=np.intp)
            full[:, jids, 0] = np_data.reshape(full.shape[0], len(jids))
        else:
            full[..., 0] = np_data.reshape(full.shape[0], full.shape[1])
        binding.write(full)

    def _write_friction_column_mask(self, data, env_mask=None, joint_mask=None) -> None:
        """Write static friction coefficient via mask into column 0 of DOF_FRICTION_PROPERTIES."""
        binding = self._get_binding(TT.DOF_FRICTION_PROPERTIES)
        if binding is None:
            return
        full = np.zeros(binding.shape, dtype=np.float32)
        binding.read(full)
        if isinstance(data, (int, float)):
            new_col = np.full((full.shape[0], full.shape[1]), data, dtype=np.float32)
        else:
            new_col = self._to_cpu_numpy(data).reshape(full.shape[0], full.shape[1])
        if env_mask is not None:
            emask = self._to_cpu_numpy(env_mask).astype(bool)
            if joint_mask is not None:
                jmask = self._to_cpu_numpy(joint_mask).astype(bool)
                full[emask][:, jmask, 0] = new_col[emask][:, jmask]
            else:
                full[emask, :, 0] = new_col[emask]
        elif joint_mask is not None:
            jmask = self._to_cpu_numpy(joint_mask).astype(bool)
            full[:, jmask, 0] = new_col[:, jmask]
        else:
            full[..., 0] = new_col
        binding.write(full)

    def _write_joint_subset(self, tensor_type: int, buffer: wp.array, joint_ids: list[int]) -> None:
        """Write a full-width joint buffer into the simulation for an actuator's joints."""
        binding = self._get_binding(tensor_type)
        if binding is None:
            return
        binding.write(self._to_flat_f32(buffer))

    @staticmethod
    def _to_cpu_numpy(data) -> np.ndarray:
        """Convert data (warp, torch, numpy, scalar) to a CPU numpy array."""
        if isinstance(data, wp.array):
            return data.numpy().astype(np.float32)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    def _set_target_into_buffer(self, buffer: wp.array, data, env_ids=None, joint_ids=None) -> None:
        """Set user-provided target data into a warp command buffer.

        For the common case (no index subset), this uses wp.copy to stay on
        the simulation device.  Subset writes (specific env_ids or joint_ids)
        fall back to CPU because warp does not support scatter indexing.
        """
        if env_ids is None and joint_ids is None:
            src = self._to_flat_f32(data)
            if isinstance(src, np.ndarray):
                src = wp.from_numpy(src, dtype=wp.float32, device=buffer.device)
            wp.copy(buffer, src)
        else:
            np_data = self._to_cpu_numpy(data)
            buf_np = buffer.numpy()
            env_idx = self._to_cpu_numpy(env_ids).astype(np.intp) if env_ids is not None else None
            jnt_idx = self._to_cpu_numpy(joint_ids).astype(np.intp) if joint_ids is not None else None
            if env_idx is not None and jnt_idx is not None:
                buf_np[np.ix_(env_idx, jnt_idx)] = np_data
            elif env_idx is not None:
                buf_np[env_idx] = np_data
            else:
                buf_np[:, jnt_idx] = np_data
            wp.copy(buffer, wp.from_numpy(buf_np, dtype=wp.float32, device=buffer.device))

    def _set_target_into_buffer_mask(self, buffer: wp.array, data, env_mask=None, joint_mask=None) -> None:
        if env_mask is None:
            src = self._to_flat_f32(data)
            if isinstance(src, np.ndarray):
                src = wp.from_numpy(src, dtype=wp.float32, device=buffer.device)
            wp.copy(buffer, src)
        else:
            np_data = self._to_cpu_numpy(data)
            buf_np = buffer.numpy()
            mask_np = self._to_cpu_numpy(env_mask).astype(bool)
            buf_np[mask_np] = np_data[mask_np]
            wp.copy(buffer, wp.from_numpy(buf_np, dtype=wp.float32, device=buffer.device))

    # ------------------------------------------------------------------
    # Internal: pattern matching for joint/body lookup by name
    #
    # IsaacLab lets users specify joints and bodies by glob/regex patterns
    # like ".*_hip" or "joint_[0-3]" instead of numeric indices.  These
    # helpers convert those human-readable patterns into the integer index
    # lists that tensor bindings need.
    # ------------------------------------------------------------------

    @staticmethod
    def _find_names(
        names: list[str], keys: str | Sequence[str], preserve_order: bool
    ) -> tuple[list[int], list[str]]:
        if isinstance(keys, str):
            keys = [keys]
        matched_indices: list[int] = []
        matched_names: list[str] = []
        if preserve_order:
            for key in keys:
                for idx, name in enumerate(names):
                    if fnmatch.fnmatch(name, key) or re.fullmatch(key, name):
                        if idx not in matched_indices:
                            matched_indices.append(idx)
                            matched_names.append(name)
        else:
            for idx, name in enumerate(names):
                for key in keys:
                    if fnmatch.fnmatch(name, key) or re.fullmatch(key, name):
                        matched_indices.append(idx)
                        matched_names.append(name)
                        break
        return matched_indices, matched_names

    def _resolve_joint_values(self, pattern_dict: dict[str, float], buffer: wp.array) -> None:
        """Resolve a {pattern: value} dict into a per-joint buffer."""
        buf_np = buffer.numpy()
        for pattern, value in pattern_dict.items():
            for j, name in enumerate(self._joint_names):
                if re.fullmatch(pattern, name):
                    buf_np[:, j] = value
