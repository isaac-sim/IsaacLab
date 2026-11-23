# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A virtual articulation: joint state and joint targets of the world-model backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from isaaclab.utils.warp import ProxyArray

from .articulation_data import UNSUPPORTED, ArticulationData

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg
    from isaaclab.utils.wrench_composer import WrenchComposer

    from ...physics import NeuralManager


def _usd_joints(prim) -> tuple[list[str], list[tuple[float, float]]]:
    """Names and position limits of the movable joints below ``prim``, in stage traversal order.

    Revolute limits are authored in degrees and returned in radians; prismatic limits are in meters. A limit that is
    not authored is unbounded, as in USD.
    """
    from pxr import Usd, UsdPhysics

    names, limits = [], []
    for p in Usd.PrimRange(prim):
        for joint_type, scale in ((UsdPhysics.RevoluteJoint, math.pi / 180.0), (UsdPhysics.PrismaticJoint, 1.0)):
            if p.IsA(joint_type):
                joint = joint_type(p)
                lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
                names.append(p.GetName())
                limits.append(
                    (
                        -math.inf if lower is None or not math.isfinite(lower) else lower * scale,
                        math.inf if upper is None or not math.isfinite(upper) else upper * scale,
                    )
                )
    return names, limits


class Articulation(BaseArticulation):
    """Joint state and joint targets of a robot that the world model imagines.

    Observation terms read :attr:`data.joint_pos <ArticulationData.joint_pos>` and action terms write joint position
    targets, as they do for a simulated robot, so the same task configuration runs on a physics backend or on the
    world model. The joint positions are the manager's: the reset record's state at reset, then the last commanded
    target. Everything that needs a simulated body (poses, velocities, forces, physical properties) raises.

    :meth:`reset` starts each episode from a reset record and makes the record's state the default joint position, so
    the standard reset events (``reset_joints_by_offset``, ``reset_joints_by_scale``) start from it as well.

    Joints and their limits come from the spawned USD asset, as for a physics backend: its movable joints in
    traversal order, or in the order of ``ArticulationCfg.joint_ordering``, given as a list of every joint name. The
    order must match the reset records.
    """

    __backend_name__ = "neural"
    cfg: ArticulationCfg

    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)
        self._data: ArticulationData | None = None
        self._joint_names: list[str] = []
        self._num_instances = 0

    # ------------------------------------------------------------- properties

    @property
    def data(self) -> ArticulationData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def num_bodies(self) -> int:
        return 0

    @property
    def num_fixed_tendons(self) -> int:
        return 0

    @property
    def num_spatial_tendons(self) -> int:
        return 0

    @property
    def fixed_tendon_names(self) -> list[str]:
        return []

    @property
    def spatial_tendon_names(self) -> list[str]:
        return []

    @property
    def backend_joint_names(self) -> list[str]:
        return list(self._joint_names)

    # --------------------------------------------------------------- lifecycle

    def _initialize_impl(self) -> None:
        manager: NeuralManager = self._backend_manager
        prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        self._num_instances = len(prims)
        usd_names, usd_limits = _usd_joints(prims[0])
        # The joints are virtual, so they are stored directly in the configured public order.
        self._joint_names = usd_names
        names, _ = self._resolve_axis_ordering("joint")
        limits = [usd_limits[usd_names.index(name)] for name in names]
        self._joint_names = names
        manager.bind(self._num_instances, len(names))
        self._reset_mask = torch.zeros(self._num_instances, dtype=torch.bool, device=self._device)

        default = torch.zeros((self._num_instances, len(names)), device=self._device)
        if self.cfg.init_state.joint_pos:
            ids, _, values = resolve_matching_names_values(self.cfg.init_state.joint_pos, names)
            default[:, ids] = torch.tensor(values, dtype=torch.float32, device=self._device)
        manager.joint_state.copy_(default)
        manager.joint_target.copy_(default)
        self._data = ArticulationData(
            self._device,
            names,
            manager.joint_state,
            manager.joint_target,
            default,
            torch.tensor(limits, dtype=torch.float32, device=self._device)[None],
            self.cfg.soft_joint_pos_limit_factor,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        super()._invalidate_initialize_callback(event)
        self._data = None

    def reset(self, env_ids=None, env_mask=None) -> None:
        """Start new episodes from reset records and make each record's state the default joint position."""
        manager: NeuralManager = self._backend_manager
        if env_mask is not None:
            selected = wp.to_torch(env_mask).bool()
        else:
            selected = self._reset_mask
            selected.zero_()
            selected.index_fill_(0, self._indices(env_ids, self._num_instances), True)
        manager.reset_envs(selected)
        default = self._data.default_joint_pos.torch
        torch.where(selected[:, None], manager.joint_state, default, out=default)

    def write_data_to_sim(self) -> None:
        """No-op: the joint targets are the manager's buffer, which it applies on the next step."""

    def update(self, dt: float) -> None:
        """No-op: the joint positions are the manager's buffer."""

    # ----------------------------------------------------------------- finders

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: list[str] | None = None,
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ):
        names = self._joint_names if joint_subset is None else joint_subset
        _, matched = resolve_matching_names(name_keys, names, preserve_order)
        ids = [self._joint_names.index(name) for name in matched]
        if as_proxy:
            return ProxyArray(wp.array(ids, dtype=wp.int32, device=self._device)), matched
        return ids, matched

    # ----------------------------------------------------------- joint targets

    def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None, full_data=False) -> None:
        """Set joint position targets. They apply on the next :meth:`write_data_to_sim`."""
        self._set(self._data.joint_pos_target.torch, target, joint_ids, env_ids, full_data)

    def set_joint_position_target_mask(self, *, target, joint_mask=None, env_mask=None) -> None:
        self._set_mask(self._data.joint_pos_target.torch, target, joint_mask, env_mask)

    def set_joint_velocity_target_index(self, **kwargs) -> None:
        raise NotImplementedError("The world model takes joint position targets only")

    set_joint_velocity_target_mask = set_joint_effort_target_index = set_joint_effort_target_mask = (
        set_joint_velocity_target_index
    )

    # ------------------------------------------------------------ state writes

    def write_joint_position_to_sim_index(
        self, *, position, joint_ids=None, env_ids=None, skip_forward: bool = False
    ) -> None:
        """Set joint positions, including the state the next chunk starts from."""
        self._set(self._data.joint_pos.torch, position, joint_ids, env_ids, False)

    def write_joint_position_to_sim_mask(self, *, position, joint_mask=None, env_mask=None, skip_forward=False) -> None:
        self._set_mask(self._data.joint_pos.torch, position, joint_mask, env_mask)

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids=None, env_ids=None, skip_forward=False) -> None:
        """Ignored: the world model has no joint velocity."""

    def write_joint_velocity_to_sim_mask(self, *, velocity, joint_mask=None, env_mask=None, skip_forward=False) -> None:
        """Ignored: the world model has no joint velocity."""

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None) -> None:
        """Set joint positions; the velocity is ignored because the world model has none."""
        self.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)

    # ----------------------------------------------------------------- helpers

    def _indices(self, ids, size: int) -> torch.Tensor:
        if ids is None or isinstance(ids, slice):
            return torch.arange(size, device=self._device)[ids if isinstance(ids, slice) else slice(None)]
        if isinstance(ids, ProxyArray):
            ids = ids.torch
        elif isinstance(ids, wp.array):
            ids = wp.to_torch(ids)
        return torch.as_tensor(ids, dtype=torch.long, device=self._device).flatten()

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        if isinstance(value, ProxyArray):
            return value.torch
        return wp.to_torch(value) if isinstance(value, wp.array) else torch.as_tensor(value)

    def _set(self, buffer: torch.Tensor, value, joint_ids, env_ids, full_data: bool) -> None:
        value = self._tensor(value)
        value = value.to(device=buffer.device, dtype=buffer.dtype)
        envs, joints = self._indices(env_ids, buffer.shape[0]), self._indices(joint_ids, buffer.shape[1])
        if full_data:
            value = value[envs][:, joints]
        buffer[envs[:, None], joints[None, :]] = value.reshape(len(envs), len(joints))

    def _set_mask(self, buffer: torch.Tensor, value, joint_mask, env_mask) -> None:
        """Write the full-shape ``value`` where both masks are set, without reading the masks on the host."""
        value = self._tensor(value).to(device=buffer.device, dtype=buffer.dtype)
        selected = torch.ones(buffer.shape, dtype=torch.bool, device=buffer.device)
        if env_mask is not None:
            selected &= wp.to_torch(env_mask).bool()[:, None]
        if joint_mask is not None:
            selected &= wp.to_torch(joint_mask).bool()[None, :]
        torch.where(selected, value, buffer, out=buffer)

    # Private hooks of the base class that have nothing to do here.
    def _create_buffers(self) -> None:
        pass

    def _log_articulation_info(self) -> None:
        pass

    def _process_actuators_cfg(self) -> None:
        pass

    def _process_cfg(self) -> None:
        pass

    def _process_tendons(self) -> None:
        pass

    def _validate_cfg(self) -> None:
        pass

    # The world model has joint state only: these parts of the articulation are not available.

    @property
    def is_fixed_base(self) -> bool:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def root_view(self):
        raise NotImplementedError(UNSUPPORTED)

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        raise NotImplementedError(UNSUPPORTED)

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        raise NotImplementedError(UNSUPPORTED)

    def find_bodies(
        self,
        name_keys: str | Sequence[str],
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        raise NotImplementedError(UNSUPPORTED)

    def find_fixed_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        raise NotImplementedError(UNSUPPORTED)

    def find_spatial_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subsets: list[str] | None = None,
        preserve_order: bool = False,
        *,
        as_proxy: bool = False,
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_link_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_link_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_com_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_com_pose_to_sim_mask(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_com_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_com_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_link_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_link_velocity_to_sim_mask(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_mask: wp.array | None = None,
        skip_forward: bool = False,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_stiffness_to_sim_index(
        self,
        *,
        stiffness: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_stiffness_to_sim_mask(
        self,
        *,
        stiffness: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_damping_to_sim_index(
        self,
        *,
        damping: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_damping_to_sim_mask(
        self,
        *,
        damping: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_position_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_position_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
        warn_limit_violation: bool = True,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_velocity_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_velocity_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_effort_limit_to_sim_index(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_effort_limit_to_sim_mask(
        self,
        *,
        limits: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_armature_to_sim_index(
        self,
        *,
        armature: torch.Tensor | float | wp.array,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_armature_to_sim_mask(
        self,
        *,
        armature: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_friction_coefficient_to_sim_index(
        self,
        *,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_dynamic_friction_coeff: torch.Tensor | float | wp.array | None = None,
        joint_viscous_friction_coeff: torch.Tensor | float | wp.array | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_joint_friction_coefficient_to_sim_mask(
        self,
        *,
        joint_friction_coeff: torch.Tensor | float | wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_masses_index(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_masses_mask(
        self,
        *,
        masses: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_coms_index(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_coms_mask(
        self,
        *,
        coms: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_inertias_index(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_inertias_mask(
        self,
        *,
        inertias: torch.Tensor | wp.array,
        body_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_position_limit_index(
        self,
        *,
        limit: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_position_limit_mask(
        self,
        *,
        limit: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_rest_length_index(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_rest_length_mask(
        self,
        *,
        rest_length: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_position_target_index(
        self,
        *,
        target: torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_position_target_mask(
        self,
        *,
        target: torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_fixed_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_fixed_tendon_properties_to_sim_index(
        self,
        *,
        fixed_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_fixed_tendon_properties_to_sim_mask(
        self,
        *,
        fixed_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_stiffness_index(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_stiffness_mask(
        self,
        *,
        stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_damping_index(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_damping_mask(
        self,
        *,
        damping: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_limit_stiffness_index(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_limit_stiffness_mask(
        self,
        *,
        limit_stiffness: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_offset_index(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def set_spatial_tendon_offset_mask(
        self,
        *,
        offset: float | torch.Tensor | wp.array,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_spatial_tendon_properties_to_sim_index(
        self,
        *,
        spatial_tendon_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_spatial_tendon_properties_to_sim_mask(
        self,
        *,
        spatial_tendon_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_com_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)

    def write_root_link_state_to_sim(
        self,
        root_state: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        raise NotImplementedError(UNSUPPORTED)
