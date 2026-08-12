# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral actuator control interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAliasType

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_pd import ImplicitActuator

if TYPE_CHECKING:
    from .actuator_collection import ActuatorCollection

_WarpInt32 = TypeAliasType("_WarpInt32", wp.array(dtype=wp.int32))
_WarpInt64 = TypeAliasType("_WarpInt64", wp.array(dtype=wp.int64))
_WarpIndex = TypeAliasType("_WarpIndex", _WarpInt32 | _WarpInt64)


@dataclass(frozen=True)
class ActuatorJointProperties:
    """Default joint properties used to construct actuator models."""

    stiffness: torch.Tensor
    """Default joint stiffness values [N/m or N·m/rad, depending on joint type]."""

    damping: torch.Tensor
    """Default joint damping values [N·s/m or N·m·s/rad, depending on joint type]."""

    armature: torch.Tensor
    """Default joint armature values [kg or kg·m², depending on joint type]."""

    friction: torch.Tensor
    """Default backend-specific joint friction values.

    The physical meaning and units depend on the concrete backend and solver. See
    :attr:`isaaclab.assets.ArticulationData.joint_friction_coeff` for the active
    backend's convention.
    """

    dynamic_friction: torch.Tensor
    """Default backend-specific dynamic friction values.

    PhysX interprets these as dynamic friction efforts [N or N·m, depending on
    joint type]. OVPhysX interprets them as dimensionless Coulomb friction
    coefficients. Newton has no separate dynamic-friction property, so its
    control adapter supplies zeros.
    """

    viscous_friction: torch.Tensor
    """Default passive joint damping [N·s/m or N·m·s/rad, depending on joint type]."""

    effort_limit: torch.Tensor
    """Default joint effort limits [N or N·m, depending on joint type]."""

    velocity_limit: torch.Tensor
    """Default joint velocity limits [m/s or rad/s, depending on joint type]."""


class ActuatorControl(ABC):
    """Backend-neutral bridge used by :class:`~isaaclab.actuators.ActuatorCollection`."""

    @staticmethod
    def _normalize_index_sequence(
        indices: Sequence[int] | slice | torch.Tensor | _WarpIndex | None,
    ) -> list[int] | slice | torch.Tensor | _WarpIndex | None:
        """Convert non-list integer sequences to the backend's list convention."""
        if isinstance(indices, Sequence) and not isinstance(indices, list):
            return list(indices)
        return indices

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of articulation instances."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of articulation joints."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons."""
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        """Warp/Torch device string."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_pos(self) -> ProxyArray:
        """Current joint positions [m or rad, depending on joint type]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_vel(self) -> ProxyArray:
        """Current joint velocities [m/s or rad/s, depending on joint type]."""
        raise NotImplementedError

    @property
    def joint_stiffness(self) -> ProxyArray:
        """Current joint stiffness values [N/m or N·m/rad, depending on joint type]."""
        raise NotImplementedError(
            "ActuatorControl.joint_stiffness is required for Lab implicit actuator execution. "
            "The subclass must provide the current articulation-order joint stiffness as a ProxyArray."
        )

    @property
    def joint_damping(self) -> ProxyArray:
        """Current joint damping values [N·s/m or N·m·s/rad, depending on joint type]."""
        raise NotImplementedError(
            "ActuatorControl.joint_damping is required for Lab implicit actuator execution. "
            "The subclass must provide the current articulation-order joint damping as a ProxyArray."
        )

    @property
    def joint_effort_limits(self) -> ProxyArray:
        """Current joint effort limits [N or N·m, depending on joint type]."""
        raise NotImplementedError(
            "ActuatorControl.joint_effort_limits is required for Lab implicit actuator execution. "
            "The subclass must provide the current articulation-order joint effort limits as a ProxyArray."
        )

    @abstractmethod
    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int] | ProxyArray, list[str]]:
        """Resolve joint name expressions to user-order joint indices and names.

        Args:
            name_keys: Joint-name regular expressions.

        Returns:
            Resolved joint indices and names in user order.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_env_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor | _WarpIndex | None,
    ) -> torch.Tensor | _WarpIndex:
        """Resolve optional environment indices.

        Args:
            env_ids: Environment indices. Defaults to all environments.

        Returns:
            Device-local environment indices.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_joint_ids(
        self,
        joint_ids: Sequence[int] | torch.Tensor | _WarpIndex | None,
    ) -> torch.Tensor | _WarpIndex:
        """Resolve optional joint indices.

        Args:
            joint_ids: Joint indices. Defaults to all joints.

        Returns:
            Device-local joint indices.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_env_mask(self, env_mask: wp.array(dtype=wp.bool) | None) -> wp.array(dtype=wp.bool):
        """Resolve an optional environment mask.

        Args:
            env_mask: Environment selection mask. Defaults to all environments.

        Returns:
            Full environment selection mask.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_joint_mask(self, joint_mask: wp.array(dtype=wp.bool) | None) -> wp.array(dtype=wp.bool):
        """Resolve an optional joint mask.

        Args:
            joint_mask: Joint selection mask. Defaults to all joints.

        Returns:
            Full joint selection mask.
        """
        raise NotImplementedError

    @abstractmethod
    def assert_shape_and_dtype(
        self,
        tensor: torch.Tensor | wp.array(dtype=wp.float32) | float,
        shape: tuple[int, ...],
        dtype: type,
        name: str,
    ) -> None:
        """Validate tensor shape and dtype using the owning asset's policy.

        Args:
            tensor: Tensor or scalar to validate.
            shape: Required tensor shape.
            dtype: Required Warp dtype.
            name: Value name used in validation errors.
        """
        raise NotImplementedError

    @abstractmethod
    def assert_shape_and_dtype_mask(
        self,
        tensor: torch.Tensor | wp.array(dtype=wp.float32) | float,
        masks: tuple[wp.array(dtype=wp.bool), ...],
        dtype: type,
        name: str,
    ) -> None:
        """Validate a full-sized mask-write tensor.

        Args:
            tensor: Tensor or scalar to validate.
            masks: Selection masks that define the required shape.
            dtype: Required Warp dtype.
            name: Value name used in validation errors.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_joint_properties(self, joint_ids: torch.Tensor | _WarpIndex | slice) -> ActuatorJointProperties:
        """Return backend defaults used to construct one actuator group.

        Args:
            joint_ids: Articulation joints in the actuator group.

        Returns:
            Default properties for the selected joints.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_joint_properties(self, joint_ids: torch.Tensor | _WarpIndex | slice) -> ActuatorJointProperties:
        """Return current joint properties for deprecated group-level accessors.

        Args:
            joint_ids: Articulation joints in the actuator group.

        Returns:
            Current properties for the selected joints.
        """
        raise NotImplementedError

    @abstractmethod
    def write_resolved_joint_properties(
        self,
        properties: ActuatorJointProperties,
        joint_ids: torch.Tensor | _WarpIndex | slice,
        *,
        implicit: bool,
        native_managed: bool,
    ) -> None:
        """Write construction-resolved joint properties to the backend.

        Args:
            properties: Resolved joint properties for one configured group.
            joint_ids: Articulation joints in the configured group.
            implicit: Whether the group uses an implicit solver drive.
            native_managed: Whether the backend executes this group natively.
        """
        raise NotImplementedError

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection,
        env_ids: torch.Tensor | _WarpIndex | None,
        joint_ids: torch.Tensor | _WarpIndex | None,
        env_mask: wp.array(dtype=wp.bool) | None,
        joint_mask: wp.array(dtype=wp.bool) | None,
    ) -> None:
        """Stage a raw user command when the backend requires eager binding writes.

        Args:
            command_name: Command field to stage.
            collection: Collection that owns the command buffers.
            env_ids: Selected environment indices, or None for a mask write.
            joint_ids: Selected joint indices, or None for a mask write.
            env_mask: Selected environments, or None for an index write.
            joint_mask: Selected joints, or None for an index write.
        """

    @property
    def native_actuator_path_active(self) -> bool:
        """Whether backend handling replaces the Isaac Lab actuator loop."""
        return False

    def prepare_native_actuators(
        self, collection: ActuatorCollection, actuator_cfgs: dict[str, ActuatorBaseCfg]
    ) -> set[str]:
        """Prepare backend-native actuators.

        Args:
            collection: Collection being constructed.
            actuator_cfgs: Configured actuator groups.

        Returns:
            Names of groups managed by the backend.
        """
        return set()

    def finalize_native_actuators(self, collection: ActuatorCollection) -> None:
        """Finalize backend-native state after group construction.

        Args:
            collection: Fully constructed actuator collection.
        """

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        """Compute backend-native actuator outputs.

        Args:
            collection: Collection that owns actuator command and telemetry buffers.
            dt: Physics step size [s].

        Returns:
            True when native handling replaced the standard Python actuator loop.
        """
        return False

    @abstractmethod
    def submit_commands(self, collection: ActuatorCollection) -> None:
        """Submit processed command buffers to the backend.

        Args:
            collection: Collection that owns the processed commands.
        """
        raise NotImplementedError

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        """Reset backend-native actuator state.

        Args:
            env_ids: Environments to reset.
        """

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write backend-native actuator gains.

        Args:
            attr: Native gain name, either ``"kp"`` or ``"kd"``.
            values: Stiffness [N/m or N·m/rad] or damping [N·s/m or N·m·s/rad]
                values, shape ``(len(env_ids), len(joint_ids))``.
            env_ids: Environment indices to update.
            joint_ids: Articulation joint indices to update.
        """

    def get_native_actuator_gain(
        self,
        attr: Literal["kp", "kd"],
        joint_ids: torch.Tensor | slice,
    ) -> torch.Tensor | None:
        """Return controller-owned gains for one native actuator group.

        Args:
            attr: Controller gain name.
            joint_ids: Articulation joint indices in public order.

        Returns:
            Current controller gains, or ``None`` when the selected group is
            not completely managed by a supported native controller.
        """
        return None


class ArticulationActuatorControl(ActuatorControl):
    """Shared control adapter for articulation-backed actuator collections.

    This class implements the backend-independent forwarding and joint-property
    plumbing used by articulation backends. Backend subclasses only need to
    provide command submission and override the small hooks where their write
    APIs differ.

    Args:
        articulation: Articulation object that owns backend simulation handles.
    """

    def __init__(self, articulation):
        self._articulation = articulation
        self._native_actuator_path_active = False

    @property
    def native_actuator_path_active(self) -> bool:
        """Whether backend handling replaces the Isaac Lab actuator loop."""
        return self._native_actuator_path_active

    @property
    def num_instances(self) -> int:
        return self._articulation.num_instances

    @property
    def num_joints(self) -> int:
        return self._articulation.num_joints

    @property
    def num_fixed_tendons(self) -> int:
        return self._articulation.num_fixed_tendons

    @property
    def device(self) -> str:
        return self._articulation.device

    @property
    def joint_pos(self) -> ProxyArray:
        return self._articulation.data.joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._articulation.data.joint_vel

    @property
    def joint_stiffness(self) -> ProxyArray:
        return self._articulation.data.joint_stiffness

    @property
    def joint_damping(self) -> ProxyArray:
        return self._articulation.data.joint_damping

    @property
    def joint_effort_limits(self) -> ProxyArray:
        return self._articulation.data.joint_effort_limits

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[ProxyArray, list[str]]:
        return self._articulation.find_joints(name_keys, as_proxy=True)

    def resolve_env_ids(
        self,
        env_ids: Sequence[int] | torch.Tensor | _WarpIndex | None,
    ) -> torch.Tensor | _WarpIndex:
        return self._articulation._resolve_env_ids(self._normalize_index_sequence(env_ids))

    def resolve_joint_ids(
        self,
        joint_ids: Sequence[int] | torch.Tensor | _WarpIndex | None,
    ) -> torch.Tensor | _WarpIndex:
        return self._articulation._resolve_joint_ids(self._normalize_index_sequence(joint_ids))

    def resolve_env_mask(self, env_mask: wp.array(dtype=wp.bool) | None) -> wp.array(dtype=wp.bool):
        if hasattr(self._articulation, "_resolve_env_mask"):
            return self._articulation._resolve_env_mask(env_mask)
        return self._articulation._resolve_mask(env_mask, self._articulation._ALL_ENV_MASK)

    def resolve_joint_mask(self, joint_mask: wp.array(dtype=wp.bool) | None) -> wp.array(dtype=wp.bool):
        if hasattr(self._articulation, "_resolve_joint_mask"):
            return self._articulation._resolve_joint_mask(joint_mask)
        return self._articulation._resolve_mask(joint_mask, self._articulation._ALL_JOINT_MASK)

    def assert_shape_and_dtype(
        self,
        tensor: torch.Tensor | wp.array(dtype=wp.float32) | float,
        shape: tuple[int, ...],
        dtype: type,
        name: str,
    ) -> None:
        self._articulation.assert_shape_and_dtype(tensor, shape, dtype, name)

    def assert_shape_and_dtype_mask(
        self,
        tensor: torch.Tensor | wp.array(dtype=wp.float32) | float,
        masks: tuple[wp.array(dtype=wp.bool), ...],
        dtype: type,
        name: str,
    ) -> None:
        self._articulation.assert_shape_and_dtype_mask(tensor, masks, dtype, name)

    def get_default_joint_properties(self, joint_ids: torch.Tensor | _WarpIndex | slice) -> ActuatorJointProperties:
        current = self.get_current_joint_properties(joint_ids)
        return ActuatorJointProperties(
            stiffness=current.stiffness.clone(),
            damping=current.damping.clone(),
            armature=current.armature.clone(),
            friction=current.friction.clone(),
            dynamic_friction=current.dynamic_friction.clone(),
            viscous_friction=current.viscous_friction.clone(),
            effort_limit=current.effort_limit.clone(),
            velocity_limit=current.velocity_limit.clone(),
        )

    def get_current_joint_properties(self, joint_ids: torch.Tensor | _WarpIndex | slice) -> ActuatorJointProperties:
        joint_ids = self._as_torch_joint_ids(joint_ids)
        data = self._articulation.data
        stiffness = data.joint_stiffness.torch[:, joint_ids]
        return ActuatorJointProperties(
            stiffness=stiffness,
            damping=data.joint_damping.torch[:, joint_ids],
            armature=data.joint_armature.torch[:, joint_ids],
            friction=data.joint_friction_coeff.torch[:, joint_ids],
            dynamic_friction=self._joint_property_or_zeros("joint_dynamic_friction_coeff", joint_ids, stiffness),
            viscous_friction=self._joint_property_or_zeros("joint_viscous_friction_coeff", joint_ids, stiffness),
            effort_limit=data.joint_effort_limits.torch[:, joint_ids],
            velocity_limit=data.joint_vel_limits.torch[:, joint_ids],
        )

    def _as_torch_joint_ids(self, joint_ids: torch.Tensor | _WarpIndex | slice) -> torch.Tensor | slice:
        """Normalize an optional Warp joint selector for Torch property projections."""
        if isinstance(joint_ids, wp.array):
            return wp.to_torch(joint_ids).to(device=self.device, dtype=torch.long)
        return joint_ids

    def write_resolved_joint_properties(
        self,
        properties: ActuatorJointProperties,
        joint_ids: torch.Tensor | _WarpIndex | slice,
        *,
        implicit: bool,
        native_managed: bool,
    ) -> None:
        articulation = self._articulation
        articulation.write_joint_effort_limit_to_sim_index(
            limits=properties.effort_limit,
            joint_ids=joint_ids,
        )
        articulation.write_joint_velocity_limit_to_sim_index(
            limits=properties.velocity_limit,
            joint_ids=joint_ids,
        )
        articulation.write_joint_armature_to_sim_index(armature=properties.armature, joint_ids=joint_ids)
        self._write_joint_friction_properties(properties, joint_ids)
        if implicit and not native_managed:
            articulation.write_joint_stiffness_to_sim_index(stiffness=properties.stiffness, joint_ids=joint_ids)
            articulation.write_joint_damping_to_sim_index(damping=properties.damping, joint_ids=joint_ids)
        else:
            articulation.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=joint_ids)
            articulation.write_joint_damping_to_sim_index(damping=0.0, joint_ids=joint_ids)

    def _write_joint_friction_properties(
        self,
        properties: ActuatorJointProperties,
        joint_ids: torch.Tensor | _WarpIndex | slice,
    ) -> None:
        self._articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=properties.friction,
            joint_ids=joint_ids,
        )

    def _joint_property_or_zeros(
        self,
        attr_name: str,
        joint_ids: torch.Tensor | _WarpIndex | slice,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        joint_property = getattr(self._articulation.data, attr_name, None)
        if joint_property is None:
            return torch.zeros_like(reference)
        return joint_property.torch[:, joint_ids]

    @staticmethod
    def _is_implicit_cfg(actuator_cfg: ActuatorBaseCfg) -> bool:
        class_type = actuator_cfg.class_type
        return (
            "ImplicitActuator" in class_type
            if isinstance(class_type, str)
            else issubclass(class_type, ImplicitActuator)
        )
