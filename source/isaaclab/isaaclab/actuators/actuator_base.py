# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg
    from .actuator_control import ActuatorControl


class ActuatorBase(ABC):
    """Base class for actuator models over a collection of actuated joints in an articulation.

    Actuator models augment the simulated articulation joints with an external drive dynamics model.
    The model is used to convert the user-provided joint commands (positions, velocities and efforts)
    into the desired joint positions, velocities and efforts that are applied to the simulated articulation.

    The base class provides the interface for the actuator models. It is responsible for parsing the
    actuator parameters from the configuration and storing them as buffers. It also provides the
    interface for resetting the actuator state and computing the desired joint commands for the simulation.

    For each actuator model, a corresponding configuration class is provided. The configuration class
    is used to parse the actuator parameters from the configuration. It also specifies the joint names
    for which the actuator model is applied. These names can be specified as regular expressions, which
    are matched against the joint names in the articulation.

    To see how the class is used, check the :class:`isaaclab.assets.Articulation` class.
    """

    is_implicit_model: ClassVar[bool] = False
    """Flag indicating if the actuator is an implicit or explicit actuator model.

    If a class inherits from :class:`ImplicitActuator`, then this flag should be set to :obj:`True`.
    """

    computed_effort: torch.Tensor
    """The computed effort for the actuator group. Shape is (num_envs, num_joints)."""

    applied_effort: torch.Tensor
    """The applied effort for the actuator group. Shape is (num_envs, num_joints).

    This is the effort obtained after clipping the :attr:`computed_effort` based on the
    actuator characteristics.
    """

    effort_limit: torch.Tensor
    """The effort limit for the actuator group. Shape is (num_envs, num_joints).

    This limit is used differently depending on the actuator type:

    - **Explicit actuators**: Used for internal torque clipping within the actuator model
      (e.g., motor torque limits in DC motor models).
    - **Implicit actuators**: Live projection of the articulation joint effort limit.
    """

    velocity_limit: torch.Tensor
    """The joint velocity limit for the actuator group [rad/s or m/s]. Shape is (num_envs, num_joints).

    The peak velocity of the actuated joint (the actuator's rated speed reflected at the joint,
    after any gearbox). Feeds the articulation data buffers (e.g. soft joint velocity limits) and
    explicit-model effort clipping; it is not pushed to the physics solver. Defaults to
    ``joint_velocity_limit`` when only the solver constraint is configured.
    """

    stiffness: torch.Tensor
    """The stiffness (P gain) of the PD controller. Shape is (num_envs, num_joints)."""

    damping: torch.Tensor
    """The damping (D gain) of the PD controller. Shape is (num_envs, num_joints)."""

    class _NativeActuatorGains:
        """Live controller-owned gain projection for one native actuator group."""

        def __init__(self, control: ActuatorControl, joint_indices: slice | torch.Tensor):
            self._control = control
            self._joint_indices = joint_indices

        def get(self, attr: Literal["kp", "kd"]) -> torch.Tensor | None:
            """Read one controller gain in the group's public joint order."""
            return self._control.get_native_actuator_gain(attr, self._joint_indices)

    _DEFAULT_MAX_EFFORT_SIM: ClassVar[float] = 1.0e9
    """The default maximum effort for the actuator joints in the simulation. Defaults to 1.0e9.

    If :attr:`ActuatorBaseCfg.joint_effort_limit` is not specified and the actuator is an explicit
    actuator, then this value is used for the construction-time solver property.
    """

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        """Initialize the actuator.

        The actuator parameters are parsed from the configuration and stored as buffers. If the parameters
        are not specified in the configuration, then their values provided in the constructor are used.

        .. note::
            The values in the constructor are typically obtained through the USD values passed from the PhysX API calls
            corresponding to the joints in the actuator model; these values serve as default values if the parameters
            are not specified in the cfg.



        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_ids: The joint indices in the articulation. If :obj:`slice(None)`, then all
                the joints in the articulation are part of the group.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
            stiffness: The default joint stiffness (P gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            damping: The default joint damping (D gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            armature: The default joint armature. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            friction: The default joint static friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            dynamic_friction: The default joint dynamic friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            viscous_friction: The default joint viscous friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            effort_limit: The default effort limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            velocity_limit: The default velocity limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
        """
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._joint_names = joint_names
        self._joint_indices = joint_ids
        self.joint_property_resolution_table: dict[str, list] = {}
        self._joint_property_snapshot = self._resolve_joint_property_snapshot(
            armature=armature,
            friction=friction,
            dynamic_friction=dynamic_friction,
            viscous_friction=viscous_friction,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
        )

        for parameter_name, default_value in (("stiffness", stiffness), ("damping", damping)):
            cfg_value = getattr(self.cfg, parameter_name)
            value = self._parse_joint_parameter(cfg_value, default_value)
            setattr(self, parameter_name, value)
            allclose = (
                torch.all(value == default_value)
                if isinstance(default_value, (float, int))
                else torch.allclose(value, default_value)
            )
            if cfg_value is None or not allclose:
                self._record_actuator_resolution(
                    cfg_val=cfg_value,
                    new_val=value[0],
                    usd_val=default_value,
                    joint_names=joint_names,
                    joint_ids=joint_ids,
                    actuator_param=parameter_name,
                )

        self.velocity_limit = self._parse_joint_parameter(
            self.cfg.velocity_limit,
            self._joint_property_snapshot["velocity_limit"],
        )
        effort_default = self._joint_property_snapshot["effort_limit"] if self.is_implicit_model else effort_limit
        self.effort_limit = self._parse_joint_parameter(self.cfg.effort_limit, effort_default)

        # create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        # resolve joint indices for printing
        joint_indices = self.joint_indices
        if isinstance(joint_indices, slice):
            joint_indices = list(range(self.num_joints))
        # resolve model type (implicit or explicit)
        model_type = "implicit" if self.is_implicit_model else "explicit"

        return (
            f"<class {self.__class__.__name__}> object:\n"
            f"\tModel type            : {model_type}\n"
            f"\tNumber of joints      : {self.num_joints}\n"
            f"\tJoint names expression: {self.cfg.joint_names_expr}\n"
            f"\tJoint names           : {self.joint_names}\n"
            f"\tJoint indices         : {joint_indices}\n"
        )

    """
    Properties.
    """

    @property
    def num_joints(self) -> int:
        """Number of actuators in the group."""
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        """Articulation's joint names that are part of the group."""
        return self._joint_names

    @property
    def joint_indices(self) -> slice | torch.Tensor:
        """Articulation's joint indices that are part of the group.

        Note:
            If :obj:`slice(None)` is returned, then the group contains all the joints in the articulation.
            We do this to avoid unnecessary indexing of the joints for performance reasons.
        """
        return self._joint_indices

    @property
    def stiffness(self) -> torch.Tensor:
        """Current stiffness values [N/m or N·m/rad, depending on joint type]."""
        native_gains = self.__dict__.get("_native_actuator_gains")
        return self._stiffness if native_gains is None else native_gains.get("kp")

    @stiffness.setter
    def stiffness(self, value: torch.Tensor) -> None:
        self._set_actuator_gain_property("stiffness", value)

    @property
    def damping(self) -> torch.Tensor:
        """Current damping values [N·s/m or N·m·s/rad, depending on joint type]."""
        native_gains = self.__dict__.get("_native_actuator_gains")
        return self._damping if native_gains is None else native_gains.get("kd")

    @damping.setter
    def damping(self, value: torch.Tensor) -> None:
        self._set_actuator_gain_property("damping", value)

    @property
    def effort_limit_sim(self) -> torch.Tensor:
        """Deprecated solver effort limit [N or N·m, depending on joint type].

        .. deprecated:: 3.0
            Read :attr:`isaaclab.assets.ArticulationData.joint_effort_limits` instead.
            This compatibility accessor will be removed in 4.0.
        """
        return self._get_deprecated_joint_property("effort_limit_sim", "effort_limit")

    @property
    def velocity_limit_sim(self) -> torch.Tensor:
        """Deprecated solver velocity limit [m/s or rad/s, depending on joint type].

        .. deprecated:: 3.0
            Read :attr:`isaaclab.assets.ArticulationData.joint_vel_limits` instead.
            This compatibility accessor will be removed in 4.0.
        """
        return self._get_deprecated_joint_property("velocity_limit_sim", "velocity_limit")

    @property
    def armature(self) -> torch.Tensor:
        """Deprecated joint armature [kg or kg·m², depending on joint type].

        .. deprecated:: 3.0
            Read :attr:`isaaclab.assets.ArticulationData.joint_armature` instead.
            This compatibility accessor will be removed in 4.0.
        """
        return self._get_deprecated_joint_property("armature", "armature")

    @property
    def friction(self) -> torch.Tensor:
        """Deprecated static joint friction.

        .. deprecated:: 3.0
            Read :attr:`isaaclab.assets.ArticulationData.joint_friction_coeff` instead.
            This compatibility accessor will be removed in 4.0.
        """
        return self._get_deprecated_joint_property("friction", "friction")

    @property
    def dynamic_friction(self) -> torch.Tensor:
        """Deprecated dynamic joint friction [N or N·m, depending on joint type].

        .. deprecated:: 3.0
            Read :attr:`isaaclab.assets.ArticulationData.joint_dynamic_friction_coeff` instead.
            This compatibility accessor will be removed in 4.0.
        """
        return self._get_deprecated_joint_property("dynamic_friction", "dynamic_friction")

    @property
    def viscous_friction(self) -> torch.Tensor:
        """Deprecated viscous joint friction [N·s/m or N·m·s/rad, depending on joint type].

        .. deprecated:: 3.0
            Read :attr:`isaaclab.assets.ArticulationData.joint_viscous_friction_coeff` instead.
            This compatibility accessor will be removed in 4.0.
        """
        return self._get_deprecated_joint_property("viscous_friction", "viscous_friction")

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int]):
        """Reset the internals within the group.

        Args:
            env_ids: List of environment IDs to reset.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        It computes the articulation actions based on the actuator model type

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
        raise NotImplementedError

    """
    Helper functions.
    """

    @classmethod
    def _build_execution_actuator(cls, actuators: Sequence[ActuatorBase]) -> ActuatorBase:
        """Build one private executor from resolved logical actuator groups."""
        # Retain config metadata; replace every execution tensor below without
        # cloning the logical groups' tensor storage.
        executor = copy.copy(actuators[0])
        executor._joint_names = [name for actuator in actuators for name in actuator.joint_names]
        for name in cls.__dict__["_EXECUTION_PARAMETER_NAMES"]:
            setattr(executor, name, torch.cat([getattr(actuator, name) for actuator in actuators], dim=1))
        executor.computed_effort = torch.zeros(executor._num_envs, len(executor._joint_names), device=executor._device)
        executor.applied_effort = torch.zeros_like(executor.computed_effort)
        return executor

    def _bind_joint_properties(self, control: ActuatorControl) -> None:
        """Bind deprecated joint-property accessors to the owning articulation control."""
        self._joint_property_control = control
        self.__dict__.pop("_joint_property_snapshot", None)

    def _bind_native_actuator_gains(self, control: ActuatorControl) -> None:
        """Bind native gain reads when controllers cover every joint in this group."""
        native_gains = self._NativeActuatorGains(control, self.joint_indices)
        if native_gains.get("kp") is None or native_gains.get("kd") is None:
            return
        self._native_actuator_gains = native_gains
        self.__dict__.pop("_stiffness", None)
        self.__dict__.pop("_damping", None)

    def _set_actuator_gain_property(self, name: Literal["stiffness", "damping"], value: torch.Tensor) -> None:
        """Store a construction gain or reject assignment after native binding."""
        if "_native_actuator_gains" in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is controller-owned after native binding. Use "
                "randomize_actuator_gains() or the backend native gain API to update it."
            )
        self.__dict__[f"_{name}"] = value

    def _get_deprecated_joint_property(self, accessor_name: str, property_name: str) -> torch.Tensor:
        """Return one deprecated joint-property projection without owning its storage."""
        warnings.warn(
            f"{type(self).__name__}.{accessor_name} is deprecated. Read the corresponding "
            "ArticulationData joint property instead; this accessor will be removed in 4.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        control = self.__dict__.get("_joint_property_control")
        if control is not None:
            return getattr(control.get_current_joint_properties(self.joint_indices), property_name)
        return self._joint_property_snapshot[property_name]

    def _resolve_joint_property_snapshot(
        self,
        *,
        armature: torch.Tensor | float,
        friction: torch.Tensor | float,
        dynamic_friction: torch.Tensor | float,
        viscous_friction: torch.Tensor | float,
        effort_limit: torch.Tensor | float,
        velocity_limit: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        """Resolve direct-construction snapshots for deprecated joint-property accessors."""
        effort_cfg = self.cfg.joint_effort_limit
        if effort_cfg is None:
            effort_cfg = self.cfg.effort_limit_sim
        if effort_cfg is None and self.is_implicit_model:
            effort_cfg = self.cfg.effort_limit
        effort_default = effort_limit if self.is_implicit_model else self._DEFAULT_MAX_EFFORT_SIM

        velocity_cfg = self.cfg.joint_velocity_limit
        if velocity_cfg is None:
            velocity_cfg = self.cfg.velocity_limit_sim

        return {
            "effort_limit": self._parse_joint_parameter(effort_cfg, effort_default),
            "velocity_limit": self._parse_joint_parameter(velocity_cfg, velocity_limit),
            "armature": self._parse_joint_parameter(self.cfg.armature, armature),
            "friction": self._parse_joint_parameter(self.cfg.friction, friction),
            "dynamic_friction": self._parse_joint_parameter(self.cfg.dynamic_friction, dynamic_friction),
            "viscous_friction": self._parse_joint_parameter(self.cfg.viscous_friction, viscous_friction),
        }

    def _record_actuator_resolution(self, cfg_val, new_val, usd_val, joint_names, joint_ids, actuator_param: str):
        if actuator_param not in self.joint_property_resolution_table:
            self.joint_property_resolution_table[actuator_param] = []
        table = self.joint_property_resolution_table[actuator_param]

        ids = joint_ids if isinstance(joint_ids, torch.Tensor) else list(range(len(joint_names)))
        for idx, name in enumerate(joint_names):
            cfg_val_log = "Not Specified" if cfg_val is None else float(new_val[idx])
            default_usd_val = usd_val if isinstance(usd_val, (float, int)) else float(usd_val[0][idx])
            applied_val_log = default_usd_val if cfg_val is None else float(new_val[idx])
            table.append([name, int(ids[idx]), default_usd_val, cfg_val_log, applied_val_log])

    def _parse_joint_parameter(
        self, cfg_value: float | dict[str, float] | None, default_value: float | torch.Tensor | None
    ) -> torch.Tensor:
        """Parse the joint parameter from the configuration.

        Args:
            cfg_value: The parameter value from the configuration. If None, then use the default value.
            default_value: The default value to use if the parameter is None. If it is also None,
                then an error is raised.

        Returns:
            The parsed parameter value.

        Raises:
            TypeError: If the parameter value is not of the expected type.
            TypeError: If the default value is not of the expected type.
            ValueError: If the parameter value is None and no default value is provided.
            ValueError: If the default value tensor is the wrong shape.
        """
        # create parameter buffer
        param = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        # parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, dict):
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(cfg_value, self.joint_names)
                # note: need to specify type to be safe (e.g. values are ints, but we want floats)
                param[:, indices] = torch.tensor(values, dtype=torch.float, device=self._device)
            else:
                raise TypeError(
                    f"Invalid type for parameter value: {type(cfg_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or dict."
                )
        elif default_value is not None:
            if isinstance(default_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(default_value)
            elif isinstance(default_value, torch.Tensor):
                # if tensor, then use the same tensor for all joints
                if default_value.shape == (self._num_envs, self.num_joints):
                    param = default_value.float()
                else:
                    raise ValueError(
                        "Invalid default value tensor shape.\n"
                        f"Got: {default_value.shape}\n"
                        f"Expected: {(self._num_envs, self.num_joints)}"
                    )
            else:
                raise TypeError(
                    f"Invalid type for default value: {type(default_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or Tensor."
                )
        else:
            raise ValueError("The parameter value is None and no default value is provided.")

        return param

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques: The desired torques to clip.

        Returns:
            The clipped torques.
        """
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
