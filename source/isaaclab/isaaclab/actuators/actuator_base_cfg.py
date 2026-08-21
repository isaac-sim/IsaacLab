# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


def _is_implicit_actuator_cfg(cfg: ActuatorBaseCfg) -> bool:
    """Return whether an actuator configuration resolves to an implicit actuator class.

    Reads the :attr:`~isaaclab.actuators.ActuatorBase.is_implicit_model` class flag.
    Lazily resolving string references participate through attribute forwarding.
    """
    return bool(getattr(cfg.class_type, "is_implicit_model", False))


@configclass
class ActuatorBaseCfg:
    """Configuration for default actuators in an articulation."""

    class_type: type = MISSING
    """The associated actuator class.

    The class should inherit from :class:`isaaclab.actuators.ActuatorBase`.
    """

    joint_names_expr: list[str] = MISSING
    """Articulation's joint names that are part of the group.

    Note:
        This can be a list of joint names or a list of regex expressions (e.g. ".*").
    """

    actuator_effort_limit: dict[str, float] | float | None = None
    """Actuator-model effort clipping limit [N or N·m, depending on joint type].

    The actuator's rated force/torque reflected at the joint. Explicit actuator models
    clip their computed effort with it; implicit actuators use it as the model-facing
    limit for effort telemetry. If None, it defaults to the authored/USD joint effort
    limit (explicit) or tracks the live solver limit (implicit). It is not a solver
    limit; that is :attr:`joint_effort_limit`.

    :class:`~isaaclab.actuators.RemotizedPDActuator` instead uses the
    angle-dependent limits in its ``joint_parameter_lookup``.
    """

    actuator_velocity_limit: dict[str, float] | float | None = None
    """Velocity limit of the joints in the group. Defaults to None.

    This limit is used by the actuator model. If None, the limit is set to the value specified
    in the USD joint prim.

    .. attention::

        This attribute describes the actuator's peak velocity, i.e. the actuator's rated speed
        reflected at the joint (after any gearbox). It populates the actuator data
        buffers (e.g. :attr:`~isaaclab.assets.ArticulationData.soft_joint_vel_limits`, read by
        velocity-limit terminations and rewards). Explicit models with speed-dependent limits,
        such as :class:`DCMotor`, also use it to clip effort. It is **not** pushed to the physics
        solver.

        Use :attr:`joint_velocity_limit` to request a solver-level hard clamp. A physical
        actuator limits joint speed through its torque curve rather than a kinematic clamp,
        so the two limits are resolved independently. When only
        :attr:`joint_velocity_limit` is set, it also serves as the joint velocity limit.
    """

    joint_effort_limit: dict[str, float] | float | None = None
    """Construction-time joint solver effort override [N or N·m, depending on joint type].

    The live value is owned by :class:`isaaclab.assets.ArticulationData`.
    """

    joint_velocity_limit: dict[str, float] | float | None = None
    """Construction-time requested joint solver velocity limit [m/s or rad/s, depending on joint type].

    The live value is owned by :class:`isaaclab.assets.ArticulationData`; enforcement is
    backend-dependent.
    """

    effort_limit_sim: dict[str, float] | float | None = None
    """Deprecated alias for :attr:`joint_effort_limit`.

    .. deprecated:: 3.0
        Use :attr:`joint_effort_limit` instead. This alias will be removed in 4.0.
    """

    velocity_limit_sim: dict[str, float] | float | None = None
    """Deprecated alias for :attr:`joint_velocity_limit`.

    .. deprecated:: 3.0
        Use :attr:`joint_velocity_limit` instead. This alias will be removed in 4.0.
    """

    stiffness: dict[str, float] | float | None = MISSING
    """Stiffness gains (also known as p-gain) of the joints in the group.

    The behavior of the stiffness is different for implicit and explicit actuators. For implicit actuators,
    the stiffness gets set into the physics engine directly. For explicit actuators, the stiffness is used
    by the actuator model to compute the joint efforts.

    If None, the stiffness is set to the value from the USD joint prim.
    """

    damping: dict[str, float] | float | None = MISSING
    """Damping gains (also known as d-gain) of the joints in the group.

    The behavior of the damping is different for implicit and explicit actuators. For implicit actuators,
    the damping gets set into the physics engine directly. For explicit actuators, the damping gain is used
    by the actuator model to compute the joint efforts.

    If None, the damping is set to the value from the USD joint prim.
    """

    armature: dict[str, float] | float | None = None
    """Armature of the joints in the group. Defaults to None.

    The armature is directly added to the corresponding joint-space inertia. It helps improve the
    simulation stability by reducing the joint velocities.

    It is a physics engine solver parameter that gets set into the simulation.

    If None, the armature is set to the value from the USD joint prim.
    """

    friction: dict[str, float] | float | None = None
    r"""The static friction parameter of the joints in the group. Defaults to None.

    The meaning and units depend on the Isaac Sim version. In Isaac Sim 4.5, this value is a unitless
    coefficient relating the transmitted spatial force to the maximum static friction force. In Isaac Sim
    5.0 and later, PhysX interprets it as the maximum static friction effort [N or N·m, depending on joint type].

    If None, the joint static friction is set to the value from the USD joint prim.
    """

    dynamic_friction: dict[str, float] | float | None = None
    """The dynamic friction parameter of the joints in the group. Defaults to None.

    Dynamic joint friction is supported by this backend in Isaac Sim 5.0 and later, where PhysX interprets
    this value as the dynamic friction effort [N or N·m, depending on joint type]. In Isaac Sim 4.5, this
    configuration value is not applied by the backend.

    If None, the joint dynamic friction is set to the value from the USD joint prim in supported versions.
    """

    viscous_friction: dict[str, float] | float | None = None
    """The viscous friction coefficient of the joints in the group. Defaults to None.

    Viscous joint friction is supported by this backend in Isaac Sim 5.0 and later. It is a coefficient
    [N·s/m or N·m·s/rad, depending on joint type] multiplying joint velocity to produce a friction effort.
    In Isaac Sim 4.5, this configuration value is not applied by the backend.

    If None, the joint viscous friction is set to the value from the USD joint prim in supported versions.
    """

    effort_limit: dict[str, float] | float | None = None
    """Deprecated effort limit [N or N·m, depending on joint type].

    .. deprecated:: 3.0
        For explicit actuators, use :attr:`actuator_effort_limit`. For implicit
        actuators, use :attr:`joint_effort_limit`. This alias will be removed in 4.0.
    """

    velocity_limit: dict[str, float] | float | None = None
    """Deprecated velocity limit [m/s or rad/s, depending on joint type].

    .. deprecated:: 3.0
        Use :attr:`actuator_velocity_limit` for the actuator-model limit or
        :attr:`joint_velocity_limit` for the solver limit. This alias will be
        removed in 4.0.
    """
