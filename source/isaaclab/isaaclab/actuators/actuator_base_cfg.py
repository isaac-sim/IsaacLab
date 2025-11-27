# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


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

    effort_limit: dict[str, float] | float | None = None
    """Force/Torque limit of the joints in the group. Defaults to None.

    This limit is used to clip the computed torque sent to the simulation. If None, the
    limit is set to the value specified in the USD joint prim.

    .. attention::

        The :attr:`effort_limit_sim` attribute should be used to set the effort limit for
        the simulation physics solver.

        The :attr:`effort_limit` attribute is used for clipping the effort output of the
        actuator model **only** in the case of explicit actuators, such as the
        :class:`~isaaclab.actuators.IdealPDActuator`.

    .. note::

        For implicit actuators, the attributes :attr:`effort_limit` and :attr:`effort_limit_sim`
        are equivalent. However, we suggest using the :attr:`effort_limit_sim` attribute because
        it is more intuitive.

    """

    velocity_limit: dict[str, float] | float | None = None
    """Velocity limit of the joints in the group. Defaults to None.

    This limit is used by the actuator model. If None, the limit is set to the value specified
    in the USD joint prim.

    .. attention::

        The :attr:`velocity_limit_sim` attribute should be used to set the velocity limit for
        the simulation physics solver.

        The :attr:`velocity_limit` attribute is used for clipping the effort output of the
        actuator model **only** in the case of explicit actuators, such as the
        :class:`~isaaclab.actuators.IdealPDActuator`.

    .. note::

        For implicit actuators, the attribute :attr:`velocity_limit` is not used. This is to stay
        backwards compatible with previous versions of the Isaac Lab, where this parameter was
        unused since PhysX did not support setting the velocity limit for the joints using the
        PhysX Tensor API.
    """

    effort_limit_sim: dict[str, float] | float | None = None
    """Effort limit of the joints in the group applied to the simulation physics solver. Defaults to None.

    The effort limit is used to constrain the computed joint efforts in the physics engine. If the
    computed effort exceeds this limit, the physics engine will clip the effort to this value.

    Since explicit actuators (e.g. DC motor), compute and clip the effort in the actuator model, this
    limit is by default set to a large value to prevent the physics engine from any additional clipping.
    However, at times, it may be necessary to set this limit to a smaller value as a safety measure.

    If None, the limit is resolved based on the type of actuator model:

    * For implicit actuators, the limit is set to the value specified in the USD joint prim.
    * For explicit actuators, the limit is set to 1.0e9.

    """

    velocity_limit_sim: dict[str, float] | float | None = None
    """Velocity limit of the joints in the group applied to the simulation physics solver. Defaults to None.

    The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
    be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
    faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

    If None, the limit is set to the value specified in the USD joint prim for both implicit and explicit actuators.

    .. tip::
        If the velocity limit is too tight, the physics engine may have trouble converging to a solution.
        In such cases, we recommend either keeping this value sufficiently large or tuning the stiffness and
        damping parameters of the joint to ensure the limits are not violated.

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
    r"""The static friction coefficient of the joints in the group. Defaults to None.

    The joint static friction is a unitless quantity. It relates the magnitude of the spatial force transmitted
    from the parent body to the child body to the maximal static friction force that may be applied by the solver
    to resist the joint motion.

    Mathematically, this means that: :math:`F_{resist} \leq \mu F_{spatial}`, where :math:`F_{resist}`
    is the resisting force applied by the solver and :math:`F_{spatial}` is the spatial force
    transmitted from the parent body to the child body. The simulated static friction effect is therefore
    similar to static and Coulomb static friction.

    If None, the joint static friction is set to the value from the USD joint prim.

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """

    dynamic_friction: dict[str, float] | float | None = None
    """The dynamic friction coefficient of the joints in the group. Defaults to None.

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """

    viscous_friction: dict[str, float] | float | None = None
    """The viscous friction coefficient of the joints in the group. Defaults to None.
    """
