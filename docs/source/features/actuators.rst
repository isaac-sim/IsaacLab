Actuators
=========

An articulated system comprises of actuated joints, also called the degrees of freedom (DOF).
In a physical system, the actuation typically happens either through active components, such as
electric or hydraulic motors, or passive components, such as springs. These components can introduce
certain non-linear characteristics which includes delays or maximum producible velocity or torque.

In simulation, the joints are either position, velocity, or torque-controlled. For position and velocity
control, the physics engine internally implements a spring-damp (PD) controller which computes the torques
applied on the actuated joints. In torque-control, the commands are set directly as the joint efforts.
While this mimics an ideal behavior of the joint mechanism, it does not truly model how the drives work
in the physical world. Thus, we provide a mechanism to inject external models to compute the
joint commands that would represent the physical robot's behavior.

Actuator models
---------------

We name two different types of actuator models:

1. **implicit**: corresponds to the ideal simulation mechanism (provided by physics engine).
2. **explicit**: corresponds to external drive models (implemented by user).

The explicit actuator model performs two steps: 1) it computes the desired joint torques for tracking
the input commands, and 2) it clips the desired torques based on the motor capabilities. The clipped
torques are the desired actuation efforts that are set into the simulation.

All explicit models inherit from the base actuator model, :class:`IdealActuator`, which implements a
PD controller with feed-forward effort, and simple clipping based on the configured maximum effort:

.. math::

    \tau_{j, computed} & = k_p * (q - q_{des}) + k_d * (\dot{q} - \dot{q}_{des}) + \tau_{ff} \\
    \tau_{j, applied} & = clip(\tau_{computed}, -\tau_{j, max}, \tau_{j, max})


where, :math:`k_p` and :math:`k_d` are joint stiffness and damping gains, :math:`q` and :math:`\dot{q}`
are the current joint positions and velocities, :math:`q_{des}`, :math:`\dot{q}_{des}` and :math:`\tau_{ff}`
are the desired joint positions, velocities and torques commands. The parameters :math:`\gamma` and
:math:`\tau_{motor, max}`  are the gear box ratio and the maximum motor effort possible.

.. seealso::

    We provide implementations for various explicit actuator models. These are detailed in
    `omni.isaac.orbit.actuators.model <../api/orbit.actuators.model.html>`_ sub-package.

Actuator groups
---------------

The actuator models by themselves are computational blocks that take as inputs the desired joint commands
and output the the joint commands to apply into the simulator. They do not contain any knowledge about the
joints they are acting on themselves. These are handled by the actuator groups.

Actuator groups collect a set of actuated joints on an articulation that are using the same actuator model.
For instance, the quadruped, ANYmal-C, uses series elastic actuator, ANYdrive 3.0, for all its joints. This
grouping configures the actuator model for those joints, translates the input commands to the joint level
commands, and returns the articulation action to set into the simulator. Through this mechanism, it is also
possible to configure the included joints under some constraints, such as mimicking of gripper commands or
introducing non-holonomic constraints for a wheel base.

An articulated system can be composed of different actuator groups. For instance, a legged mobile manipulator
can be composed of a base group, an arm group, and a gripper group. If the base is the quadruped, ANYmal-C,
the base group can utilize the actuator network to model the series elastic actuation. Similarly, the arm,
a Kinova Jaco2, can use a DC motor model and take joint positions as input commands. Finally, the gripper group
can employ the gripper mimic group which processes binary open/close command into individual gripper joint actions.

.. image:: ../_static/actuator_groups.svg
    :width: 600
    :align: center
    :alt: Actuator groups for a legged mobile manipulator


.. seealso::

    For more information on the actuator groups, please refer to the documentation of the
    `omni.isaac.orbit.actuators.group <../api/orbit.actuators.group.html>`_ subpackage.
