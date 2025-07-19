.. _overview-actuators:


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

As an example of an ideal explicit actuator model, we provide the :class:`isaaclab.actuators.IdealPDActuator`
class, which implements a PD controller with feed-forward effort, and simple clipping based on the configured
maximum effort:

.. math::

    \tau_{j, computed} & = k_p * (q_{des} - q) + k_d * (\dot{q}_{des} - \dot{q}) + \tau_{ff} \\
    \tau_{j, max} & = \gamma \times \tau_{motor, max} \\
    \tau_{j, applied} & = clip(\tau_{computed}, -\tau_{j, max}, \tau_{j, max})


where, :math:`k_p` and :math:`k_d` are joint stiffness and damping gains, :math:`q` and :math:`\dot{q}`
are the current joint positions and velocities, :math:`q_{des}`, :math:`\dot{q}_{des}` and :math:`\tau_{ff}`
are the desired joint positions, velocities and torques commands. The parameters :math:`\gamma` and
:math:`\tau_{motor, max}`  are the gear box ratio and the maximum motor effort possible.

Actuator groups
---------------

The actuator models by themselves are computational blocks that take as inputs the desired joint commands
and output the joint commands to apply into the simulator. They do not contain any knowledge about the
joints they are acting on themselves. These are handled by the :class:`isaaclab.assets.Articulation`
class, which wraps around the physics engine's articulation class.

Actuator are collected as a set of actuated joints on an articulation that are using the same actuator model.
For instance, the quadruped, ANYmal-C, uses series elastic actuator, ANYdrive 3.0, for all its joints. This
grouping configures the actuator model for those joints, translates the input commands to the joint level
commands, and returns the articulation action to set into the simulator. Having an arm with a different
actuator model, such as a DC motor, would require configuring a different actuator group.

The following figure shows the actuator groups for a legged mobile manipulator:

.. image:: ../../_static/actuator-group/actuator-light.svg
    :class: only-light
    :align: center
    :alt: Actuator models for a legged mobile manipulator
    :width: 80%

.. image:: ../../_static/actuator-group/actuator-dark.svg
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Actuator models for a legged mobile manipulator

.. seealso::

    We provide implementations for various explicit actuator models. These are detailed in
    `isaaclab.actuators <../../api/lab/isaaclab.actuators.html>`_ sub-package.

Considerations when using actuators
-----------------------------------

As explained in the previous sections, there are two main types of actuator models: implicit and explicit.
The implicit actuator model is provided by the physics engine. This means that when the user sets either
a desired position or velocity, the physics engine will internally compute the efforts that need to be
applied to the joints to achieve the desired behavior. In PhysX, the PD controller adds numerical damping
to the desired effort, which results in more stable behavior.

The explicit actuator model is provided by the user. This means that when the user sets either a desired
position or velocity, the user's model will compute the efforts that need to be applied to the joints to
achieve the desired behavior. While this provides more flexibility, it can also lead to some numerical
instabilities. One way to mitigate this is to use the ``armature`` parameter of the actuator model, either in
the USD file or in the articulation config. This parameter is used to dampen the joint response and helps
improve the numerical stability of the simulation. More details on how to improve articulation stability
can be found in the `OmniPhysics documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html>`_.

What does this mean for the user? It means that policies trained with implicit actuators may not transfer
to the exact same robot model when using explicit actuators. If you are running into issues like this, or
in cases where policies do not converge on explicit actuators while they do on implicit ones, increasing
or setting the ``armature`` parameter to a higher value may help.
