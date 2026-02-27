Motion Generators
=================

Robotic tasks are typically defined in task-space in terms of desired
end-effector trajectory, while control actions are executed in the
joint-space. This naturally leads to *joint-space* and *task-space*
(operational-space) control methods. However, successful execution of
interaction tasks using motion control often requires an accurate model
of both the robot manipulator as well as its environment. While a
sufficiently precise manipulator's model might be known, detailed
description of environment is hard to obtain :cite:p:`siciliano2009force`.
Planning errors caused by this mismatch can be overcome by introducing a
*compliant* behavior during interaction.

While compliance is achievable passively through robot's structure (such
as elastic actuators, soft robot arms), we are more interested in
controller designs that focus on active interaction control. These are
broadly categorized into:

1. **impedance control:** indirect control method where motion deviations
   caused during interaction relates to contact force as a mass-spring-damper
   system with adjustable parameters (stiffness and damping). A specialized case
   of this is *stiffness* control where only the static relationship between
   position error and contact force is considered.

2. **hybrid force/motion control:** active control method which controls motion
   and force along unconstrained and constrained task directions respectively.
   Among the various schemes for hybrid motion control, the provided implementation
   is based on inverse dynamics control in the operational space :cite:p:`khatib1987osc`.

.. note::

    To provide an even broader set of motion generators, we welcome contributions from the
    community. If you are interested, please open an issue to start a discussion!


Joint-space controllers
-----------------------

Torque control
~~~~~~~~~~~~~~

Action dimensions: ``"n"`` (number of joints)

In torque control mode, the input actions are directly set as feed-forward
joint torque commands, i.e. at every time-step,

.. math::

    \tau = \tau_{des}

Thus, this control mode is achievable by setting the command type for the actuator group, via
the :class:`ActuatorControlCfg` class, to ``"t_abs"``.


Velocity control
~~~~~~~~~~~~~~~~

Action dimensions: ``"n"`` (number of joints)

In velocity control mode, a proportional control law is required to reduce the error between the
current and desired joint velocities. Based on input actions, the joint torques commands are computed as:

.. math::

    \tau = k_d (\dot{q}_{des} - \dot{q})

where :math:`k_d` are the gains parsed from configuration.

This control mode is achievable by setting the command type for the actuator group, via
the :class:`ActuatorControlCfg` class, to ``"v_abs"`` or ``"v_rel"``.

.. attention::

    While performing velocity control, in many cases, gravity compensation is required to ensure better
    tracking of the command. In this case, we suggest disabling gravity for the links in the articulation
    in simulation.

Position control with fixed impedance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions: ``"n"`` (number of joints)

In position control mode, a proportional-damping (PD) control law is employed to track the desired joint
positions and ensuring the articulation remains still at the desired location (i.e., desired joint velocities
are zero). Based on the input actions, the joint torque commands are computed as:

.. math::

    \tau = k_p (q_{des} - q)  - k_d \dot{q}

where :math:`k_p` and :math:`k_d` are the gains parsed from configuration.

In its simplest above form,  the control mode is achievable by setting the command type for the actuator group,
via the :class:`ActuatorControlCfg` class, to ``"p_abs"`` or ``"p_rel"``.

However, a more complete formulation which considers the dynamics of the articulation would be:

.. math::

    \tau = M \left( k_p (q_{des} - q)  - k_d \dot{q} \right) + g

where :math:`M` is the joint-space inertia matrix of size :math:`n \times n`, and :math:`g` is the joint-space
gravity vector. This implementation is available through the :class:`JointImpedanceController` class by setting the
impedance mode to ``"fixed"``. The gains :math:`k_p` are parsed from the input configuration and :math:`k_d`
are computed while considering the system as a decoupled point-mass oscillator, i.e.,

.. math::

    k_d = 2 \sqrt{k_p} \times D

where :math:`D` is the damping ratio of the system. Critical damping is achieved for :math:`D = 1`, overcritical
damping for :math:`D > 1` and undercritical damping for :math:`D < 1`.

Additionally, it is possible to disable the inertial or gravity compensation in the controller by setting the
flags :attr:`inertial_compensation` and  :attr:`gravity_compensation` in the configuration to :obj:`False`,
respectively.

Position control with variable stiffness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions: ``"2n"`` (number of joints)

In stiffness control, the same formulation as above is employed, however, the gains :math:`k_p` are part of
the input commands. This implementation is available through the :class:`JointImpedanceController` class by
setting the impedance mode to ``"variable_kp"``.

Position control with variable impedance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions: ``"3n"`` (number of joints)

In impedance control, the same formulation as above is employed, however, both :math:`k_p` and :math:`k_d`
are part of the input commands. This implementation is available through the :class:`JointImpedanceController`
class by setting the impedance mode to ``"variable"``.

Task-space controllers
----------------------

Differential inverse kinematics (IK)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions:  ``"3"`` (relative/absolute position), ``"6"`` (relative pose), or ``"7"`` (absolute pose)

Inverse kinematics converts the task-space tracking error to joint-space error. In its most typical implementation,
the pose error in the task-sace, :math:`\Delta \chi_e = (\Delta p_e, \Delta \phi_e)`, is computed as the cartesian
distance between the desired and current task-space positions, and the shortest distance in :math:`\mathbb{SO}(3)`
between the desired and current task-space orientations.

Using the geometric Jacobian :math:`J_{eO} \in \mathbb{R}^{6 \times n}`, that relates task-space velocity to joint-space velocities,
we design the control law to obtain the desired joint positions as:

.. math::

    q_{des} = q + \eta J_{eO}^{-} \Delta \chi_e

where :math:`\eta` is a scaling parameter and :math:`J_{eO}^{-}` is the pseudo-inverse of the Jacobian.

It is possible to compute the pseudo-inverse of the Jacobian using different formulations:

* Moore-Penrose pseduo-inverse: :math:`A^{-} = A^T(AA^T)^{-1}`.
* Levenberg-Marquardt pseduo-inverse (damped least-squares): :math:`A^{-} = A^T (AA^T + \lambda \mathbb{I})^{-1}`.
* Tanspose pseudo-inverse: :math:`A^{-} = A^T`.
* Adaptive singular-vale decomposition (SVD) pseduo-inverse from :cite:t:`buss2004ik`.

These implementations are available through the :class:`DifferentialInverseKinematics` class.

Impedance controller
~~~~~~~~~~~~~~~~~~~~


It uses task-space pose error and Jacobian to compute join torques through mass-spring-damper system
with a) fixed stiffness, b) variable stiffness (stiffness control),
and c) variable stiffness and damping (impedance control).

Operational-space controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to task-space impedance
control but uses the Equation of Motion (EoM) for computing the
task-space force

Closed-loop proportional force controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It uses a proportional term
to track the desired wrench command with respect to current wrench at
the end-effector.

Hybrid force-motion controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It combines closed-loop force control
and operational-space motion control to compute the desired wrench at
the end-effector. It uses selection matrices that define the
unconstrainted and constrained task directions.


Reactive planners
-----------------

Typical task-space controllers do not account for motion constraints
such as joint limits, self-collision and environment collision. Instead
they rely on high-level planners (such as RRT) to handle these
non-Euclidean constraints and give joint/task-space way-points to the
controller. However, these methods are often conservative and have
undesirable deceleration when close to an object. More recently,
different approaches combine the constraints directly into an
optimization problem, thereby providing a holistic solution for motion
generation and control.

We currently support the following planners:

-  **RMPFlow (lula):** An acceleration-based policy that composes various Reimannian Motion Policies (RMPs) to
   solve a hierarchy of tasks :cite:p:`cheng2021rmpflow`. It is capable of performing dynamic collision
   avoidance while navigating the end-effector to a target.

-  **MPC (OCS2):** A receding horizon control policy based on sequential linear-quadratic (SLQ) programming.
   It formulates various constraints into a single optimization problem via soft-penalties and uses automatic
   differentiation to compute derivatives of the system dynamics, constraints and costs. Currently, we support
   the MPC formulation for end-effector trajectory tracking in fixed-arm and mobile manipulators. The formulation
   considers a kinematic system model with joint limits and self-collision avoidance :cite:p:`mittal2021articulated`.


.. warning::

    We wrap around the python bindings for these reactive planners to perform a batched computing of
    robot actions. However, their current implementations are CPU-based which may cause certain
    slowdown for learning.
