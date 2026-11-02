Controllers
===========

.. currentmodule:: isaaclab.controllers

Controllers turn desired motion or contact forces into joint commands. Joint-space
controllers act on individual joints; task-space controllers act on an end effector.

Contact tasks often need compliance to tolerate errors in the robot or environment
model :cite:p:`siciliano2009force`. Two common approaches are:

* **Impedance control:** Relates motion error to contact force through a
  mass-spring-damper model. Stiffness control uses only the static relationship
  between position error and force.
* **Hybrid force/motion control:** Controls motion along free directions and force
  along constrained directions. Isaac Lab's implementation uses operational-space
  inverse dynamics :cite:p:`khatib1987osc`.


Choosing a controller
---------------------

The core controllers below live in ``isaaclab.controllers``. Isaac Lab also provides
:ref:`Newton controllers <newton-controllers>` with additional control options and
an iterative IK solver.

.. list-table::
   :header-rows: 1
   :widths: 22 24 20 34

   * - Controller
     - Input command
     - Output
     - Required state
   * - :class:`DifferentialIKController`
     - End-effector position or pose, absolute or relative
     - Joint position targets
     - Joint positions, end-effector pose, and Jacobian in a consistent frame
   * - :class:`OperationalSpaceController`
     - End-effector pose and/or wrench, optionally stiffness and damping ratios
     - Joint efforts
     - Jacobian; pose and velocity for motion control; mass matrix for inertial decoupling;
       gravity vector, contact force, and joint state as required by enabled features

Basic joint-space control sets joint targets directly on the actuators.
For joint-space impedance control, use :class:`JointImpedanceController`
(see `Joint-space controllers`_).

Start with differential IK for reaching or teleoperation with position-controlled
actuators. Choose OSC when you need torque commands, task-space compliance, or
simultaneous motion and contact force control. For example, wiping a table requires
motion along its surface while maintaining a force normal to it.

The :doc:`IK walkthrough <../how-to/run_diff_ik>` and
:doc:`OSC walkthrough <../how-to/run_osc>` provide complete scenes, state extraction,
and runnable demonstrations. Both tutorial scripts use Isaac Sim PhysX and require
an Isaac Sim installation.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item::

      .. image:: ../_static/tutorials/tutorial_task_space_controller_crop.jpg
         :alt: Franka arms on stands with coordinate-frame markers at their end effectors.

      **IK: reach a pose.** Each arm tracks an end-effector goal through joint
      position targets. :doc:`Run the IK demonstration <../how-to/run_diff_ik>`
      to watch the measured and desired frame markers converge.

   .. grid-item::

      .. image:: ../_static/tutorials/tutorial_operational_space_controller.jpg
         :alt: Franka arms touching translucent red tilted walls with their fingertips.

      **OSC: control motion and contact force.** The tilted-wall example
      commands 10 N along the wall normal while tracking pose in the other
      directions. :doc:`Run the OSC demonstration <../how-to/run_osc>`.

.. _newton-controllers:

Newton controllers
------------------

``isaaclab_newton.controllers`` wraps Newton's batched controllers for use with
PyTorch tensors. These three controllers work with any physics backend: the caller
supplies the robot state, Jacobian, and dynamics required by the enabled features.
They do not require a Newton simulation model.

* **Differential IK** —
  :class:`~isaaclab_newton.controllers.NewtonDifferentialIKController` produces
  joint position targets from an end-effector goal. It supports axis weighting,
  adaptive damping, truncated SVD, joint-limit avoidance, and null-space posture
  control. Configure it with
  :class:`~isaaclab_newton.controllers.NewtonDifferentialIKControllerCfg`.
* **Joint impedance** —
  :class:`~isaaclab_newton.controllers.NewtonJointImpedanceController` produces
  joint efforts from desired joint positions and velocities. Options include
  inertia decoupling, gravity and Coriolis compensation, acceleration feedforward,
  and gains that can change each control step. Configure it with
  :class:`~isaaclab_newton.controllers.NewtonJointImpedanceControllerCfg`.
* **Operational-space control** —
  :class:`~isaaclab_newton.controllers.NewtonOperationalSpaceController` produces
  joint efforts for task-space motion and wrench control. It supports desired
  end-effector velocity, separate linear and angular selection frames, and
  null-space posture control. Configure it with
  :class:`~isaaclab_newton.controllers.NewtonOperationalSpaceControllerCfg`.

In a direct environment, create the controller once, pass current state and targets
to ``compute()`` each control step, and apply the returned joint targets or efforts
through ``articulation.actuators``. These wrappers have their own configurations and
``compute()`` arguments; they do not use the core controllers' ``set_command()`` API.
They compute in float32, and contiguous float32 inputs can be shared without copies.
Returned tensors reference reusable output buffers, so clone them if you need to
keep a result after the next call.

For manager-based tasks, use
:class:`~isaaclab_newton.envs.mdp.NewtonDifferentialInverseKinematicsActionCfg` or
:class:`~isaaclab_newton.envs.mdp.NewtonOperationalSpaceControllerActionCfg` to connect
the corresponding controller to the robot. For example, this action commands an
absolute Franka hand position in the robot's root frame:

.. code-block:: python

   from isaaclab_newton.controllers import NewtonDifferentialIKControllerCfg
   from isaaclab_newton.envs.mdp import NewtonDifferentialInverseKinematicsActionCfg

   arm_action = NewtonDifferentialInverseKinematicsActionCfg(
       asset_name="robot",
       joint_names=["panda_joint[1-7]"],
       body_name="panda_hand",
       command_type="position",
       controller=NewtonDifferentialIKControllerCfg(
           ik_method="damped_least_squares",
           bandwidth=10.0,
           damping=0.05,
       ),
   )

Add this term to the task's actions configuration and match the asset, joint, and
body names to your robot. Use position-controlled actuators for differential IK;
use effort control for operational-space control.

Iterative IK with Newton
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab_newton.controllers.ik.NewtonIKSolver` solves pose objectives
iteratively using a Newton model, rather than taking a differential IK step from
a supplied Jacobian. Combine
:class:`~isaaclab_newton.controllers.ik.NewtonIKPoseObjectiveCfg` objectives for one
or more bodies with
:class:`~isaaclab_newton.controllers.ik.NewtonIKJointLimitObjectiveCfg` to penalize
joint-limit violations. :class:`~isaaclab_newton.controllers.ik.NewtonIKSolverCfg`
selects LM or L-BFGS optimization, the Jacobian method, seed sampling, and iteration
count.

Use :class:`~isaaclab_newton.envs.mdp.NewtonInverseKinematicsActionCfg` to drive this
solver from a manager-based task. This action currently requires a fixed-base
articulation with a registered Newton prototype model; its joints and target
bodies must resolve in both Isaac Lab and that model.

See :doc:`/source/api/lab_newton/isaaclab_newton.controllers` for the full controller,
solver, and configuration APIs.


.. _controller-control-loop:

The control loop
----------------

The IK and OSC instances compute targets from state supplied by the caller;
they do not read the simulation or submit actuator commands.
:meth:`~DifferentialIKController.set_command` stores a goal until it is replaced,
and every control cycle reads fresh state, calls ``compute()``, stages the targets,
and steps the simulation; see :ref:`controller-api-examples` for the loop from the
IK walkthrough. IK targets are joint positions that the configured position
actuator tracks. OSC targets are joint efforts, so configure zero actuator
stiffness and damping to keep actuator feedback from adding to them.

Keep the command, end-effector pose, and Jacobian in a consistent coordinate frame.
The tutorials use the suffix ``_b`` for the robot base frame and ``_w`` for the
world frame; these names do not transform the data.
Absolute poses contain position followed by a quaternion in ``(x, y, z, w)`` order.
For OSC, commands and axis selections can use a separate task frame, while state
passed to ``compute()`` remains in the base frame.

Joint-space controllers
-----------------------

There is no dedicated configuration class for selecting a joint-space control mode. Instead, the
control mode follows from which targets you set through ``articulation.actuators``: effort targets
for torque control, velocity targets for velocity control, and position targets for position
control. See :ref:`actuators-runtime-api` for the runtime API.

Torque control
~~~~~~~~~~~~~~

Action dimensions: ``"n"`` (number of joints)

In torque control mode, the input actions are directly set as feed-forward
joint torque commands, i.e. at every time-step,

.. math::

    \tau = \tau_{des}


Velocity control
~~~~~~~~~~~~~~~~

Action dimensions: ``"n"`` (number of joints)

In velocity control mode, a proportional control law is required to reduce the error between the
current and desired joint velocities. Based on input actions, the joint torques commands are computed as:

.. math::

    \tau = k_d (\dot{q}_{des} - \dot{q})

where :math:`k_d` are the gains parsed from configuration.

.. attention::

    While performing velocity control, in many cases, gravity compensation is required to ensure better
    tracking of the command. In this case, we suggest disabling gravity for the links in the articulation
    in simulation.

Position control with fixed impedance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions: ``"n"`` (number of joints)

Position control uses a proportional-derivative (PD) law to track joint positions
with zero target velocity:

.. math::

    \tau = k_p (q_{des} - q)  - k_d \dot{q}

where :math:`k_p` and :math:`k_d` are the gains parsed from configuration.

A more complete formulation which considers the dynamics of the articulation would be:

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

Variable-stiffness control uses the same law, with :math:`k_p` supplied in the
command. Set :class:`JointImpedanceController` to ``"variable_kp"`` mode.

Position control with variable impedance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions: ``"3n"`` (number of joints)

Variable-impedance control also supplies damping through the command. Set
:class:`JointImpedanceController` to ``"variable"`` mode.

Task-space controllers
----------------------

Differential inverse kinematics (IK)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Action dimensions:  ``"3"`` (relative/absolute position), ``"6"`` (relative pose), or ``"7"`` (absolute pose)

Inverse kinematics converts the task-space tracking error to joint-space error. In its most typical implementation,
the pose error in the task-space, :math:`\Delta \chi_e = (\Delta p_e, \Delta \phi_e)`, is computed as the Cartesian
distance between the desired and current task-space positions, and the shortest distance in :math:`\mathbb{SO}(3)`
between the desired and current task-space orientations.

Using the geometric Jacobian :math:`J_{eO} \in \mathbb{R}^{6 \times n}`, that relates task-space velocity to joint-space velocities,
we design the control law to obtain the desired joint positions as:

.. math::

    q_{des} = q + \eta J_{eO}^{-} \Delta \chi_e

where :math:`\eta` is a scaling parameter and :math:`J_{eO}^{-}` is the pseudo-inverse of the Jacobian.

It is possible to compute the pseudo-inverse of the Jacobian using different formulations:

* Moore-Penrose pseudo-inverse (``"pinv"``): :math:`A^{-} = A^T(AA^T)^{-1}`.
* Levenberg-Marquardt pseudo-inverse (damped least-squares, ``"dls"``):
  :math:`A^{-} = A^T (AA^T + \lambda^2 \mathbb{I})^{-1}`.
* Transpose pseudo-inverse (``"trans"``): :math:`A^{-} = A^T`.
* Adaptive singular-value decomposition (SVD) pseudo-inverse (``"svd"``) from :cite:t:`buss2004ik`.
* Manipulability-aware damped least-squares (``"adaptive_dls"``): the damping :math:`\lambda`
  increases as the Jacobian approaches a singularity.

These implementations are available through the :class:`DifferentialIKController` class.
See :doc:`../how-to/run_diff_ik` for a runnable example.

Impedance controller
~~~~~~~~~~~~~~~~~~~~

It uses task-space pose error and Jacobian to compute joint torques through mass-spring-damper system
with a) fixed stiffness, b) variable stiffness (stiffness control),
and c) variable stiffness and damping (impedance control).

Operational-space controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to task-space impedance
control but uses the Equation of Motion (EoM) for computing the
task-space force. See :doc:`../how-to/run_osc` for a runnable example.

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
unconstrained and constrained task directions.


.. _controller-api-examples:

Practical API examples
----------------------

Joint targets
~~~~~~~~~~~~~

These alternative command examples assume an initialized ``robot`` and resolved
``arm_joint_ids``. Each target has shape ``(num_envs, len(arm_joint_ids))``.
Set position or velocity feedback gains in the actuator configuration; for direct
effort control, set actuator stiffness and damping to zero. See
:ref:`actuators-runtime-api` for the command API and :doc:`actuators` for gains.

.. tab-set::

   .. tab-item:: Position

      .. code-block:: python

         joint_pos_target = robot.data.default_joint_pos.torch[:, arm_joint_ids].clone()
         robot.actuators.target_command.set_position_index(value=joint_pos_target, joint_ids=arm_joint_ids)
         robot.write_data_to_sim()

   .. tab-item:: Velocity

      .. code-block:: python

         joint_vel_target = robot.data.default_joint_vel.torch[:, arm_joint_ids].clone()
         robot.actuators.target_command.set_velocity_index(value=joint_vel_target, joint_ids=arm_joint_ids)
         robot.write_data_to_sim()

   .. tab-item:: Effort

      .. code-block:: python

         import torch

         joint_effort_target = torch.zeros_like(robot.data.joint_pos.torch[:, arm_joint_ids])
         robot.actuators.target_command.set_effort_index(value=joint_effort_target, joint_ids=arm_joint_ids)
         robot.write_data_to_sim()

These calls stage and submit commands. Advance physics and update the scene on
each cycle as shown in :ref:`controller-control-loop`.

Task-space goals
~~~~~~~~~~~~~~~~

The :doc:`IK walkthrough <../how-to/run_diff_ik>` resets the robot and changes the goal every
150 steps. On intermediate steps, it reads fresh state and computes joint position targets;
every step applies the targets and advances the simulation:

.. literalinclude:: ../../../scripts/tutorials/05_controllers/run_diff_ik.py
   :language: python
   :caption: scripts/tutorials/05_controllers/run_diff_ik.py
   :start-at: if count % 150 == 0:
   :end-at: scene.update(sim_dt)
   :dedent: 8

Relative commands also need the measured pose in
:meth:`DifferentialIKController.set_command` to resolve the delta at command time.

An :class:`OperationalSpaceController` instance ``osc`` similarly stores its
motion/force goal with :meth:`OperationalSpaceController.set_command`, but
:meth:`OperationalSpaceController.compute` returns efforts to stage with
``set_effort_index``. Its required state depends on the configured features.
Use the :doc:`IK walkthrough <../how-to/run_diff_ik>` and
:doc:`OSC walkthrough <../how-to/run_osc>` for controller construction, state
extraction, and complete repeated loops.


Reactive planners
-----------------

Task-space controllers usually leave joint limits and collision avoidance to a
higher-level planner, such as RRT. Reactive planners can instead include these
constraints in the optimization that computes motion and control.

Examples of reactive planners include:

-  **RMPFlow (lula):** An acceleration-based policy that composes various Reimannian Motion Policies (RMPs) to
   solve a hierarchy of tasks :cite:p:`cheng2021rmpflow`. It is capable of performing dynamic collision
   avoidance while navigating the end-effector to a target.

-  **MPC (OCS2):** A receding horizon control policy based on sequential linear-quadratic (SLQ) programming.
   It formulates various constraints into a single optimization problem via soft-penalties and uses automatic
   differentiation to compute derivatives of the system dynamics, constraints and costs. The MPC formulation
   for end-effector trajectory tracking in fixed-arm and mobile manipulators described in
   :cite:p:`mittal2021articulated` considers a kinematic system model with joint limits and self-collision avoidance.
   See the `OCS2 documentation <https://leggedrobotics.github.io/ocs2/>`_ for the external toolbox and examples.


.. warning::

    :class:`RmpFlowController` wraps the CPU-based Lula bindings, so its computation can
    limit throughput in large batches. It also needs the Lula library and robot description
    files; these dependencies are separate from the selected physics backend.
