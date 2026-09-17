isaaclab.controllers
====================

.. automodule:: isaaclab.controllers

  .. rubric:: Classes

  .. autosummary::

    DifferentialIKController
    DifferentialIKControllerCfg
    OperationalSpaceController
    OperationalSpaceControllerCfg
    pink_ik.PinkIKController
    pink_ik.PinkIKControllerCfg
    pink_ik.NullSpacePostureTask

Newton controller integration
-----------------------------

``DifferentialIKController``, ``JointImpedanceController``, and ``OperationalSpaceController``
retain their original Torch implementations by default. Set ``use_newton=True`` in the controller
configuration to select Newton's model-free solver. Both choices use the same constructor,
command and compute APIs and return independent result tensors. The controller choice is independent
of the physics backend: a Newton controller can also run with PhysX simulation.

Newton allocates persistent float32 workspace on the first ``compute()`` call. DiffIK and OSC infer
the number of controlled joints from the input tensors and rebuild the workspace if it changes;
callers do not specify a joint count. For CUDA capture, supply commands and limits and warm up
``compute()`` before capture. Recapture after changing the joint count, first enabling joint-limit
avoidance, or changing captured control flow. DiffIK retains ``set_joint_pos_limits()`` before or
after initialization. Gravity compensation can still be enabled or disabled between compute calls.

Newton 1.6.0 is required for the opt-in path. Float64 inputs do not provide float64 Newton solver
precision; leave ``use_newton=False`` when double-precision control laws are required.

Operational-space migration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``use_newton=True``, motion-axis selection precedes inertia decoupling. Hybrid force/motion
tasks that select only some axes must revalidate tracking and contact-force gains. The default
Torch path retains its original post-inertia selection. Newton inertia decoupling requires at least
six controlled joints; disable ``inertial_dynamics_decoupling`` for under-actuated arms. Newton
mass-weights null-space posture efforts only when inertia decoupling is enabled, unlike the
original implementation when a mass matrix is supplied without decoupling. Existing gains and
checkpoints therefore remain on the default path until separately validated with Newton.

SO-101 retains its original wrist-only orientation mask on the default path. With Newton selected,
it applies the mask to a copy of the Jacobian before the Newton solve.

Differential Inverse Kinematics
-------------------------------

.. autoclass:: DifferentialIKController
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DifferentialIKControllerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Operational Space controllers
-----------------------------

.. autoclass:: OperationalSpaceController
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: OperationalSpaceControllerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type


Pink IK Controller
------------------

.. note::

   The standard Isaac Lab installation provides Pink IK dependencies only on Linux x86_64 and aarch64.
   Pink IK requires ``pin`` (Pinocchio), ``pin-pink``, and ``daqp``. The Windows uv/pip installation does not
   provide Pinocchio, so Pink IK tasks cannot run with that installation. This is an installation limitation;
   upstream Pinocchio supports Windows through other distribution methods.

.. automodule:: isaaclab.controllers.pink_ik

.. autoclass:: PinkIKController
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: PinkIKControllerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Available Pink IK Tasks
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NullSpacePostureTask

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.controllers.pink_ik` API.

.. currentmodule:: isaaclab.controllers.pink_ik

.. autosummary::
   :nosignatures:

   DampingTask
   DampingTaskCfg
   FrameTask
   FrameTaskCfg
   LocalFrameTask
   LocalFrameTaskCfg
   NullSpacePostureTaskCfg
   PinkIKTaskCfg
   PinkKinematicsConfiguration

.. autoclass:: DampingTask
   :show-inheritance:

.. autoclass:: DampingTaskCfg
   :show-inheritance:

.. autoclass:: FrameTask
   :show-inheritance:

.. autoclass:: FrameTaskCfg
   :show-inheritance:

.. autoclass:: LocalFrameTask
   :show-inheritance:

.. autoclass:: LocalFrameTaskCfg
   :show-inheritance:

.. autoclass:: NullSpacePostureTaskCfg
   :show-inheritance:

.. autoclass:: PinkIKTaskCfg
   :show-inheritance:

.. autoclass:: PinkKinematicsConfiguration
   :show-inheritance:
