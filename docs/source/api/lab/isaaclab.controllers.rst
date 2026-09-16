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
use Newton's model-free controller APIs. They accept Torch tensors independently of the simulation
backend. Isaac Lab resolves commands and gain schedules, copies inputs into persistent float32
buffers, and invokes Newton's ``step()`` method. Returned tensors are independent snapshots;
``DifferentialIKController.compute(out=...)`` can instead fill a caller-owned buffer.

The controller constructors retain their existing arguments. DiffIK and OSC infer the fixed
joint count on the first ``compute()`` call and initialize Newton once. Call ``set_joint_pos_limits()``
to supply or update DiffIK limits as before. When avoidance is configured, supplying limits for the
first time after a solve rebuilds Newton to enable that feature; ordinary limit updates do not.

For CUDA capture, warm up ``compute()`` after setting commands and any joint limits. Commands can
then update existing target storage. Recapture OSC after ``reset()`` and DiffIK after first enabling
joint-limit avoidance on an already initialized backend.

Newton 1.6.0 is required. Float64 inputs do not provide float64 solver precision. Applications that
require double-precision control laws must retain the previous implementation.

Operational-space migration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Motion-axis selection now precedes inertia decoupling. Hybrid force/motion tasks that select only
some axes must revalidate tracking and contact-force gains; the former post-inertia masking is not
available. Inertia decoupling requires at least six controlled joints. Set
``inertial_dynamics_decoupling=False`` for under-actuated arms. Null-space posture efforts are
mass-weighted only when inertia decoupling is enabled; retune posture gains if a task previously
passed a mass matrix while leaving decoupling disabled.

Newton applies orientation weights and computes differential-IK pose errors and solver updates.
The SO-101 controller masks orientation-Jacobian columns before calling the base controller.

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
