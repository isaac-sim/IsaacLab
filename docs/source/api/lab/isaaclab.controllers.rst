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

Newton controllers
------------------

The differential IK, joint impedance, and operational-space controllers use their original Torch
implementations by default. Set ``use_newton=True`` in the controller configuration to use Newton's
model-free solver instead, for example to compare the two. The constructor, command, and compute
APIs are unchanged, and the choice is independent of the physics backend.

The Newton path differs in a few ways:

* It computes in float32 and allocates its buffers on the first ``compute()`` call. Warm up before
  capturing CUDA graphs, and recapture if the joint count changes.
* Differential IK with a positive ``joint_limit_avoidance_gain`` requires ``set_joint_pos_limits()``
  before the first ``compute()``.
* Operational-space control applies motion-axis selection before inertia decoupling, so hybrid
  force/motion tasks need their gains revalidated. Inertia decoupling requires at least six
  controlled joints, and null-space posture efforts are mass-weighted only when inertia decoupling
  is enabled.

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
