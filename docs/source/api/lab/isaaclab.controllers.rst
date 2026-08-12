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
