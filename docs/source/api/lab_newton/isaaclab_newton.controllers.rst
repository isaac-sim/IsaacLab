isaaclab_newton.controllers
===========================

.. automodule:: isaaclab_newton.controllers

.. currentmodule:: isaaclab_newton.controllers

.. autosummary::
   :nosignatures:

   NewtonDifferentialIKController
   NewtonDifferentialIKControllerCfg
   NewtonJointImpedanceController
   NewtonJointImpedanceControllerCfg
   NewtonOperationalSpaceController
   NewtonOperationalSpaceControllerCfg

These controllers are separate from the :mod:`isaaclab.controllers` implementations. Their configurations map
one-to-one onto the model-free controllers in :mod:`newton.controllers`, including features the Isaac Lab
controllers do not have: null-space posture control for differential IK, Coriolis compensation and acceleration
feedforward for joint impedance, and separate linear and angular selection frames and a desired-twist target for
operational-space control. The caller supplies the Jacobian and dynamics, so they work with any physics backend.
They compute in float32 and bind contiguous float32 inputs without copies. Use
:class:`~isaaclab_newton.envs.mdp.NewtonDifferentialInverseKinematicsActionCfg` and
:class:`~isaaclab_newton.envs.mdp.NewtonOperationalSpaceControllerActionCfg` to drive them from an environment.

Differential Inverse Kinematics
-------------------------------

.. autoclass:: NewtonDifferentialIKController
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: NewtonDifferentialIKControllerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Joint Impedance
---------------

.. autoclass:: NewtonJointImpedanceController
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: NewtonJointImpedanceControllerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Operational Space
-----------------

.. autoclass:: NewtonOperationalSpaceController
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: NewtonOperationalSpaceControllerCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Inverse Kinematics
------------------

.. automodule:: isaaclab_newton.controllers.ik

.. currentmodule:: isaaclab_newton.controllers.ik

.. currentmodule:: isaaclab_newton.controllers.ik

.. autosummary::
   :nosignatures:

   NewtonIKJointLimitObjective
   NewtonIKJointLimitObjectiveCfg
   NewtonIKObjective
   NewtonIKObjectiveCfg
   NewtonIKPoseObjective
   NewtonIKPoseObjectiveCfg
   NewtonIKSolver
   NewtonIKSolverCfg

.. autoclass:: NewtonIKJointLimitObjective
   :show-inheritance:

.. autoclass:: NewtonIKJointLimitObjectiveCfg
   :show-inheritance:

.. autoclass:: NewtonIKObjective
   :show-inheritance:

.. autoclass:: NewtonIKObjectiveCfg
   :show-inheritance:

.. autoclass:: NewtonIKPoseObjective
   :show-inheritance:

.. autoclass:: NewtonIKPoseObjectiveCfg
   :show-inheritance:

.. autoclass:: NewtonIKSolver
   :show-inheritance:

.. autoclass:: NewtonIKSolverCfg
   :show-inheritance:
