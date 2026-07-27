isaaclab_newton.ik
==================

.. automodule:: isaaclab_newton.ik

Newton inverse kinematics
-------------------------

Newton IK expresses a solve as an ordered list of objectives. Pose objectives
consume action coordinates, while joint-limit and joint-posture objectives add
constraints without changing the action dimension.

The following manager-based action tracks a relative Franka end-effector pose,
respects the model's joint limits, and uses the robot's nominal configuration
as a low-weight preference in the redundant solution space:

.. code-block:: python

   from isaaclab_newton.envs.mdp import NewtonInverseKinematicsActionCfg
   from isaaclab_newton.ik import (
       NewtonIKJointLimitObjectiveCfg,
       NewtonIKJointPostureObjectiveCfg,
       NewtonIKPoseObjectiveCfg,
   )

   arm_action = NewtonInverseKinematicsActionCfg(
       asset_name="robot",
       joint_names=["panda_joint[1-7]"],
       objectives=[
           NewtonIKPoseObjectiveCfg(
               body_name="panda_hand",
               body_offset_pos=(0.0, 0.0, 0.107),
               command_type="pose",
               use_relative_mode=True,
               scale=0.2,
           ),
           NewtonIKJointLimitObjectiveCfg(weight=0.1),
           NewtonIKJointPostureObjectiveCfg(
               joint_names=[f"panda_joint{i}" for i in range(1, 8)],
               target_positions=(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741),
               weight=0.01,
           ),
       ],
   )

Posture-objective joint names are exact names rather than regular expressions.
Set :attr:`~isaaclab_newton.ik.NewtonIKJointPostureObjectiveCfg.target_positions`
to the robot's nominal configuration, or leave it as ``None`` to use positions
from the finalized Newton prototype.

Manager-based action
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: isaaclab_newton.envs.mdp

.. autoclass:: NewtonInverseKinematicsActionCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: NewtonInverseKinematicsAction
   :members:
   :show-inheritance:

Solver
~~~~~~

.. currentmodule:: isaaclab_newton.ik

.. autoclass:: NewtonIKSolverCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: NewtonIKSolver
   :members:
   :show-inheritance:

Objectives
~~~~~~~~~~

.. autoclass:: NewtonIKPoseObjectiveCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: NewtonIKJointLimitObjectiveCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: NewtonIKJointPostureObjectiveCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__
