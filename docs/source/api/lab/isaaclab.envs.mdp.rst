isaaclab.envs.mdp
=================

.. automodule:: isaaclab.envs.mdp

Observations
------------

.. automodule:: isaaclab.envs.mdp.observations
    :members:

Actions
-------

.. important::

    ``PinkInverseKinematicsAction`` is supported only on Linux. Tasks configured with this action cannot run on
    Windows because Pink IK and its dependencies are not available there.

.. automodule:: isaaclab.envs.mdp.actions

.. automodule:: isaaclab.envs.mdp.actions.actions_cfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Events
------

.. automodule:: isaaclab.envs.mdp.events
    :members:

.. automodule:: isaaclab.envs.mdp.visual_events
    :members:

Commands
--------

.. automodule:: isaaclab.envs.mdp.commands

.. automodule:: isaaclab.envs.mdp.commands.commands_cfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Rewards
-------

.. automodule:: isaaclab.envs.mdp.rewards
    :members:

Terminations
------------

.. automodule:: isaaclab.envs.mdp.terminations
    :members:

Curriculum
----------

.. automodule:: isaaclab.envs.mdp.curriculums
    :members:

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.envs.mdp` API.

.. currentmodule:: isaaclab.envs.mdp

.. autosummary::
   :nosignatures:

   AbsBinaryJointPositionAction
   ActionStateRecorderManagerCfg
   BinaryJointAction
   BinaryJointPositionAction
   BinaryJointVelocityAction
   EMAJointPositionToLimitsAction
   FixedTendonPositionAction
   InitialStateRecorder
   InitialStateRecorderCfg
   JointAction
   JointEffortAction
   JointPositionAction
   JointPositionToLimitsAction
   JointVelocityAction
   NonHolonomicAction
   NormalVelocityCommand
   NullCommand
   PostStepProcessedActionsRecorder
   PostStepProcessedActionsRecorderCfg
   PostStepStatesRecorder
   PostStepStatesRecorderCfg
   PreStepActionsRecorder
   PreStepActionsRecorderCfg
   PreStepFlatPolicyObservationsRecorder
   PreStepFlatPolicyObservationsRecorderCfg
   RelativeJointPositionAction
   SurfaceGripperBinaryAction
   TerrainBasedPose2dCommand
   UniformPose2dCommand
   UniformPoseCommand
   UniformVelocityCommand

.. autoclass:: AbsBinaryJointPositionAction
   :show-inheritance:

.. autoclass:: ActionStateRecorderManagerCfg
   :show-inheritance:

.. autoclass:: BinaryJointAction
   :show-inheritance:

.. autoclass:: BinaryJointPositionAction
   :show-inheritance:

.. autoclass:: BinaryJointVelocityAction
   :show-inheritance:

.. autoclass:: EMAJointPositionToLimitsAction
   :show-inheritance:

.. autoclass:: FixedTendonPositionAction
   :show-inheritance:

.. autoclass:: InitialStateRecorder
   :show-inheritance:

.. autoclass:: InitialStateRecorderCfg
   :show-inheritance:

.. autoclass:: JointAction
   :show-inheritance:

.. autoclass:: JointEffortAction
   :show-inheritance:

.. autoclass:: JointPositionAction
   :show-inheritance:

.. autoclass:: JointPositionToLimitsAction
   :show-inheritance:

.. autoclass:: JointVelocityAction
   :show-inheritance:

.. autoclass:: NonHolonomicAction
   :show-inheritance:

.. autoclass:: NormalVelocityCommand
   :show-inheritance:

.. autoclass:: NullCommand
   :show-inheritance:

.. autoclass:: PostStepProcessedActionsRecorder
   :show-inheritance:

.. autoclass:: PostStepProcessedActionsRecorderCfg
   :show-inheritance:

.. autoclass:: PostStepStatesRecorder
   :show-inheritance:

.. autoclass:: PostStepStatesRecorderCfg
   :show-inheritance:

.. autoclass:: PreStepActionsRecorder
   :show-inheritance:

.. autoclass:: PreStepActionsRecorderCfg
   :show-inheritance:

.. autoclass:: PreStepFlatPolicyObservationsRecorder
   :show-inheritance:

.. autoclass:: PreStepFlatPolicyObservationsRecorderCfg
   :show-inheritance:

.. autoclass:: RelativeJointPositionAction
   :show-inheritance:

.. autoclass:: SurfaceGripperBinaryAction
   :show-inheritance:

.. autoclass:: TerrainBasedPose2dCommand
   :show-inheritance:

.. autoclass:: UniformPose2dCommand
   :show-inheritance:

.. autoclass:: UniformPoseCommand
   :show-inheritance:

.. autoclass:: UniformVelocityCommand
   :show-inheritance:

Physics event backend selection
-------------------------------

Configure physics randomization through the shared ``isaaclab.envs.mdp`` terms:

.. code-block:: python

    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import EventTermCfg

    gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="startup",
        params={
            "gravity_distribution_params": ((0.0, 0.0, -10.0), (0.0, 0.0, -9.0)),
            "operation": "abs",
        },
    )

Material, collider-offset, and gravity terms select the backend at construction from
the simulation's resolved physics configuration, including the ``physx`` auto selector.
Use the same terms for Newton, Isaac Sim PhysX, and OVPhysX; parameter translation is
handled internally.

See the shared terms for backend limits: scene-wide versus per-environment gravity,
PhysX material buckets, Newton's single friction coefficient, and Kamino's shared materials.

Backend handling in other terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``randomize_visual_color`` and ``randomize_visual_texture_material`` use the
  Isaac Sim Replicator implementation independently of the physics configuration.
  They require Kit and update all matched USD prims; ``env_ids`` does not restrict
  these updates. Kitless runtimes report this requirement at construction.
* ``randomize_visual_shape`` selects its implementation from the active renderers
  and visualizers. ``randomize_visual_material`` writes through ``RenderContext``.
* ``randomize_actuator_gains`` follows each actuator group's owner. Newton
  controllers can run with PhysX, so physics selection does not determine gain storage.
* Mass, inertia, center-of-mass, joint, tendon, reset, and wrench terms use shared
  asset APIs. Observations, rewards, terminations, and actions use shared asset and
  sensor APIs. Their backend handling stays in those APIs.
* ``randomize_rigid_body_scale`` edits USD before simulation starts. The USD stage
  is shared across physics backends.
