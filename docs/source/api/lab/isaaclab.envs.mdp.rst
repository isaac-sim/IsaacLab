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

Explicit physics event selection
--------------------------------

Physics randomization with backend-specific semantics is configured through
``EventTermCfg.func``. Mass, inertia, center-of-mass, joint, and actuator terms
remain in ``isaaclab.envs.mdp`` because they use the shared asset APIs.

Select the complete event configuration with the same concrete preset name used
for physics. For example, a task whose physics presets are ``newton_mjwarp`` and
``ovphysx`` can define:

.. code-block:: python

    from isaaclab.managers import EventTermCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import PresetCfg


    @configclass
    class GravityEventCfg(PresetCfg):
        newton_mjwarp = EventTermCfg(
            func="isaaclab_newton.envs.mdp:randomize_world_gravity",
            mode="startup",
            params={
                "gravity_distribution_params": ((0.0, 0.0, -10.0), (0.0, 0.0, -9.0)),
                "operation": "abs",
            },
        )
        ovphysx = newton_mjwarp.replace(
            func="isaaclab_ov.envs.mdp:randomize_physics_scene_gravity",
        )
        default = newton_mjwarp


    @configclass
    class EventsCfg:
        gravity: GravityEventCfg = GravityEventCfg()

The existing preset resolver selects matching fields throughout the task config.
Select callable and parameters together when the backend signatures differ.
Use concrete physics selectors when composing these events; the ``physx`` auto
selector can resolve to either Isaac Sim PhysX or OVPhysX only at launch. Existing
configurations using that auto selector can retain the deprecated core terms
during migration.

The contracts are deliberately explicit:

* ``isaaclab_physx.envs.mdp.randomize_physics_scene_gravity`` and its
  ``isaaclab_ov`` counterpart sample one global vector from configured gravity.
  Environment selectors cannot restrict a global change.
* ``isaaclab_newton.envs.mdp.randomize_world_gravity`` samples selected environment
  worlds and leaves the trailing global world unchanged. ``add`` and ``scale``
  operate on current values and therefore accumulate across calls.
* PhysX and OVPhysX material terms accept static/dynamic friction, restitution,
  and material buckets. Newton's material term accepts ``friction_range`` and
  ``restitution_range``. Kamino shares materials across worlds; its material
  randomization changes every environment even when a subset is supplied.
* Newton's ``randomize_rigid_body_collider_parameters`` samples native margin
  and gap independently, in meters. The PhysX/OVPhysX collider terms accept rest
  and contact offsets. Collider terms operate on every shape of the selected
  asset environments, without body filtering.

The three original core entry points remain available and emit
``DeprecationWarning`` at construction. They preserve the old signatures and
backend behavior, including Newton's ignored PhysX material parameters, cached
material ranges, and the conversion ``gap = max(contact_offset - margin, 0)``.
For that conversion, migrate both distributions together; independently sampling
a gap does not reproduce a distribution of contact offsets minus sampled margins.

Event instances own their buckets, defaults, and selections. They borrow native
bindings from assets and the active physics manager. ``BackendCfg`` continues to
configure simulation-owned native resources; event terms neither register a new
resource nor allocate a second physics model. Solver notifications stay with the
active manager. Custom manager subclasses inherit the compatibility hooks without
class-name matching; new task configurations select the event directly.
