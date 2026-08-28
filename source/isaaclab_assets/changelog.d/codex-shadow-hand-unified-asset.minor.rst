Changed
^^^^^^^

* Changed the Shadow Hand configurations to spawn one asset and select the physics engine through
  its ``Physics`` USD variant, replacing the two separate PhysX and Newton assets whose joints were
  named differently. Both engines now spawn the hand at the same orientation; the previous assets
  needed two, because one baked a root orientation that the other did not.

Added
^^^^^

* Added ``JOINT_NAMES``, ``TENDON_NAMES``,
  ``TENDON_POSITION_LIMITS`` and ``FINGERTIP_NAMES`` to
  :mod:`~isaaclab_assets.robots.shadow_hand`, so a task can name the hand's sixteen joint-driving
  motors, its four tendon-driving motors and their commandable range without restating them.
  ``SHADOW_HAND_PHYSX_CFG`` and ``SHADOW_HAND_NEWTON_CFG`` select the PhysX and Newton variants.

Removed
^^^^^^^

* Removed ``SHADOW_ACTUATED_JOINT_NAMES``, ``SHADOW_TENDON_JOINT_NAMES`` and
  ``SHADOW_PHYSX_TENDON_GEARING``. Use ``JOINT_NAMES`` and
  ``TENDON_NAMES`` instead. Note that ``SHADOW_ACTUATED_JOINT_NAMES`` listed all
  twenty motors, so code that fed it to ``find_joints`` was asking for four joints that do not
  exist; ``JOINT_NAMES`` lists only the sixteen that drive a joint.

Fixed
^^^^^

* Fixed the Shadow Hand asset applying an articulation-root schema to two prims, which made any
  consumer that resolves the root by search fail with ``Expected 1 prims ... found 2`` once the
  asset was loaded in Kit. ``JointWrenchSensor`` hit this on every backend, so the manager-based
  reorientation environment could not start. The second schema carried one attribute, the Newton
  self-collision flag, which the configuration already supplies for both engines; removing both
  leaves a single articulation root.

* Removed configuration that restated the asset or its defaults: the joint drive type, which the
  asset authors on every joint, and ``soft_joint_pos_limit_factor`` and
  ``activate_contact_sensors``, which repeated their defaults. What remains differs between the two
  variants only in the selected USD variant and the PhysX solver settings, which Newton ignores.
