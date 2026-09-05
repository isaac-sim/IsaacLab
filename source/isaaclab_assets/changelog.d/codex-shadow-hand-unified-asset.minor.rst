Changed
^^^^^^^

* Changed the Shadow Hand configurations to spawn one asset and select the physics engine through
  its ``Physics`` USD variant, replacing the two separate PhysX and Newton assets whose joints were
  named differently. Both engines now spawn the hand at the same orientation; the previous assets
  needed two, because one baked a root orientation that the other did not.

* Changed ``SHADOW_HAND_CFG`` to spawn the asset with its default ``Physics`` variant. Use
  ``SHADOW_HAND_PHYSX_CFG`` or ``SHADOW_HAND_NEWTON_CFG`` to select an engine explicitly.

Added
^^^^^

* Added ``JOINT_NAMES``, ``TENDON_NAMES``,
  ``TENDON_POSITION_LIMITS`` and ``FINGERTIP_NAMES`` to
  :mod:`~isaaclab_assets.robots.shadow_hand`, so a task can name the hand's sixteen joint-driving
  motors, its four tendon-driving motors and their commandable range without restating them.
  ``SHADOW_HAND_PHYSX_CFG`` and ``SHADOW_HAND_NEWTON_CFG`` select the PhysX and Newton variants.

Removed
^^^^^^^

* Removed ``SHADOW_ACTUATED_JOINT_NAMES``; use ``JOINT_NAMES`` for the sixteen joint-driving
  motors and ``TENDON_NAMES`` for the four tendon-driving ones. The removed list named all twenty
  motors, so code that fed it to ``find_joints`` was asking for four joints that do not exist.

* Removed ``SHADOW_FINGERTIP_BODY_NAMES``; use ``FINGERTIP_NAMES``.

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
