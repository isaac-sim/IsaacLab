Changelog
---------

0.8.0 (2026-09-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``JOINT_NAMES``, ``TENDON_NAMES``,
  ``TENDON_POSITION_LIMITS`` and ``FINGERTIP_NAMES`` to
  :mod:`~isaaclab_assets.robots.shadow_hand`, so a task can name the hand's sixteen joint-driving
  motors, its four tendon-driving motors and their commandable range without restating them.
  ``SHADOW_HAND_PHYSX_CFG`` and ``SHADOW_HAND_NEWTON_CFG`` select the PhysX and Newton variants.

Changed
^^^^^^^

* Changed the Shadow Hand configurations to spawn one asset and select the physics engine through
  its ``Physics`` USD variant, replacing the two separate PhysX and Newton assets whose joints were
  named differently. Both engines now spawn the hand at the same orientation; the previous assets
  needed two, because one baked a root orientation that the other did not.

* Changed ``SHADOW_HAND_CFG`` to spawn the asset with its default ``Physics`` variant. Use
  ``SHADOW_HAND_PHYSX_CFG`` or ``SHADOW_HAND_NEWTON_CFG`` to select an engine explicitly.

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


0.7.0 (2026-09-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the Unitree Go1 and Go2 leg actuator limits ignoring the knee reduction. Both robots applied
  the hip and thigh limits to the calf joints, which capped calf torque well below its rated value and
  let the torque-speed curve keep motoring past its rated speed. The calf joints now use the limits
  authored in ``go1.usd`` and ``go2.usd`` (Go1: 35.55 N·m, 20.06 rad/s; Go2: 45.43 N·m, 15.70 rad/s),
  and the hip and thigh limits were aligned with the same assets (23.7 N·m, 30.1 rad/s).

  These robots now produce more calf torque at lower calf speeds, so policies trained on the previous
  configuration should be retrained rather than reused directly.


0.6.6 (2026-09-03)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* **Breaking:** Updated ``SO101_CFG`` to use the SysID-capable asset and resolve actuator gains, friction, armature,
  and limits from its default Newton MJWarp USD variant. The USD-authored actuator group is now named ``usd``. The
  config also uses the workshop operational joint pose, inherits root fixation from the USD, disables
  self-collisions, enables contact sensors, and applies a 0.98 soft joint-limit factor. Tasks that require the
  previous simulation gains should migrate to ``SO101_HIGH_PD_CFG``, which retains the prior high-PD actuator
  behavior.

Fixed
^^^^^

* Fixed ``SO101_CFG`` running convex decomposition instead of using the asset's authored convex hulls.


0.6.5 (2026-08-21)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed self-collisions being uncontrolled under Newton for
  :data:`~isaaclab_assets.robots.allegro.ALLEGRO_HAND_CFG`,
  :data:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_CFG`,
  :data:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_NEWTON_CFG`, and
  :data:`~isaaclab_assets.robots.kuka_allegro.KUKA_ALLEGRO_CFG`. Their ``articulation_props`` used
  the deprecated PhysX-only ``ArticulationRootPropertiesCfg``, which never authored the
  ``newton:selfCollisionEnabled`` attribute Newton's schema resolver checks. They now pass a
  ``PhysxArticulationCfg`` + ``NewtonArticulationCfg`` fragment pair so ``enabled_self_collisions``
  is authored on both backends explicitly.


0.6.4 (2026-08-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed prim path expressions to spell a single path segment ``[^/]`` rather than ``.``, so each
  pattern selects what it selected before now that ``.`` matches ``/`` in
  :func:`~isaaclab.sim.utils.find_matching_prims`.


0.6.3 (2026-08-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Updated the Menagerie Franka configuration to use its corrected USD-authored arm drive gains.


0.6.2 (2026-08-01)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the DR Legs feet colliding as bounding boxes by setting an explicit ``convexHull`` mesh
  approximation on :data:`~isaaclab_assets.robots.dr_legs.DR_LEGS_IMPLICIT_PD_CFG`.


0.6.1 (2026-07-30)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :data:`~isaaclab_assets.sensors.GELSIGHT_MINI_CFG` to use the available GelSight render data.


0.6.0 (2026-07-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab_assets.robots.franka.FRANKA_PANDA_MENAGERIE_CFG` for the
  MuJoCo Menagerie-derived Franka asset with cross-backend actuator overrides.
* Added ``SHADOW_HAND_NEWTON_CFG``, the Newton (MJWarp) Shadow Hand configuration,
  shared by the reorientation and handover tasks.


0.5.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the dexterous-hand actuated-joint and fingertip body-name lists
  (:obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_ACTUATED_JOINT_NAMES`,
  :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_FINGERTIP_BODY_NAMES`,
  :obj:`~isaaclab_assets.robots.allegro.ALLEGRO_ACTUATED_JOINT_NAMES`,
  :obj:`~isaaclab_assets.robots.allegro.ALLEGRO_FINGERTIP_BODY_NAMES`) to the
  robot asset modules so tasks can reference them from a single source.

Changed
^^^^^^^

* **Breaking:** Removed ``ISAACLAB_ASSETS_METADATA`` from :mod:`isaaclab_assets`.
  This constant was populated from the now-deleted ``config/extension.toml`` Kit extension manifest.
* Updated ``FRANKA_PANDA_CFG`` USD path to the new Nucleus location under
  ``Robots/FrankaEmika/Legacy/panda_instanceable.usd``.
* Changed :obj:`~isaaclab_assets.robots.franka.FRANKA_PANDA_CFG` to load the Franka Panda
  from its new ``Robots/FrankaEmika/Legacy/panda_instanceable.usd`` location, following the
  asset reorganization on the Nucleus server. The robot model itself is unchanged.
* Changed the :obj:`~isaaclab_assets.robots.kuka_allegro.KUKA_ALLEGRO_CFG` actuator
  parameters to identified values: per-joint effort limits, stiffness, damping, and armature
  derived from the iiwa7 and Allegro hand references (Drake models, Wonik Robotics
  datasheet), motor velocity limits for MDP checks, and gravity enabled on the rigid bodies.

Removed
^^^^^^^

* Removed ``config/extension.toml`` Kit extension manifest. Inter-package dependencies are now
  declared via PEP 508 ``file:`` references in ``[project.dependencies]`` of ``pyproject.toml``.

Fixed
^^^^^

* Fixed excessive simulation joint velocity limits in the DR Legs asset.


0.4.2 (2026-07-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :data:`~isaaclab_assets.robots.fourbar_pole.FOURBAR_POLE_CFG` for a parallel
  four-bar linkage with an inverted pendulum pole on the coupler.


0.4.1 (2026-07-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :data:`~isaaclab_assets.robots.dr_legs.DR_LEGS_IMPLICIT_PD_CFG` for the Disney DR Legs
  closed-loop biped.


0.4.0 (2026-06-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :obj:`~isaaclab_assets.robots.so101.SO101_CFG` and
  :obj:`~isaaclab_assets.robots.so101.SO101_HIGH_PD_CFG` configurations for the
  TheRobotStudio SO-101 5-DOF follower arm.


0.3.4 (2026-05-12)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_assets.robots.unitree.G129_CFG_WITH_DEX3_BASE_FIX` robot configuration
  for the Unitree G1 29-DOF with Dex3 hands.


0.3.3 (2026-04-29)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for Flexiv Rizon 4s with Grav parallel gripper for manipulation tasks.


0.3.2 (2026-04-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed Cassie failing to load on Newton by enabling
  :attr:`~isaaclab.sim.schemas.JointDrivePropertiesCfg.ensure_drives_exist`
  in :data:`~isaaclab_assets.robots.cassie.CASSIE_CFG`.


0.3.1 (2026-02-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Configuration for Flexiv Rizon 4s robot used for manipulation tasks.

0.3.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion ordering.

0.2.4 (2025-11-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Configuration for OpenArm robots used for manipulation tasks.

0.2.3 (2025-08-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Configuration for G1 robot used for locomanipulation tasks.

0.2.2 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for the Fourier GR1T2 robot.

0.2.1 (2025-01-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for the Humanoid-28 robot.


0.2.0 (2024-12-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Restructured the assets directory into ``robots`` and ``sensors`` subdirectories.


0.1.4 (2024-08-21)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration for the Inverted Double Pendulum on a Cart robot.


0.1.2 (2024-04-03)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configurations for different arms from Kinova Robotics and Rethink Robotics.


0.1.1 (2024-03-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configurations for allegro and shadow hand assets.


0.1.0 (2023-12-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Moved all assets' configuration from ``isaaclab`` to ``isaaclab_assets`` extension.
