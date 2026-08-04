Changelog
---------

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
