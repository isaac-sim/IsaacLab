Added
^^^^^

* Added :class:`~isaaclab_assets.robots.menagerie.MenagerieUsdFileCfg`, a load-time
  adapter for Mujoco Menagerie asset conversions that fixes conversion defects on the
  composed prims before the physics engine parses them (unauthored static friction,
  collision pairs manufactured by welded-body splits, per-asset fixups such as PhysX
  fixed-tendon authoring). Each fix is tagged with the upstream converter change that
  makes it removable. Assets resolve from the public production release via
  :obj:`~isaaclab_assets.robots.menagerie.MENAGERIE_ASSET_ROOT` (no Nucleus
  authentication required; override with the ``MENAGERIE_ASSET_ROOT`` environment
  variable for local mirrors).
* Added :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_MENAGERIE_CFG` and
  :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_MENAGERIE_PHYSX_CFG` for the
  Mujoco Menagerie Shadow Hand conversion.
* Added :obj:`~isaaclab_assets.robots.allegro.ALLEGRO_HAND_MENAGERIE_CFG` for the Mujoco
  Menagerie Allegro Hand conversion.
