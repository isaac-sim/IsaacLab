Added
^^^^^

* Added :class:`~isaaclab_assets.robots.menagerie.MenageriePatchedUsdFileCfg` and
  :func:`~isaaclab_assets.robots.menagerie.patch_menagerie_asset`, which fix Mujoco
  Menagerie conversion defects by writing them into a patched copy of the asset's USD
  layers on disk (cached under ``~/.cache/isaaclab/menagerie_patched/``) instead of
  authoring prims at spawn time. Each fix is detection-first and idempotent -- kept
  drives alive in the ``mujoco`` variant, removed the converter's ``MjcActuator`` prims
  so the drives are the single actuation source (Newton otherwise builds a second,
  never-commanded servo per joint that drags it toward zero), unauthored static
  friction, collision pairs manufactured by welded-body splits, ``physx``-layer joint
  velocity limits, and the Shadow hand's fixed tendons -- and a
  ``isaaclabMenageriePatchVersion`` marker on the
  entry layer short-circuits repeated patching. The fix parameters live in a per-asset
  recipe registry keyed by the asset directory, so every physics variant and robot
  configuration referencing an asset shares one declaration. Assets resolve through the
  Isaac asset root via :obj:`~isaaclab_assets.robots.menagerie.MENAGERIE_ASSET_ROOT`
  (the conversions are currently staged on the ``isaac-dev`` Nucleus server pending
  production publication; override with the ``MENAGERIE_ASSET_ROOT`` environment
  variable to use a local mirror).
* Added :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_MENAGERIE_CFG` and
  :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_MENAGERIE_PHYSX_CFG` for the
  Mujoco Menagerie Shadow Hand conversion.
* Added :obj:`~isaaclab_assets.robots.allegro.ALLEGRO_HAND_MENAGERIE_CFG` for the Mujoco
  Menagerie Allegro Hand conversion, derived from :obj:`~isaaclab_assets.robots.allegro.ALLEGRO_HAND_CFG`
  so it expresses only the asset source, spawn-frame compensation, and joint naming.
  Physical parameters the asset can express come from the patched asset itself: it
  authors the motor rotor inertia the MJCF omits (``newton:armature`` and
  ``physxJoint:armature`` on every revolute joint); without it the stock hand's
  lightweight links leave the joint-space inertia near zero and contact dynamics are
  too noisy to train on (reorient success 0.16 vs 1.0 at 1500 iterations).
