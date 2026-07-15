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
  entry layer short-circuits repeated patching. Assets resolve from the public
  production release via :obj:`~isaaclab_assets.robots.menagerie.MENAGERIE_ASSET_ROOT`
  (no Nucleus authentication required; a local mirror is copied and the S3 release is
  downloaded via an anonymous ``ListObjectsV2`` enumeration; override with the
  ``MENAGERIE_ASSET_ROOT`` environment variable).
* Added :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_MENAGERIE_CFG` and
  :obj:`~isaaclab_assets.robots.shadow_hand.SHADOW_HAND_MENAGERIE_PHYSX_CFG` for the
  Mujoco Menagerie Shadow Hand conversion.
* Added :obj:`~isaaclab_assets.robots.allegro.ALLEGRO_HAND_MENAGERIE_CFG` for the Mujoco
  Menagerie Allegro Hand conversion. The actuator configuration authors the motor rotor
  inertia (``armature``) that the MJCF omits; without it the stock hand's lightweight
  links leave the joint-space inertia near zero and Newton/MJWarp contact dynamics are
  too noisy to train on (reorient success 0.16 vs 1.0 at 1500 iterations).
