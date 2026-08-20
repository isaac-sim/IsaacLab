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
