Added
^^^^^

* Added the Newton MJWarp physics preset to the ``IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs``
  task. The class is a sibling of :class:`PickPlaceGR1T2EnvCfg` rather than a subclass, so it never
  picked up the ``sim.physics`` assignment and ``physics=newton_mjwarp`` failed with
  ``Unknown preset(s)``. The scene-level Newton adjustments were already shared through
  ``ObjectTableSceneCfg``. Also aligned ``num_rerenders_on_reset`` with the fixed-waist task.
