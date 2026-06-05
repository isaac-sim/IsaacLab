Added
^^^^^

* Added the tendon schema-fragment markers
  :class:`~isaaclab.sim.schemas.FixedTendonFragment` and
  :class:`~isaaclab.sim.schemas.SpatialTendonFragment`, which type the spawner
  ``fixed_tendons_props`` / ``spatial_tendons_props`` slots.
* Added :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties` and
  :func:`~isaaclab.sim.schemas.apply_spatial_tendon_properties`, the family writers that
  dispatch a list of tendon fragments via each fragment's ``func``. Tendons are a
  *tune-not-apply* family, so the writers tune the existing multi-instance tendon schemas
  without applying a new anchor schema.

Changed
^^^^^^^

* Changed the spawner ``fixed_tendons_props`` / ``spatial_tendons_props`` slots
  (:attr:`~isaaclab.sim.spawners.from_files.FileCfg.fixed_tendons_props`,
  :attr:`~isaaclab.sim.spawners.from_files.FileCfg.spatial_tendons_props`) to also accept one
  or more tendon fragments. Legacy single cfgs continue to work through a transition bridge in
  the spawn writer.
