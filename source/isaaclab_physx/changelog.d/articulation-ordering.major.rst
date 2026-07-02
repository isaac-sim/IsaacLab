Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_physx.assets.articulation.Articulation`.

Changed
^^^^^^^

* Changed :attr:`~isaaclab.assets.ArticulationData.body_mass` and
  :attr:`~isaaclab.assets.ArticulationData.body_inertia` to refresh from the
  simulation lazily, at most once per simulation step, instead of on every
  read. Values written directly through the tensor view become visible at the
  first read after the next simulation update; use the asset's
  :meth:`~isaaclab.assets.Articulation.set_masses_index` and
  :meth:`~isaaclab.assets.Articulation.set_inertias_index` for immediately
  coherent writes.
