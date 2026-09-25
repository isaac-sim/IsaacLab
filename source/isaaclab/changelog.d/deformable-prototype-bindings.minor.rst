Added
^^^^^

* Retained asset and sensor declarations in :attr:`isaaclab.cloner.ClonePlan.cfgs` so backend importers
  could resolve individual assets without runtime asset registration or stage discovery.
  Both config-based planners retained these declarations, including shared assets and assets without spawners.

* Added :func:`isaaclab.sim.utils.queries.find_cloned_prim_paths` to expand matching prototype
  prims into exact destination paths without inspecting generated clones.

Changed
^^^^^^^

* **Breaking:** Renamed ``ClonePlan.cfg_rows`` to ``cfg_source_indices`` and ``ClonePlan.context_rows``
  to ``context_source_indices``. Update attribute access and constructor keywords to the new names;
  both mappings still contain indices into ``ClonePlan.sources``.
