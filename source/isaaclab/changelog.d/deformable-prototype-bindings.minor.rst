Added
^^^^^

* Retained asset and sensor declarations in :attr:`isaaclab.cloner.ClonePlan.cfgs` so backend importers
  could resolve individual assets without runtime asset registration or stage discovery.
  Both config-based planners retained these declarations, including shared assets and assets without spawners.

* Added :func:`isaaclab.sim.utils.queries.find_cloned_prim_paths` to expand matching prototype
  prims into exact destination paths without inspecting generated clones.
