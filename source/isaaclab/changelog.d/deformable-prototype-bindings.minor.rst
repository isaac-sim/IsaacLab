Added
^^^^^

* Retained declared asset paths in :attr:`isaaclab.cloner.ClonePlan.asset_paths` so backend importers
  could resolve individual assets without runtime asset registration or stage discovery.
  Both config-based planners populated these paths automatically, including assets without spawners.
