Fixed
^^^^^

* Fixed :func:`~isaaclab.cloner.replicate` skipping USD replication in kit-less runs, which left every
  spawner-only asset (such as an :class:`~isaaclab.assets.AssetBaseCfg` referencing a scene) authored under
  ``env_0`` alone. :class:`~isaaclab.cloner.UsdReplicateContext` replicates through ``pxr.Sdf`` and never
  needed Kit, so it is now added whenever the cfg has a spawner. A kit-less renderer that reads the stage,
  such as the OVRTX renderer, previously rendered empty tiles for every env past the first, and listing
  :class:`~isaaclab.cloner.UsdReplicateContext` in
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` is no longer required to work around it.
