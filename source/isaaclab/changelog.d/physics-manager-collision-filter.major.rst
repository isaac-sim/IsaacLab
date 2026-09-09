Added
^^^^^

* Added manager-owned collision groups with whole-path regular-expression selectors and per-group deny or allow lists.

Changed
^^^^^^^

* Made replicated-environment isolation a default cloner rule stored in :class:`~isaaclab.cloner.ClonePlan`;
  set ``CloneCfg.isolate_environments=False`` to allow cross-environment contact.
* Added ``CloneCfg`` to the plan builders and :class:`isaaclab.cloner.ReplicateSession`; legacy options remain
  accepted, and :func:`isaaclab.cloner.replicate` retains its ``replicate_physics`` override.
* Deprecated, but retained, ``AssetBaseCfg.collision_group``, ``TerrainImporterCfg.collision_group``,
  ``InteractiveSceneCfg.replicate_physics``, and :func:`isaaclab.cloner.filter_collisions`.

Removed
^^^^^^^

* Removed ``InteractiveSceneCfg.filter_collisions`` and ``InteractiveScene.filter_collisions``; configure
  isolation through ``InteractiveSceneCfg.clone_cfg``.
