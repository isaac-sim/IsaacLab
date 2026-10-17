* **Breaking:** Removed ``InteractiveSceneCfg.replicate_physics`` and ``CloneCfg.replicate_physics``.
  Scenes always use a clone plan. Use ``CloneCfg.clone_combinations`` for heterogeneous environments;
  assets such as PhysX deformables can select a USD clone context when native physics replication is unsupported.
  ``cloner.replicate``, ``ReplicateSession``, ``spawn_multi_asset``, and ``spawn_multi_usd_file`` no longer
  accept a ``replicate_physics`` argument.
