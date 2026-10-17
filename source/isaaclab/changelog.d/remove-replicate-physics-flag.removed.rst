* **Breaking:** Removed the non-replicated clone path. The ``replicate_physics`` argument remains accepted by
  scene and clone configurations, ``cloner.replicate``, ``ReplicateSession``, and multi-asset spawners, but
  now warns and is ignored. Use ``CloneCfg.clone_combinations`` for heterogeneous environments; assets such
  as PhysX deformables can select a USD clone context when native physics replication is unsupported.
