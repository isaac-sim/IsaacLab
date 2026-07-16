Added
^^^^^

* Added :data:`~isaaclab_ovphysx.cloner.REPLICATION`, the backend's default replication stack
  referenced by asset cfgs: the clone replay authors USD itself, so it replicates alone. With
  physics replication disabled, OvPhysX assets are not cloned.

Removed
^^^^^^^

* Removed ``queue_ovphysx_replication``: direct the contexts through
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` and
  :func:`~isaaclab.cloner.queue_replication` instead.
