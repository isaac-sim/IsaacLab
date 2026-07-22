Added
^^^^^

* Added :data:`~isaaclab_ovphysx.cloner.PHYSICS_CONTEXT`, the backend's default physics
  replication context referenced by asset cfgs: its clone replay authors USD itself, so it
  replicates alone. With physics replication disabled, OvPhysX assets are not cloned.

Removed
^^^^^^^

* Removed ``queue_ovphysx_replication``: direct the contexts through
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` and
  :func:`~isaaclab.cloner.queue_replication` instead.
