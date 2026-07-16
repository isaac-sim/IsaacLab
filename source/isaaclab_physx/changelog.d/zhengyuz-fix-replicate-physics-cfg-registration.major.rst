Added
^^^^^

* Added :data:`~isaaclab_physx.cloner.REPLICATION`, the backend's default replication stack
  referenced by asset cfgs: native physics replication plus USD clones for visuals.

Removed
^^^^^^^

* Removed ``queue_physx_replication``: direct the contexts through
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` and
  :func:`~isaaclab.cloner.queue_replication` instead.
