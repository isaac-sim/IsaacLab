Added
^^^^^

* Added :data:`~isaaclab_physx.cloner.PHYSICS_CONTEXT`, the backend's default physics
  replication context referenced by asset cfgs. USD clones for visuals are added
  automatically under Kit by :func:`~isaaclab.cloner.replicate`.

Removed
^^^^^^^

* Removed ``queue_physx_replication``: direct the contexts through
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` and
  :func:`~isaaclab.cloner.queue_replication` instead.
