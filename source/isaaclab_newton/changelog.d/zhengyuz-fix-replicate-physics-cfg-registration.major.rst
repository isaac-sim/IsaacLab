Added
^^^^^

* Added :data:`~isaaclab_newton.cloner.REPLICATION`, the backend's default replication stack
  referenced by asset cfgs: USD clones accompany Newton replication only under Kit, so
  headless runs skip the USD authoring cost without assets branching on Kit availability.

Removed
^^^^^^^

* Removed ``queue_newton_physics_replication``: direct the contexts through
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` and
  :func:`~isaaclab.cloner.queue_replication` instead.
