Added
^^^^^

* Added :data:`~isaaclab_newton.cloner.PHYSICS_CONTEXT`, the backend's default physics
  replication context referenced by asset cfgs. USD clones accompany Newton replication only
  under Kit — added automatically by :func:`~isaaclab.cloner.replicate` — so headless runs
  skip the USD authoring cost without assets branching on Kit availability.

Removed
^^^^^^^

* Removed ``queue_newton_physics_replication``: direct the contexts through
  :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` and
  :func:`~isaaclab.cloner.queue_replication` instead.
