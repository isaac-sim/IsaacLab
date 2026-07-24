Fixed
^^^^^

* Fixed :attr:`~isaaclab.scene.InteractiveSceneCfg.replicate_physics` being ignored since the
  replication-session refactor: scenes configured with ``replicate_physics=False`` invoked
  native physics replication anyway, silently discarding per-environment USD differences
  (e.g. prestartup scale randomization). Cloning is now USD-only in that case, so the physics
  engine parses each environment's USD prims directly; assets whose only cloning mechanism is
  physics replication are not cloned.

Added
^^^^^

* Added :attr:`~isaaclab.cloner.CloneCfg.replicate_physics` as the cloner-side home of the
  policy; :class:`~isaaclab.scene.InteractiveScene` pipes the scene flag into it and
  :func:`~isaaclab.cloner.replicate` applies it.
* Added :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` to override, per asset, the
  physics cloning contexts resolved at replication time; ``None`` uses the active backend's
  default physics context (``isaaclab_<backend>.cloner.PHYSICS_CONTEXT``). USD clones are
  added automatically under Kit and are not authored through this field.

Changed
^^^^^^^

* **Breaking:** Changed :data:`~isaaclab.cloner.REPLICATION_QUEUE` to hold asset cfgs instead
  of ``(cfg, BackendCtxCls)`` pairs: construction registers *which* cfgs participate via
  :func:`~isaaclab.cloner.queue_replication`, and *how* each is cloned resolves inside
  :func:`~isaaclab.cloner.replicate`. Code appending pairs should register the cfg and direct
  contexts through :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts`.

Removed
^^^^^^^

* Removed ``queue_usd_replication``: register the cfg with
  :func:`~isaaclab.cloner.queue_replication` and direct contexts through the cfg's
  ``cloning_contexts`` field instead.
