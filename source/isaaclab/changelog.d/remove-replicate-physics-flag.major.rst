Removed
^^^^^^^

* **Breaking:** Removed ``InteractiveSceneCfg.replicate_physics`` and ``CloneCfg.replicate_physics``.
  Physics is now always replicated from one environment across the rest; the ``False`` path (USD-only
  cloning, with the physics engine parsing each environment's prims directly) is no longer supported.
  This path was already unsupported on the Newton physics backend and was a source of recurring
  "flat builder" bugs (environment-index vs. world-index conflation) across the Newton stack.

  Declare heterogeneous per-environment assets through :attr:`~isaaclab.scene.InteractiveSceneCfg.clone_cfg`
  (:class:`~isaaclab.cloner.CloneCfg.clone_combinations`) instead of opting out of replication; this
  mechanism already composes heterogeneous worlds under full replication and does not depend on the
  removed flag. :func:`~isaaclab.cloner.replicate` and :class:`~isaaclab.cloner.ReplicateSession` no
  longer accept a ``replicate_physics`` argument.

  USD-level ``"prestartup"``-mode randomization (scale, visual color, visual texture material) no
  longer requires or validates ``replicate_physics=False``; its interaction with always-replicated
  scenes needs further verification for scenarios that previously relied on non-replicated per-environment
  USD edits (see the follow-up note in the PR description).
