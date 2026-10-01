* Moved the OVRTX renderer's ovstage clone sequence (the clone-plan copies and the environment-root
  transform write) into :func:`~isaaclab_ov.cloner.ovstage_replicate`, backed by
  :func:`~isaaclab_ov.renderers.ovrtx_usd.iter_clone_copies` and
  :func:`~isaaclab_ov.renderers.ovrtx_usd.env_root_transforms`, so the renderer and
  :class:`~isaaclab_ov.cloner.OvstageReplicateContext` share one implementation. ``iter_clone_copies`` now
  accepts the routed asset ids. Cloning behavior is unchanged.
