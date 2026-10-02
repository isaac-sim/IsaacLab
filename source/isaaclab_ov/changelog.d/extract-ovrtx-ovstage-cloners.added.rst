* Added :func:`~isaaclab_ov.cloner.ovrtx_replicate` and :func:`~isaaclab_ov.cloner.ovstage_replicate` to apply
  prepared subtree copies and environment positions to either native scene without interpreting a clone plan.
* Added :class:`~isaaclab_ov.cloner.OvrtxReplicateContext` and :class:`~isaaclab_ov.cloner.OvstageReplicateContext`
  with shared subtree preparation and separate routing to internal OVRTX scenes and simulation-owned stages.
* Added :class:`~isaaclab_ov.stage.OvstageBackend` to own detached stages, path dictionaries, population domains,
  stage cloning, and write ordinals independently of consumers. With ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``,
  OVRTX borrowed an isolated rendering stage while OVPhysX retained native cloning.
