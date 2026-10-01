* Added :func:`~isaaclab_ov.cloner.ovrtx_replicate` and :func:`~isaaclab_ov.cloner.ovstage_replicate` to apply
  prepared subtree copies and environment positions to either native scene without interpreting a clone plan.
* Added :class:`~isaaclab_ov.cloner.OvRenderReplicateContext` to prepare native copies during clone dispatch
  for OVRTX scenes and simulation-owned OVStage resources.
* Added :class:`~isaaclab_ov.stage.OvstageBackend` as the shared owner of detached rendering stages and path
  dictionaries. OVRTX borrowed the stage; other viewers can use the same population and cloning API.
