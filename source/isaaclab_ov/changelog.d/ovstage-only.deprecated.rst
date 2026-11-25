* Deprecated :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.async_rendering`, which has no effect now that OVRTX
  renders synchronously through OvStage; setting it to ``True`` emits a :class:`DeprecationWarning`.
* Deprecated :func:`~isaaclab_ov.cloner.ovrtx_replicate` in favor of :func:`~isaaclab_ov.cloner.ovstage_replicate`.
