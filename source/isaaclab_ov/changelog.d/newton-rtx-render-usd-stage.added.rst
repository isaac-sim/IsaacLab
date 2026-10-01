* Added :class:`~isaaclab_ov.cloner.OvstageReplicateContext`, a clone context that builds an OVStage stage of the
  scene from the assets routed to it, so a renderer such as Newton's ``ViewerRTX`` can draw each environment
  with its authored materials. A consumer declares it through ``cloning_contexts`` and reads
  :attr:`~isaaclab_ov.cloner.OvstageReplicateContext.render_stage` after replication. It builds the stage with
  :func:`~isaaclab_ov.stage.create_render_ovstage` and requires OVStage 0.2 or newer.
