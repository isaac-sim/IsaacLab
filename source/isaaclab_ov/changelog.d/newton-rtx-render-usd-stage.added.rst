* Added :func:`~isaaclab_ov.stage.create_render_ovstage`, which exports the simulation's USD stage into a
  new OVStage stage and clones it onto every environment from the clone plan, so a renderer such as
  Newton's ``ViewerRTX`` can draw each environment with its authored materials. It requires OVStage 0.2 or
  newer.
