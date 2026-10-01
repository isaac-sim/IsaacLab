* Added :attr:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg.render_usd_stage`. When enabled, the
  config declares :class:`~isaaclab_ov.cloner.OvstageReplicateContext` in its ``cloning_contexts`` and the Newton
  RTX visualizer draws the stage it builds through ``ViewerRTX(ovstage=...)`` instead of a scene rebuilt from the
  Newton model, so MDL materials such as glass stay visible. It is off by default and needs a Newton release
  whose ``ViewerRTX`` accepts ``ovstage=`` together with OVRTX 0.5 and OVStage 0.2.
