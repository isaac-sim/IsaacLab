* Added :attr:`~isaaclab_visualizers.newton.NewtonRTXVisualizerCfg.render_usd_stage`. When enabled, the
  Newton RTX visualizer draws the simulation's own USD stage through ``ViewerRTX(ovstage=...)`` instead of a
  scene rebuilt from the Newton model, so MDL materials such as glass stay visible. It is off by default and
  needs a Newton release whose ``ViewerRTX`` accepts ``ovstage=`` together with OVRTX 0.5 and OVStage 0.2.
