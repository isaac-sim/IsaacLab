* Changed the Newton RTX visualizer to render the simulation-owned OVStage through ``ViewerRTX(ovstage=...)``,
  preserving authored USD materials and lights while Newton drives body poses. No opt-in is required.
  Deprecated ``rtx_environment``; configure lights in the simulation scene instead. Runtime visual-material
  randomization still updates only the Newton model and does not change the rendered stage.
