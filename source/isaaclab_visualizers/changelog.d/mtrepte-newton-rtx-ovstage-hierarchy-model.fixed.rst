* Fixed the Newton RTX visualizer freezing on Newton 1.6.1 with OVStage 0.2, where body motion was not
  reflected in rendered frames. The viewer now holds an OVStage stage using Isaac Lab's hierarchy computation
  model so the stage Newton creates inherits it.
