* Added Gaussian splat visuals that follow deformables on OVRTX, in legacy and OVStage modes. The renderer writes
  splat centers to ``positions`` and other visuals to ``points``, and repeats the first OVStage writes, which OVStage
  drops, so that a paused simulation still applies them.
