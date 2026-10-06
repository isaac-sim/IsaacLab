* Fixed the Shadow Hand ``randomized`` preset setting a restitution of 1.0 on the hand and the cube. The material
  term samples absolute values, so the cube bounced off the hand on PhysX and OvPhysX, which slowed or stalled
  learning; the preset now uses 0.0, the value without randomization. Newton is unaffected, as its MuJoCo solver does
  not use restitution.
