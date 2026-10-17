* Updated the shared NIST task geometry, assembly profiles, contact screening, and reset-state restoration.
  Kept the existing ``IsaacContrib-Factory-Franka`` registration and ``FactoryEnvCfg`` entry point.
  Training used full gravity and four physics steps per action; playback sampled uniformly from random starts.
  Robot, action, observation, and reset defaults changed; existing checkpoints are not guaranteed to work with
  these defaults. To replay an existing checkpoint, restore its saved environment and agent configuration,
  including the robot asset and observation ordering, rather than using the new defaults directly.
