* Fixed the Newton IMU, PVA and frame transformer registering their sites again on a hard reset (a further
  ``sim.reset()``), which injected duplicate sites into the kept builder. Scenes with several environments failed
  on the second reset, and a single-environment ray caster could report another body's pose. Hard resets also drop
  the previous model's native IMU and frame transform sensors.
* Fixed the Newton frame transformer giving a target another body's name on a hard reset when a wildcard target
  comes before it.
