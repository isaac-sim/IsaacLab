* Fixed Newton-native actuators driving only the first asset of a multi-asset scene. ``NewtonActuator`` prims
  were authored on the first prim matching the articulation path, so every other spawned asset
  imported into the Newton model without actuators and its joints received no torque. They are now authored on
  every matching articulation.
