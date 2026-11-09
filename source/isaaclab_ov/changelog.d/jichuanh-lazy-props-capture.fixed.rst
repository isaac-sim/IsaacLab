* Fixed lazily computed asset data properties replaying the value of the recording step from a CUDA
  graph recorded while they were current. A recording now holds their update, so every replay
  recomputes them.
