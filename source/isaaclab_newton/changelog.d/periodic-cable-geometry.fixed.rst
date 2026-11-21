* Fixed geometry synchronization for imported periodic cables so closed curves follow their simulated capsule segments, including the closing seam.
  Cable bindings retain lists of capsule shape indices for open cables and accept ``(indices, closed)`` tuples for explicit topology.
