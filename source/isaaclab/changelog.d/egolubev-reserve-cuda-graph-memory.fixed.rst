* Fixed multi-minute stalls on Windows in the first CUDA graph capture after the renderer's first frame,
  seen with Newton physics under OVRTX and under Kit. :class:`~isaaclab.sim.SimulationContext` now reserves
  CUDA graph memory when it is created, before the CUDA driver's address placement can be blocked.
