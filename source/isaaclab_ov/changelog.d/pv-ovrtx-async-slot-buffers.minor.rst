Added
^^^^^

* Added an opt-in asynchronous OVRTX render path controlled by
  :attr:`~isaaclab.renderers.RendererCfg.async_rendering`, the only implementation of that setting.
  ``True`` trades one frame of camera latency for pipelined rendering: rendering then overlaps the
  next step's simulation and Python work, improving throughput, and camera outputs are one step
  stale. The first frame after (re)initialization is consumed immediately so the first camera read
  returns a valid frame; later frames are pipelined. Available on the legacy scene-ownership path
  only: the ovstage path warns and renders synchronously, because each of its scene writes must
  first drain any render still in flight (OVRTX retains a single committed snapshot that renders
  read in place), which leaves too little overlap to pay off. Asynchronous ovstage rendering is
  postponed to a follow-up; deeper queues on the legacy path are possible future work.
