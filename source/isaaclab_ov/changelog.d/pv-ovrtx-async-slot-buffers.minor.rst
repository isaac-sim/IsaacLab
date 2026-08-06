Added
^^^^^

* Added an opt-in asynchronous OVRTX render path controlled by
  :attr:`~isaaclab_ov.renderers.ovrtx_renderer_cfg.OVRTXRendererCfg.async_rendering`
  (default ``False``). ``False`` renders synchronously (unchanged behavior); ``True``
  enables asynchronous rendering, so rendering overlaps with simulation and Python work
  and camera outputs arrive one frame later, improving throughput. The first frame after
  (re)initialization is waited on and consumed immediately, so the first camera read returns a
  valid frame rather than the zero-initialized output buffer; later frames are pipelined.
* Added the ``OVRTX_ASYNC_RENDERING`` environment variable to override
  :attr:`~isaaclab_ov.renderers.ovrtx_renderer_cfg.OVRTXRendererCfg.async_rendering` for tests.
* Added the ``OVRTX_NUM_BUFFERS`` environment variable to configure the number of asynchronous
  ``step_async`` renders kept in flight (the render queue depth, default ``2``). Larger values
  overlap more simulation with rendering at the cost of extra frames of camera latency; values
  below ``2`` are clamped to ``2``.
* Asynchronous rendering is available on both the legacy and ovstage scene-ownership paths. Under
  ovstage the renderer borrows the stage's storage, so each ovstage scene write first settles any
  render still in flight; rendering overlaps simulation, inference and product reads, and only the
  next frame's first write closes the window.
