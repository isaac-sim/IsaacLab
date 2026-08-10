Added
^^^^^

* Added an opt-in asynchronous OVRTX render path controlled by
  :attr:`~isaaclab_ov.renderers.ovrtx_renderer_cfg.OVRTXRendererCfg.async_rendering`
  (default ``False``). When enabled, rendering overlaps simulation and Python work and camera
  outputs arrive one frame later, improving throughput. The first frame after (re)initialization
  is consumed immediately so the first camera read returns a valid frame; later frames are
  pipelined. Available on both the legacy and ovstage scene-ownership paths; under ovstage each
  scene write first settles any render still in flight, since the renderer borrows the stage's
  storage.
* Added the ``OVRTX_ASYNC_RENDERING`` environment variable to override
  :attr:`~isaaclab_ov.renderers.ovrtx_renderer_cfg.OVRTXRendererCfg.async_rendering` for tests.
* Added the ``OVRTX_NUM_BUFFERS`` environment variable to configure the render queue depth: the
  number of asynchronous renders kept in flight (default ``2``, values below ``2`` are clamped).
  Larger values overlap more simulation with rendering at the cost of extra frames of camera
  latency.
