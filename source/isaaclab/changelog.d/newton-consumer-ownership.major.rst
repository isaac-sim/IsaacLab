Changed
^^^^^^^

* Extended ``SimulationContext.get_or_create_backend(cfg)`` to own Python resources as well as native
  backends. Resources declared through ``BackendCfg`` implemented ``close()``; plain construction cfgs
  shared Python-owned data released with the registry reference.
  Resource construction remained ``cfg.class_type(cfg)`` without backend-specific context fields.
* Reused SDP transform mappings for matching source and destination layouts, allowing independent
  consumers to share converted buffers without retaining SDP bindings on a native backend. Matching
  unique native and consumer paths needed no index-map allocation. Binding rejected duplicate named
  paths and destinations without native publications, including when the publication was empty.
* Added ``RayCasterCfg.use_cuda_graph`` for Newton query execution independently of physics capture.
  Set it to ``False`` to execute Newton ray casts eagerly.
* **Breaking:** Replaced ``VisualizerCfg.streaming_cam_renderer`` and its deprecated
  ``tiled_cam_renderer`` alias with ``streaming_cam_renderer_cfg``. Pass a renderer configuration,
  such as ``NewtonWarpRendererCfg()`` or ``OVRTXRendererCfg()``, instead of a renderer nickname.
  Its ``class_type`` accepted either an implementation class or a resolvable string.

Fixed
^^^^^

* Released rendering bindings on physics stop so a hard reset reinitialized consumers against
  the replacement native resource.
