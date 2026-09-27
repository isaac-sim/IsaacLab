Changed
^^^^^^^

* Extended ``SimulationContext.get_or_create_backend(cfg)`` to own Python resources as well as native
  backends. Teardown called ``close()`` when provided and otherwise released the registry reference.
  Resource construction remained ``cfg.class_type(cfg)`` without backend-specific context fields.
* Reused SDP transform mappings for matching source and destination layouts, allowing independent
  consumers to share converted buffers without retaining SDP bindings on a native backend.
* Added ``RayCasterCfg.use_cuda_graph`` for Newton query execution independently of physics capture.
  Set it to ``False`` to execute Newton ray casts eagerly.

Fixed
^^^^^

* Released rendering bindings on physics stop so a hard reset reinitialized consumers against
  the replacement native resource.
