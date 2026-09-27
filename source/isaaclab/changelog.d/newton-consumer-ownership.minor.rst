Changed
^^^^^^^

* Extended ``SimulationContext.get_or_create_backend(cfg)`` to own Python resources as well as native
  backends. Teardown called ``cfg.close(resource)``, requiring ``resource.close()`` by default.
  Configurations for Python-owned data explicitly overrode cleanup instead of guessing whether it existed.
  Resource construction remained ``cfg.class_type(cfg)`` without backend-specific context fields.
* Reused SDP transform mappings for matching source and destination layouts, allowing independent
  consumers to share converted buffers without retaining SDP bindings on a native backend. Matching
  unique native and consumer paths needed no index-map allocation.
* Added ``RayCasterCfg.use_cuda_graph`` for Newton query execution independently of physics capture.
  Set it to ``False`` to execute Newton ray casts eagerly.

Fixed
^^^^^

* Released rendering bindings on physics stop so a hard reset reinitialized consumers against
  the replacement native resource.
