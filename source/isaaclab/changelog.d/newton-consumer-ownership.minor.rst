Changed
^^^^^^^

* Added ``SimulationContext.register_backend(cfg, backend)`` for completed native resources and
  ``get_backend(cfg)`` for lookup without construction. Consumers retained declarative configurations;
  ``SimulationContext`` did not expose backend-specific configuration fields.
* Reused SDP transform mappings for matching source and destination layouts, allowing independent
  consumers to share converted buffers without retaining SDP bindings on a native backend.
* Added ``RayCasterCfg.use_cuda_graph`` for Newton query execution independently of physics capture.
  Set it to ``False`` to execute Newton ray casts eagerly.

Fixed
^^^^^

* Released rendering bindings on physics stop so a hard reset reinitialized consumers against
  the replacement native resource.
