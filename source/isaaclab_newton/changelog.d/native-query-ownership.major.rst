Changed
^^^^^^^

* Moved Newton camera and ray-cast consumers to the simulation-owned native backend. Consumers
  requested transforms and geometry directly through SDP and shared native BVH refits while owning
  their individual query graphs. ``NewtonWarpRendererCfg.use_cuda_graph`` controlled camera query
  capture independently of physics capture.

Removed
^^^^^^^

* **Breaking:** Removed ``NewtonManager.get_state()`` and ``update_visualization_state()``.
  Acquire the native backend through ``sim.get_or_create_backend(backend_cfg)`` and request
  ``SceneDataProvider.get_transforms()`` and ``get_geometry_points()`` directly instead.
