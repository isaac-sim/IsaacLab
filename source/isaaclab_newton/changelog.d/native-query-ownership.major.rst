Changed
^^^^^^^

* Moved Newton camera and ray-cast consumers to the simulation-owned native backend. Consumers
  requested transforms and geometry directly through SDP and shared native BVH refits while owning
  their individual query graphs. ``NewtonWarpRendererCfg.use_cuda_graph`` controlled camera query
  capture independently of physics capture.
* **Breaking:** Kept physics settings in ``NewtonBackendCfg(physics_cfg=...)`` and moved native
  builders, geometry ranges, and allocation arguments to ``NewtonBackend(cfg, builder=..., device=...)``.
  Registered completed resources with ``sim.register_backend(cfg, backend)``; consumers used
  ``sim.get_backend(cfg)`` instead of constructing native models.

Removed
^^^^^^^

* **Breaking:** Removed ``NewtonManager.get_state()`` and ``update_visualization_state()``.
  Acquire the native backend through ``sim.get_backend(backend_cfg)`` and request
  ``SceneDataProvider.get_transforms()`` and ``get_geometry_points()`` directly instead.
