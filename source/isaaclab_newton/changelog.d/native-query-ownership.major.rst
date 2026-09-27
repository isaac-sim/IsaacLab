Changed
^^^^^^^

* Moved Newton camera and ray-cast consumers to the simulation-owned native backend. Consumers
  requested transforms and geometry directly through SDP and shared native BVH refits while owning
  their individual query graphs. ``NewtonWarpRendererCfg.use_cuda_graph`` controlled camera query
  capture independently of physics capture.
* **Breaking:** Replaced the native builder input on ``NewtonBackendCfg`` with a ``NewtonBuilderCfg``
  dependency. Acquired builders and finalized models through ``sim.get_or_create_backend(cfg)``.
  Use ``NewtonBuilderCfg(physics_cfg=sim.cfg.physics)`` for Newton physics, or ``physics_cfg=None``
  for a foreign-physics rendering representation, then declare
  ``NewtonBackendCfg(builder_cfg=builder_cfg, device=sim.device)`` for its native model and state.

Removed
^^^^^^^

* **Breaking:** Removed ``NewtonManager.set_builder()``. Acquire and populate the shared builder with
  ``sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))`` before ``sim.reset()``.
* **Breaking:** Removed ``NewtonManager.get_state()`` and ``update_visualization_state()``.
  Acquire the native backend through ``sim.get_or_create_backend(backend_cfg)`` and request
  ``SceneDataProvider.get_transforms()`` and ``get_geometry_points()`` directly instead.
