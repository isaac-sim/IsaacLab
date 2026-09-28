Changed
^^^^^^^

* Moved Newton camera and ray-cast consumers to the simulation-owned native backend. Consumers
  requested transforms and geometry directly through SDP and shared native BVH refits while owning
  their individual query graphs. ``NewtonWarpRendererCfg.use_cuda_graph`` controlled camera query
  capture independently of physics capture. Deformable triangle-mesh rendering stayed eager because
  native mesh updates required host reads and allocations.
  Stateless query and capture functions lived in ``NewtonQueries`` alongside ``NewtonManager``;
  the container retained no model or scheduling state.
* Applied ray-cast BVH requirements to the shared builder before finalization, including when
  another consumer acquired the builder first.
* **Breaking:** Replaced the native builder input in ``NewtonBackendCfg`` with the selected physics
  configuration. Acquire builders through ``NewtonBuilderCfg(physics_cfg=sim.cfg.physics)``
  and finalized models through ``NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)``
  using ``sim.get_or_create_backend(cfg)``. Non-Newton physics selected a render-only representation;
  model allocation followed cloning rather than occurring inside it. Kept construction and runtime
  cfgs independent: closing a backend released its native buffers but retained the builder for hard reset.

Removed
^^^^^^^

* **Breaking:** Removed ``NewtonManager.set_builder()``. Acquire and populate the shared builder with
  ``sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))`` before ``sim.reset()``.
* **Breaking:** Removed ``NewtonManager.get_state()`` and ``update_visualization_state()``.
  Acquire the native backend through ``sim.get_or_create_backend(backend_cfg)`` and request
  ``SceneDataProvider.get_transforms()`` and ``get_geometry_points()`` directly instead.
* **Breaking:** Removed ``NewtonManager.sync_transforms_to_fabric()`` and ``sync_transforms_to_usd()``.
  Kit/Isaac RTX rendering already requested published poses through SDP. Custom Fabric consumers
  should call ``FabricBackend.update_transforms(provider)`` instead of asking the physics manager to render.
