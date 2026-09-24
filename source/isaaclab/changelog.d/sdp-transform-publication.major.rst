Changed
^^^^^^^

* **Breaking:** Added ``transforms_version`` to scene-data backends. Custom backends must initialize
  it to zero and increment it after native pose writes or buffer swaps. SDP reads the existing
  ``transforms`` property through ``get_transforms(output_format)`` without resetting the version. Backends
  publishing multiple native formats may override that method and ``native_transform_formats``.
  ``SceneDataProvider.get_transforms`` bound shared,
  read-only arrays by default: matching layouts aliased native data and other layouts converted
  once per publication. Callers requiring their own writable or preallocated arrays must pass
  ``allow_passthrough=False``; this wrote directly into the supplied arrays without a staging copy.
* Routed rigid Fabric conversion through SDP while the Kit rendering integration owned destination binding and
  GPU hierarchy propagation, preserving native PhysX publication and authored scale. Converted rigid destinations
  became Fabric-only reset-stack roots so nested bodies retained their absolute physics poses.
  Transform freshness no longer depended on the physics-step counter;
  ``RenderContext.reset_scene_state_cadence`` remained available for geometry updates.
  ``SimulationContext.fabric_transforms_cfg`` declared the shared Kit destination without allocating bindings.
* Used native Warp structs for Fabric transform bindings, relying on the project-managed Warp
  dependency selected by Isaac Lab's Kit launch configuration.
