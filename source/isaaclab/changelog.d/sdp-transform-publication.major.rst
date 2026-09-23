Changed
^^^^^^^

* **Breaking:** Added ``transforms_dirty`` to scene-data backends. Custom backends must initialize
  it to ``True`` and set it after native pose writes or buffer swaps; SDP reads the existing
  ``transforms`` property before clearing it. ``SceneDataProvider.get_transforms`` bound shared,
  read-only arrays by default: matching layouts aliased native data and other layouts converted
  once per publication. Callers requiring their own writable or preallocated arrays must pass
  ``allow_passthrough=False``; this wrote directly into the supplied arrays without a staging copy.
* Moved rigid Fabric conversion and GPU hierarchy propagation into SDP, preserving the engine-owned
  Fabric path for native PhysX and authored scale for converted poses. Converted rigid destinations
  became Fabric-only reset-stack roots so nested bodies retained their absolute physics poses.
  Transform freshness no longer depended on the physics-step counter;
  ``RenderContext.reset_scene_state_cadence`` remained available for geometry updates.
* Used native Warp structs for Fabric transform bindings, relying on the project-managed Warp
  dependency selected by Isaac Lab's Kit launch configuration.
