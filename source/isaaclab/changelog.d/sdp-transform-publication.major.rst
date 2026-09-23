Changed
^^^^^^^

* **Breaking:** Added ``transforms_dirty`` to scene-data backends. Custom backends must initialize
  it to ``True`` and set it after native pose writes or buffer swaps; SDP reads the existing
  ``transforms`` property before clearing it. Renderers now request shared, read-only arrays through
  ``SceneDataProvider.request_transforms``; matching layouts alias native data and other layouts
  convert once per publication. The existing caller-owned ``get_transforms`` API remained available.
* Moved rigid Fabric conversion and GPU hierarchy propagation into SDP, preserving the engine-owned
  Fabric path for native PhysX and authored scale for converted poses. Converted rigid destinations
  became Fabric-only reset-stack roots so nested bodies retained their absolute physics poses.
  Transform freshness no longer depended on the physics-step counter;
  ``RenderContext.reset_scene_state_cadence`` remained available for geometry updates.
* Used a Warp struct for Fabric transform bindings. Backported Fabric struct kernel arguments
  in memory when Kit loaded older Warp, without changing installed dependency files.
