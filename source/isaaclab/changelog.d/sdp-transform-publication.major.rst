Changed
^^^^^^^

* **Breaking:** Added ``transforms_dirty`` to scene-data backends. Custom backends must initialize
  it to ``True`` and set it after native pose writes or buffer swaps; SDP reads the existing
  ``transforms`` property before clearing it. Renderers now request shared, read-only arrays through
  ``SceneDataProvider.request_transforms``; matching layouts alias native data and other layouts
  convert once per publication. The existing caller-owned ``get_transforms`` API remained available.
* Moved rigid Fabric conversion and propagation into SDP, preserving the engine-owned Fabric
  path for native PhysX. Transform freshness no longer depended on the physics-step counter;
  ``RenderContext.reset_scene_state_cadence`` remained available for geometry updates.
