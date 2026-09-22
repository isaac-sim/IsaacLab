Changed
^^^^^^^

* **Breaking:** Added dirty transform publications to scene-data backends. Custom backends must
  implement ``transform_publication`` with a ``SceneDataPublication`` and mark it dirty after
  native pose writes or buffer swaps. Renderers now request shared, read-only arrays through
  ``SceneDataProvider.request_transforms``; matching layouts alias native data and other layouts
  convert once per publication. The existing caller-owned ``get_transforms`` API remained available.
* Moved rigid Fabric conversion and propagation into SDP, preserving the engine-owned Fabric
  path for native PhysX. Transform freshness no longer depended on the physics-step counter;
  ``RenderContext.reset_scene_state_cadence`` remained available for geometry updates.
