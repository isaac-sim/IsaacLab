Changed
^^^^^^^

* **Breaking:** Built foreign-physics Newton render representations from declared clone-plan sources and shared roots instead of
  discovering the completed stage in model getters. Standalone render workflows must declare their assets and camera
  configurations before cloning; native model allocation occurred after physics initialization.
* Unified physics and render-only Newton construction in the clone pipeline, retaining solver-specific imports only
  for Newton physics. Render-only deformables respected context row routing and plans without position offsets.
