Changed
^^^^^^^

* **Breaking:** Built foreign-physics Newton render representations from declared clone-plan sources and shared roots instead of
  discovering the completed stage in model getters. Standalone render workflows must declare their assets and camera
  configurations before cloning; native model allocation occurred after physics initialization.
