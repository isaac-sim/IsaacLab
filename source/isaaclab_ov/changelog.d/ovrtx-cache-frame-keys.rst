Changed
^^^^^^^

* Cached OVRTX render-var frame keys during camera initialization, avoiding repeated version checks and
  path construction during frame processing. Existing camera output behavior and APIs remained unchanged.
