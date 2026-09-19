Changed
^^^^^^^

* Shared Newton model, state, and scene-query resources through the simulation backend registry.
  Newton rendering resources were built from the clone plan instead of discovering the completed
  stage on first access. Existing renderer configuration and physics accessors remained supported.
