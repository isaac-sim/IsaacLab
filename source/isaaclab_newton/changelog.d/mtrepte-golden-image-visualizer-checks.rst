Fixed
^^^^^

* Fixed stale ``_scene_data_mapping`` in :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  being reused after the visualization model was rebuilt for a stage with a different body count
  (e.g. switching from a 4-env tiled capture to a 1-env viewport capture within the same process).
  The mapping is now invalidated when its length does not match the current model's body count,
  preventing wrong body transforms from being written into the shadow ``state_0``.
