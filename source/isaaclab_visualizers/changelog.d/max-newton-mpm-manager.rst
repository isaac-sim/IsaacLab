Changed
^^^^^^^

* :class:`~isaaclab_visualizers.newton.NewtonVisualizer` now skips Newton's
  per-frame active-particle compaction (two device-to-host reads per render)
  when an MPM model's static particle flags are all active, and re-uploads the
  particle color buffer only when the point count grows or the configured color
  changes.
