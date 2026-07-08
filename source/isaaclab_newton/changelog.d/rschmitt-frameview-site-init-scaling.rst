Fixed
^^^^^

* Fixed quadratic (``O(num_envs^2)``) startup scaling in
  :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` when a frame resolves to a
  per-environment body path (e.g. a body-mounted camera). Replicated body patterns are now
  resolved against Newton body labels through an exact lookup instead of a full regex scan per
  environment, reducing simulation-start time for camera-heavy scenes at high environment counts
  (8192 environments dropped from ~29 min to seconds).
