Fixed
^^^^^

* Fixed :func:`isaaclab.sim.build_simulation_context` silently ignoring the
  ``device`` kwarg when ``sim_cfg`` is also provided. Most test callers pass
  both kwargs together; the helper now applies the explicit ``device`` over
  ``sim_cfg.device`` so the caller's choice wins. Without this, warp kernel
  launches in :mod:`isaaclab_newton.assets.articulation` raised device
  mismatch errors on non-default GPUs (``env_ids`` allocated on the test's
  device while the articulation's resolved device came from the untouched
  ``sim_cfg`` default ``cuda:0``).
