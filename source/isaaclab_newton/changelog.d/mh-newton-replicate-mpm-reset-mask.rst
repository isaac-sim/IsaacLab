Fixed
^^^^^

* Fixed environment resets raising ``ValueError: world_mask has shape ...`` under the implicit MPM
  solver. Newton's :class:`newton.solvers.SolverImplicitMPM` gained a ``reset`` that only accepts a
  per-world mask when it runs one FEM environment per world, so the MPM manager no longer forwards
  the reset and leaves particle history untouched, as it did before the solver gained ``reset``.
