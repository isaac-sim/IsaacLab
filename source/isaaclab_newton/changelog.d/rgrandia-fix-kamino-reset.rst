Fixed
^^^^^

* Fixed environment resets writing reconciled state into the wrong
  double-buffered simulation state when ``use_cuda_graph`` was disabled. With an
  odd number of substeps the canonical state flipped buffers each step, so reset
  writes landed in the stale buffer and left reset environments in an
  inconsistent state for solvers that keep separate input/output states (e.g.
  :class:`~isaaclab_newton.physics.NewtonKaminoManager`).
* Fixed forward kinematics for the Kamino solver so that
  :meth:`~isaaclab_newton.physics.NewtonManager.forward` reconciles body state
  through a solver-specialized hook instead of Newton's articulated ``eval_fk``,
  which produced incorrect poses for closed-loop systems on environment resets.
