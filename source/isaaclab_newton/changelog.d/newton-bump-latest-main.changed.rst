* Updated the Newton dependency to the current ``main`` revision, which adds the optional OptiX camera backend.
  Newton renamed Kamino's DVI forward-dynamics solver to SCC, so
  :class:`~isaaclab_newton.physics.KaminoDVISolverCfg` now configures Newton's SCC solver with unchanged fields.
