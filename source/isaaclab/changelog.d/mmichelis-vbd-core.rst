Changed
^^^^^^^

* Changed the ``newton_vbd`` launcher backend to use
  :class:`~isaaclab_newton.physics.VBDSolverCfg` from core. Custom launchers
  should import the solver configuration from ``isaaclab_newton.physics``
  instead of ``isaaclab_contrib.deformable``.
