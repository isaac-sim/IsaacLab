Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonVBDManager`,
  :class:`~isaaclab_newton.physics.VBDSolverCfg`, and
  :class:`~isaaclab_newton.physics.NewtonSoftContactCfg` to the core Newton
  physics package.

Changed
^^^^^^^

* Changed the VBD soft-contact damping default from ``0.01`` to ``10.0``
  to match Newton. Set ``soft_contact_kd=0.01`` explicitly through
  :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg` to retain the
  previous behavior.

Deprecated
^^^^^^^^^^

* Deprecated ``VBDSolverCfg.model_cfg`` in favor of
  :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg`. Move
  :class:`~isaaclab_newton.physics.NewtonSoftContactCfg` to the outer
  :class:`~isaaclab_newton.physics.NewtonCfg`.
