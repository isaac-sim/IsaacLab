Changed
^^^^^^^

* Changed standalone VBD classes to compatibility aliases of
  :class:`~isaaclab_newton.physics.NewtonVBDManager` and
  :class:`~isaaclab_newton.physics.VBDSolverCfg`. Import these classes from
  ``isaaclab_newton.physics`` for new code.
* Changed the VBD soft-contact damping default from ``0.01`` to ``10.0``
  to match Newton. Set ``soft_contact_kd=0.01`` explicitly through
  :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg` to retain the
  previous behavior.

Deprecated
^^^^^^^^^^

* Deprecated ``isaaclab_contrib.deformable.vbd_manager.NewtonVBDManager`` and
  ``isaaclab_contrib.deformable.VBDSolverCfg`` in favor of their
  :mod:`isaaclab_newton.physics` aliases.
* Deprecated ``isaaclab_contrib.deformable.NewtonModelCfg`` in favor of
  :class:`~isaaclab_newton.physics.NewtonSoftContactCfg`. Move the
  configuration from ``solver_cfg.model_cfg`` to
  :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg`.
