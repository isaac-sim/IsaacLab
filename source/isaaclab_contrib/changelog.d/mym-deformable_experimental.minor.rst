Added
^^^^^

* Added :mod:`isaaclab_contrib.deformable` with contributed Newton deformable
  asset and VBD solver support, including
  :class:`~isaaclab_contrib.deformable.DeformableObject`,
  :class:`~isaaclab_contrib.deformable.VBDSolverCfg`,
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg`, and
  :class:`~isaaclab_contrib.deformable.CoupledFeatherstoneVBDSolverCfg` for
  one- and two-way rigid-deformable coupling.
* Added :class:`~isaaclab_contrib.deformable.NewtonModelCfg` for shared Newton
  deformable contact parameters.
* Added Newton deformable coupling documentation with Franka soft-body lift
  tuning guidance for
  :class:`~isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg` and
  :class:`~isaaclab_contrib.deformable.NewtonModelCfg`.
