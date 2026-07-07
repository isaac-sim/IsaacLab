Added
^^^^^

* Added ``rigid_body_contact_buffer_size`` to
  :class:`~isaaclab_contrib.deformable.VBDSolverCfg`.

Fixed
^^^^^

* Fixed :class:`~isaaclab_contrib.deformable.NewtonVBDManager` initialization
  and stepping for rigid-only models by preparing the required body coloring
  and skipping particle BVH rebuilds.
