Added
^^^^^

* Added PhysX implementations of
  :meth:`~isaaclab.assets.BaseArticulation.get_jacobians`,
  :meth:`~isaaclab.assets.BaseArticulation.get_mass_matrix`, and
  :meth:`~isaaclab.assets.BaseArticulation.get_gravity_compensation_forces`
  as one-line passthroughs to the corresponding
  ``physx.ArticulationView`` methods, plus the
  :attr:`~isaaclab_physx.assets.Articulation.num_jacobi_joints`
  override that returns ``num_joints + 6`` for floating-base
  articulations.
