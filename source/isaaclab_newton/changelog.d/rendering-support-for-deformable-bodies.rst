Added
^^^^^

* Added :meth:`~isaaclab_newton.physics.NewtonManager.particles_dirty`,
  :meth:`~isaaclab_newton.physics.NewtonManager.transforms_dirty`, and
  :meth:`~isaaclab_newton.physics.NewtonManager.clear_particles_dirty`
  so kitless render consumers can coordinate deformable mesh sync with
  :meth:`~isaaclab_newton.physics.NewtonManager.sync_particles_to_usd`.
* Added post-step particle dirty marking for surface and volume deformable
  registry entries.

Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.sync_particles_to_usd`
  clearing the particle dirty flag when no Fabric mesh or points prims were
  updated, which blocked OVRTX deformable rendering in kitless mode.
