Added
^^^^^

* Added :class:`~isaaclab_contrib.cable.CableObject` (subclass of
  :class:`~isaaclab_newton.assets.articulation.Articulation`) and
  :class:`~isaaclab_contrib.cable.CableObjectCfg` for runtime cable assets.
* Added :class:`~isaaclab_contrib.cable.CableRegistryEntry`,
  :func:`~isaaclab_contrib.cable.add_cable_entry_to_builder`,
  :func:`~isaaclab_contrib.cable.add_registered_cables_to_builder`, and
  :func:`~isaaclab_contrib.cable.install_cable_builder_hooks` —
  the replicate-hook plumbing that mirrors the deformable contrib pattern.

Fixed
^^^^^

* Fixed an ``AttributeError`` in
  :meth:`~isaaclab_contrib.deformable.vbd_manager.NewtonVBDManager._simulate_physics_only`
  triggered in cable-only scenes (zero particles). Newton's ``SolverVBD`` skips
  ``_init_particle_system`` for zero-particle scenes and leaves
  ``particle_enable_self_contact`` unset; the manager now reads it with
  ``getattr(..., False)`` to default to no-self-contact.
* Fixed Kit / Fabric viewport sync for Newton cables by updating
  ``UsdGeomBasisCurves`` points from Newton cable body transforms at render
  cadence.
