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
* Fixed cable explosion under the Kit visualizer by overriding
  :meth:`~isaaclab_contrib.deformable.vbd_manager.NewtonVBDManager.forward` to
  mask out cable articulations. Newton's ``eval_fk`` has no
  :attr:`newton.JointType.CABLE` case and was collapsing rod segments onto
  their parent anchors every time Kit triggered a pre-render FK pass.
* Fixed curved cables (e.g. loaded via :class:`~isaaclab.sim.UsdFileCfg`)
  exploding on the first sim step. The unmasked ``eval_fk`` at the end of
  :meth:`~isaaclab_newton.physics.NewtonManager.start_simulation` was
  corrupting cable ``state_0.body_q`` (same ``JointType.CABLE`` fall-through
  as above), so non-collinear cable layouts started the simulation collapsed
  onto the root segment's local +Z axis.
  :meth:`~isaaclab_contrib.deformable.vbd_manager.NewtonVBDManager.start_simulation`
  now rebuilds ``state_0`` / ``state_1`` from ``model.state()`` after the base
  finalize step, then re-runs the masked :meth:`forward` to seed non-cable
  ``body_q`` without touching cables.
