Added
^^^^^

* Added :attr:`~isaaclab_contrib.deformable.VBDSolverCfg.rigid_body_particle_contact_buffer_size`
  to size each body's particle, edge, and face soft-contact list. Contacts past the buffer are
  dropped from the body's reaction list, which can inject energy.
