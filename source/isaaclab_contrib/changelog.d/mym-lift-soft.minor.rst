Added
^^^^^

* Added :attr:`~isaaclab_contrib.deformable.VBDSolverCfg.rigid_body_particle_contact_buffer_size`
  to size the per-body particle contact list. Contacts past the buffer are dropped from the body's
  reaction list, which pushes the particles without recoiling the body and injects energy.
