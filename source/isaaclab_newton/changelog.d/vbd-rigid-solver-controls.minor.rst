Added
^^^^^

* Exposed ``rigid_avbd_alpha``, ``rigid_contact_hard``, and
  ``rigid_body_contact_buffer_size`` on ``VBDSolverCfg`` while preserving Newton's
  existing defaults. Added ``rigid_compliant_alm`` for explicit migration from
  Newton 1.6's deprecated legacy rigid-contact path; users can opt in with
  ``rigid_compliant_alm=True`` and retune finite material stiffness as needed.
