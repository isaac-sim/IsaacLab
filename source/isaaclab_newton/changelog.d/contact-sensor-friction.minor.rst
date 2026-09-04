Added
^^^^^

* Added aggregate and filtered friction force outputs to the Newton contact sensor, including
  ``net_friction_forces_w_history`` and ``friction_force_matrix_w_history``.

Changed
^^^^^^^

* Changed Newton contact sensor normal-force outputs to exclude friction. ``net_forces_w`` and
  ``force_matrix_w`` expose Newton's total contact force (normal + friction) without a warning.
  ``friction_forces_w`` exposes the aggregate friction force (``net_friction_forces_w``) without
  a warning. Reconstruct components from ``net_normal_forces_w`` / ``net_friction_forces_w`` and
  the corresponding matrix properties when needed.
