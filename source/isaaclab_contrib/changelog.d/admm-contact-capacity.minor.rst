Added
^^^^^

* Added ``CouplerAdmmCfg.contact_max_triangle_pairs`` and
  ``contact_reduction_hashtable_size_factor`` to configure the internal ADMM collision
  buffers independently of the outer Newton collision pipeline, preserving Newton's
  defaults. Validation rejected triangle-pair capacities at or above ``2**20`` with
  ``"latest"`` or ``"sticky"`` matching and explicit overrides unsupported by Newton.
