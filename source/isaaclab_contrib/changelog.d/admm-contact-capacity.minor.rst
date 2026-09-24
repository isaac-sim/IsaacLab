Added
^^^^^

* Added ``CouplerAdmmCfg.contact_max_triangle_pairs`` and
  ``contact_reduction_hashtable_size_factor`` to configure the internal ADMM collision
  buffers independently of the outer Newton collision pipeline. Defaults preserved
  Newton's existing allocation and contact matching behavior. Configuration
  validation rejected triangle-pair capacities at or above ``2**20`` when rigid
  contact matching was ``"latest"`` or ``"sticky"``, while allowing larger
  capacities when matching was disabled.
