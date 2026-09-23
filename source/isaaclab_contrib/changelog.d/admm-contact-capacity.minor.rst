Added
^^^^^

* Added ``CouplerAdmmCfg.contact_max_triangle_pairs`` and
  ``contact_reduction_hashtable_size_factor`` to configure the internal ADMM collision
  buffers independently of the outer Newton collision pipeline. Defaults preserved
  Newton's existing allocation and contact matching behavior.

Fixed
^^^^^

* Rejected internal ADMM triangle-pair capacities at or above ``2**20`` during
  configuration validation when rigid contact matching was ``"latest"`` or
  ``"sticky"``, while allowing larger capacities when matching was disabled.
