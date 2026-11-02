* **Breaking:** Changed :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` and
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state` (and their backend counterparts) to select
  environments by mask only. Removed their ``env_ids`` argument; pass a boolean ``env_mask`` instead, and
  ``articulation_mask`` to select view-local articulations.
* Changed the ``*_index`` state writes of Newton assets to flag the written environments into a scratch mask that
  the backend clears as it reads it, so the backend's reset and FK flagging is mask-only.
* Changed asset data classes to invalidate only their derived caches.
* Changed rigid object collection ``*_mask`` state writes to use dedicated mask kernels instead of converting masks to
  indices, and limited FK invalidation to the selected objects.
* Avoided refreshing derived state while obtaining authoritative write destinations.
