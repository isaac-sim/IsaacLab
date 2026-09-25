Fixed
^^^^^

* Fixed :meth:`~isaaclab_physx.assets.Articulation.set_fixed_tendon_position_limit_index` and
  :meth:`~isaaclab_physx.assets.Articulation.set_fixed_tendon_position_limit_mask` rejecting position limits in
  the layout the asset stores and reports them in. They now take ``wp.vec2f`` arrays of shape
  (num_envs, num_fixed_tendons), or torch tensors with a trailing dimension of 2, like the joint position limits.
* Corrected tendon configuration documentation that incorrectly described tendons as a PhysX-only feature.
