Fixed
^^^^^

* Fixed :meth:`~isaaclab_ov.assets.Articulation.set_fixed_tendon_position_limit_index` and
  :meth:`~isaaclab_ov.assets.Articulation.set_fixed_tendon_position_limit_mask` rejecting ``wp.vec2f`` arrays,
  the layout of :attr:`~isaaclab_ov.assets.ArticulationData.fixed_tendon_pos_limits`, and passing a float to the
  kernel instead of raising ``ValueError``.
