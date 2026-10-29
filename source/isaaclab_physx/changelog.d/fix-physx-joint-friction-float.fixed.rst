* Fixed :meth:`~isaaclab_physx.assets.Articulation.write_joint_friction_coefficient_to_sim_index`
  and its mask variant to broadcast scalar coefficients directly in the fused Warp kernel, including integer zero,
  without allocating expanded input arrays.
