* Fixed :meth:`~isaaclab_physx.assets.Articulation.write_root_link_velocity_to_sim_index` and
  :meth:`~isaaclab_physx.assets.Articulation.write_root_link_velocity_to_sim_mask` writing the link velocity into
  PhysX, which stores the center-of-mass velocity. A root whose center of mass is offset from its link frame now
  moves with the written link velocity.
