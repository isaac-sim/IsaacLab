* Fixed :class:`~isaaclab.envs.mdp.modify_env_param` and :class:`~isaaclab.envs.mdp.modify_term_cfg` raising an
  error for an address that ends at an indexed element, such as ``"commands.base_velocity.ranges.lin_vel_x[1]"``.
  Setting an element of a tuple now replaces the tuple on its owner.
