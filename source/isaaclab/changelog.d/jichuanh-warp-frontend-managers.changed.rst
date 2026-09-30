* Changed :class:`~isaaclab.envs.mdp.curriculums.modify_term_cfg` to hand the modified term configuration back to
  the manager's ``set_term_cfg`` after writing a value, so managers that keep values read from the configuration,
  such as the recorded stages of the Warp managers, use the new value.
