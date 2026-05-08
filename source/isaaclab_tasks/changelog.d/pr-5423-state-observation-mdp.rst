Added
^^^^^

* Added explicit GR1T2 and Unitree G1 pick-place robot link pose and velocity
  MDP helpers as replacements for packed robot link state observations.

Changed
^^^^^^^

* Changed Dexsuite orientation tracking rewards to read root link orientation
  directly instead of slicing packed root state tensors.

Deprecated
^^^^^^^^^^

* Deprecated
  :func:`~isaaclab_tasks.manager_based.manipulation.pick_place.mdp.observations.get_all_robot_link_state`
  in favor of
  :func:`~isaaclab_tasks.manager_based.manipulation.pick_place.mdp.observations.get_all_robot_link_pose`
  and
  :func:`~isaaclab_tasks.manager_based.manipulation.pick_place.mdp.observations.get_all_robot_link_velocity`.
