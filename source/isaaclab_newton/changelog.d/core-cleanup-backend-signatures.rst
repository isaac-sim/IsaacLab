Fixed
^^^^^

* Fixed the deprecated :meth:`~isaaclab_newton.assets.Articulation.write_joint_friction_coefficient_to_sim` and
  :meth:`~isaaclab_newton.assets.Articulation.write_joint_friction_to_sim` always raising ``TypeError``. The Newton
  override passed arguments that :meth:`~isaaclab_newton.assets.Articulation.write_joint_friction_coefficient_to_sim_index`
  does not accept; it is removed in favor of the base class implementation.
