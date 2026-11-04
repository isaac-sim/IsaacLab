* **Breaking:** Removed ``Articulation.write_actuator_stiffness_to_sim`` and
  ``write_actuator_damping_to_sim``, deprecated in Isaac Lab 3.0 for removal in 3.1. Use
  ``randomize_actuator_gains`` for managed randomization, or
  ``isaaclab.actuators.newton.write_group_parameter`` with the ``"drive"`` component and
  ``"kp"`` or ``"kd"`` for direct native gain writes. The direct writer takes an actuator
  group name and group-local joint indices.
